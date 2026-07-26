from __future__ import annotations

from typing import Any

import itertools
import gymnasium
from packaging import version

import torch
import torch.nn as nn

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.utils import ScopedTimer

from .ppo_cfg import PPO_CFG
from ._utils import (
    any_rank_true,
    backoff_value_learning_rate,
    compute_value_loss_fp32,
    critic_guard_triggered,
    ensure_full_rollout,
    require_finite,
    require_finite_model,
    validate_ppo_setup,
    validate_scalar_output,
)


def compute_gae(
    *,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    truncated: torch.Tensor | None = None,
    discount_factor: float = 0.99,
    lambda_coefficient: float = 0.95,
) -> torch.Tensor:
    """Compute the Generalized Advantage Estimator (GAE).

    :param rewards: Rewards obtained by the agent.
    :param terminated: Signals to indicate that episodes have ended.
    :param truncated: Signals to indicate that episodes have ended because of a time limit.
    :param values: Values obtained by the agent.
    :param next_values: Next values obtained by the agent.
    :param discount_factor: Discount factor.
    :param lambda_coefficient: Lambda coefficient.

    :return: Generalized Advantage Estimator.
    """
    advantage = 0
    advantages = torch.zeros_like(rewards)
    # Both termination kinds are episode boundaries for the lambda trace. If time-limit
    # bootstrapping is enabled, the caller adds gamma * V(final_observation) to the timeout
    # reward before entering this function; the trace must still not cross into the reset
    # episode that follows it.
    done = terminated if truncated is None else terminated | truncated
    not_done = done.logical_not().to(dtype=rewards.dtype)
    memory_size = rewards.shape[0]

    # advantages computation
    for i in reversed(range(memory_size)):
        next_values = values[i + 1] if i < memory_size - 1 else next_values
        advantage = (
            rewards[i]
            - values[i]
            + discount_factor * not_done[i] * (next_values + lambda_coefficient * advantage)
        )
        advantages[i] = advantage
    # returns computation
    returns = advantages + values
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    return returns, advantages


class PPO(Agent):
    def __init__(
        self,
        *,
        models: dict[str, Model],
        memory: Memory | None = None,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        cfg: PPO_CFG | dict = {},
    ) -> None:
        """Proximal Policy Optimization (PPO).

        https://arxiv.org/abs/1707.06347

        :param models: Agent's models.
        :param memory: Memory to storage agent's data and environment transitions.
        :param observation_space: Observation space.
        :param state_space: State space.
        :param action_space: Action space.
        :param device: Data allocation and computation device. If not specified, the default device will be used.
        :param cfg: Agent's configuration.

        :raises KeyError: If a configuration key is missing.
        """
        self.cfg: PPO_CFG
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=PPO_CFG(**cfg) if isinstance(cfg, dict) else cfg,
        )

        # models
        self.policy = self.models.get("policy", None)
        self.value = self.models.get("value", None)
        validate_ppo_setup(
            name="PPO", cfg=self.cfg, policy=self.policy, value=self.value, memory=self.memory
        )

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
                if self.value is not None and self.policy is not self.value:
                    self.value.broadcast_parameters()

        # set up automatic mixed precision
        self._device_type = torch.device(self.device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self.cfg.mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)
        self.checkpoint_modules["grad_scaler"] = self.scaler

        # set up optimizer and learning rate scheduler
        if self.policy is not None and self.value is not None:
            # - optimizers
            optimizer_cfg = self.cfg.optimizer
            if isinstance(optimizer_cfg, str):
                try:
                    optimizer_cls = getattr(torch.optim, optimizer_cfg)
                except AttributeError as e:
                    raise ValueError(f"Unknown optimizer '{optimizer_cfg}' for PPO") from e
            else:
                optimizer_cls = optimizer_cfg
            optimizer_kwargs = dict(self.cfg.optimizer_kwargs)
            if self.policy is self.value:
                self.optimizer = optimizer_cls(self.policy.parameters(), lr=self.cfg.learning_rate[0], **optimizer_kwargs)
            else:
                self.optimizer = optimizer_cls(
                    [
                        {"params": self.policy.parameters(), "lr": self.cfg.learning_rate[0]},
                        {"params": self.value.parameters(), "lr": self.cfg.learning_rate[1]},
                    ],
                    lr=self.cfg.learning_rate[0],
                    **optimizer_kwargs,
                )
            self.checkpoint_modules["optimizer"] = self.optimizer
            # - learning rate schedulers
            self.scheduler = self.cfg.learning_rate_scheduler[0]
            if self.scheduler is not None:
                self.scheduler = self.cfg.learning_rate_scheduler[0](
                    self.optimizer, **self.cfg.learning_rate_scheduler_kwargs[0]
                )
                self.checkpoint_modules["scheduler"] = self.scheduler

        # set up preprocessors
        # - observations
        if self.cfg.observation_preprocessor:
            self._observation_preprocessor = self.cfg.observation_preprocessor(
                **self.cfg.observation_preprocessor_kwargs
            )
            self.checkpoint_modules["observation_preprocessor"] = self._observation_preprocessor
        else:
            self._observation_preprocessor = self._empty_preprocessor
        # - states
        if self.cfg.state_preprocessor:
            self._state_preprocessor = self.cfg.state_preprocessor(**self.cfg.state_preprocessor_kwargs)
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor
        # - values
        if self.cfg.value_preprocessor:
            self._value_preprocessor = self.cfg.value_preprocessor(**self.cfg.value_preprocessor_kwargs)
            self.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = self._empty_preprocessor

    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        """Initialize the agent.

        :param trainer_cfg: Trainer configuration.
        """
        super().init(trainer_cfg=trainer_cfg)
        self.enable_models_training_mode(False)

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="observations", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="states", size=self.state_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)

            self._tensors_names = ["observations", "states", "actions", "log_prob", "values", "returns", "advantages"]

        # create temporary variables needed for storage and computation
        self._current_next_observations = None
        self._current_next_states = None
        self._current_log_prob = None
        self._current_is_random = False
        self._rollout = 0
        self._rollout_consumed = False

    def act(
        self, observations: torch.Tensor, states: torch.Tensor | None, *, timestep: int, timesteps: int
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Process the environment's observations/states to make a decision (actions) using the main policy.

        :param observations: Environment observations.
        :param states: Environment states.
        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.

        :return: Agent output. The first component is the expected action/value returned by the agent.
            The second component is a dictionary containing extra output values according to the model.
        """
        inputs = {
            "observations": self._observation_preprocessor(observations),
            "states": self._state_preprocessor(states),
        }
        self._current_is_random = timestep < self.cfg.random_timesteps
        # sample random actions
        # TODO, check for stochasticity
        if self._current_is_random:
            actions, outputs = self.policy.random_act(inputs, role="policy")
            if "log_prob" not in outputs:
                _, likelihood_outputs = self.policy.act(
                    {**inputs, "taken_actions": actions}, role="policy"
                )
                outputs = {**likelihood_outputs, **outputs, "log_prob": likelihood_outputs["log_prob"]}
            self._current_log_prob = outputs["log_prob"]
            validate_scalar_output(
                "PPO policy log_prob", self._current_log_prob, observations.shape[0], synchronize=True
            )
            return actions, outputs

        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            actions, outputs = self.policy.act(inputs, role="policy")
            self._current_log_prob = outputs["log_prob"]
            validate_scalar_output(
                "PPO policy log_prob", self._current_log_prob, observations.shape[0], synchronize=True
            )

        return actions, outputs

    def record_transition(
        self,
        *,
        observations: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory.

        :param observations: Environment observations.
        :param states: Environment states.
        :param actions: Actions taken by the agent.
        :param rewards: Instant rewards achieved by the current actions.
        :param next_observations: Next environment observations.
        :param next_states: Next environment states.
        :param terminated: Signals that indicate episodes have terminated.
        :param truncated: Signals that indicate episodes have been truncated.
        :param infos: Additional information about the environment.
        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        super().record_transition(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=timestep,
            timesteps=timesteps,
        )

        # PPO is on-policy. Random warm-up transitions may move the environment to a
        # representative state distribution, but they must not enter the PPO rollout.
        if self._current_is_random:
            return

        if self.memory is not None:
            self._current_next_observations = next_observations
            self._current_next_states = next_states

            # reward shaping
            if self.cfg.rewards_shaper is not None:
                rewards = self.cfg.rewards_shaper(rewards, timestep, timesteps)

            # compute values
            with torch.autocast(device_type=self._device_type, enabled=self.cfg.value_mixed_precision):
                inputs = {
                    "observations": self._observation_preprocessor(observations),
                    "states": self._state_preprocessor(states),
                }
                values, _ = self.value.act(inputs, role="value")
                validate_scalar_output("PPO value", values, observations.shape[0], synchronize=True)
                values = self._value_preprocessor(values, inverse=True)

            # time-limit (truncation) bootstrapping
            timeout_mask = truncated & ~terminated
            if self.cfg.time_limit_bootstrap and any_rank_true(timeout_mask):
                reset_timeout = (
                    timeout_mask.any()
                    if isinstance(infos, dict) and infos.get("_skrl_autoreset", False)
                    else torch.zeros((), dtype=torch.bool, device=timeout_mask.device)
                )
                if any_rank_true(reset_timeout):
                    raise RuntimeError(
                        "PPO time-limit bootstrapping requires final pre-reset observations/states, "
                        "but this auto-reset environment only returned reset observations"
                    )
                # ``next_observations`` / ``next_states`` must describe the final state before
                # reset for truncated environments. Auto-reset integrations must preserve that
                # state in their wrapper rather than passing the reset observation here.
                with torch.autocast(device_type=self._device_type, enabled=self.cfg.value_mixed_precision):
                    next_inputs = {
                        "observations": self._observation_preprocessor(next_observations),
                        "states": self._state_preprocessor(next_states),
                    }
                    timeout_values, _ = self.value.act(next_inputs, role="value")
                    validate_scalar_output(
                        "PPO timeout value", timeout_values, observations.shape[0], synchronize=True
                    )
                    timeout_values = self._value_preprocessor(timeout_values, inverse=True)
                rewards = rewards + self.cfg.discount_factor * timeout_values * timeout_mask

            # storage transition in memory
            self.memory.add_samples(
                observations=observations,
                states=states,
                actions=actions,
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,
                log_prob=self._current_log_prob,
                values=values,
            )
            self._rollout_consumed = False

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Method called before the interaction with the environment.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        if timestep == 0:
            # A new trainer run starts from a newly reset environment. Discard a partial
            # rollout left by a previous run so data from two unrelated episodes can never
            # share one PPO update.
            if self.memory is not None:
                self.memory.reset()
            self._rollout = 0
            self._current_next_observations = None
            self._current_next_states = None
            self._current_log_prob = None
            self._current_is_random = False
            self._rollout_consumed = False

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Method called after the interaction with the environment.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        if self._current_is_random:
            super().post_interaction(timestep=timestep, timesteps=timesteps)
            return
        self._rollout += 1
        if not self._rollout % self.cfg.rollouts and timestep >= self.cfg.learning_starts:
            with ScopedTimer() as timer:
                self.enable_models_training_mode(True)
                self.update(timestep=timestep, timesteps=timesteps)
                self.enable_models_training_mode(False)
                self.track_data("Stats / Algorithm update time (ms)", timer.elapsed_time_ms)

        # write tracking data and checkpoints
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def update(self, *, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        ensure_full_rollout(name="PPO", memory=self.memory, consumed=self._rollout_consumed)
        require_finite_model("PPO policy", self.policy, synchronize=True)
        if self.value is not self.policy:
            require_finite_model("PPO value", self.value, synchronize=True)
        require_finite(
            "PPO rollout observations", self.memory.get_tensor_by_name("observations"), synchronize=True
        )
        require_finite("PPO rollout actions", self.memory.get_tensor_by_name("actions"), synchronize=True)
        if "states" in self.memory.tensors:
            require_finite("PPO rollout states", self.memory.get_tensor_by_name("states"), synchronize=True)
        require_finite("PPO rollout rewards", self.memory.get_tensor_by_name("rewards"), synchronize=True)
        require_finite("PPO rollout log_prob", self.memory.get_tensor_by_name("log_prob"), synchronize=True)
        require_finite("PPO rollout values", self.memory.get_tensor_by_name("values"), synchronize=True)
        require_finite("PPO next observations", self._current_next_observations, synchronize=True)
        if self._current_next_states is not None:
            require_finite("PPO next states", self._current_next_states, synchronize=True)

        # compute returns and advantages
        with torch.no_grad(), torch.autocast(
            device_type=self._device_type, enabled=self.cfg.value_mixed_precision
        ):
            inputs = {
                "observations": self._observation_preprocessor(self._current_next_observations),
                "states": self._state_preprocessor(self._current_next_states),
            }
            self.value.enable_training_mode(False)
            last_values, _ = self.value.act(inputs, role="value")
            self.value.enable_training_mode(True)
            validate_scalar_output(
                "PPO last value", last_values, self._current_next_observations.shape[0], synchronize=True
            )
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            terminated=self.memory.get_tensor_by_name("terminated"),
            truncated=self.memory.get_tensor_by_name("truncated"),
            values=values,
            next_values=last_values,
            discount_factor=self.cfg.discount_factor,
            lambda_coefficient=self.cfg.lambda_,
        )
        require_finite("PPO returns", returns, synchronize=True)
        require_finite("PPO advantages", advantages, synchronize=True)

        # Keep value normalization in one frozen coordinate system for the complete PPO
        # update. In particular, old values, current predictions and return targets must be
        # comparable for value clipping. The return population is absorbed only after all
        # epochs have finished.
        self.memory.set_tensor_by_name("values", self._value_preprocessor(values))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns))
        self.memory.set_tensor_by_name("advantages", advantages)

        # Prepare randomized mini-batches. A fresh permutation is produced for every epoch.
        mini_batches = max(1, min(self.cfg.mini_batches, len(self.memory)))

        def sample_minibatches() -> list[list[torch.Tensor]]:
            indexes = (
                torch.arange(len(self.memory), device=self.device)
                if mini_batches == 1
                else torch.randperm(len(self.memory), device=self.device)
            )
            return [
                self.memory.sample_by_index(names=self._tensors_names, indexes=batch)[0]
                for batch in torch.tensor_split(indexes, mini_batches)
            ]

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        cumulative_approx_kl = 0
        cumulative_clip_fraction = 0
        cumulative_clip_magnitude = 0
        cumulative_is_ratio_sum = 0
        cumulative_is_ratio_sumsq = 0
        cumulative_entropy = 0
        cumulative_value_clip_fraction = 0
        cumulative_grad_norm = 0
        max_approx_kl = float("-inf")
        kl_early_stop_count = 0
        observed_minibatches = 0
        observed_samples = 0
        effective_minibatches = 0
        effective_samples = 0
        entropy_samples = 0
        explained_variance_samples = 0
        explained_variance_returns_sum = 0
        explained_variance_returns_sumsq = 0
        explained_variance_residual_sum = 0
        explained_variance_residual_sumsq = 0
        update_early_stop = False
        optimizer_steps = 0
        initial_replay_max_abs_log_ratio = 0.0
        critic_guard_open = False
        critic_guard_activations = 0
        critic_skipped_minibatches = 0
        value_effective_minibatches = 0
        maximum_unscaled_value_loss = 0.0
        maximum_abs_value_prediction = 0.0

        # learning epochs
        for epoch in range(self.cfg.learning_epochs):
            epoch_kl_sum = torch.zeros((), dtype=torch.float64, device=self.device)
            epoch_kl_samples = 0
            epoch_optimizer_steps = 0
            sampled_batches = sample_minibatches()

            # mini-batches loop
            for (
                sampled_observations,
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
            ) in sampled_batches:

                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    inputs = {
                        "observations": self._observation_preprocessor(sampled_observations),
                        "states": self._state_preprocessor(sampled_states),
                    }

                    _, outputs = self.policy.act({**inputs, "taken_actions": sampled_actions}, role="policy")
                    next_log_prob = outputs["log_prob"]
                    validate_scalar_output(
                        "PPO replay log_prob", next_log_prob, sampled_log_prob.shape[0], synchronize=True
                    )
                    lower, upper = 1.0 - self.cfg.ratio_clip, 1.0 + self.cfg.ratio_clip
                    log_ratio = next_log_prob - sampled_log_prob
                    ratio = torch.exp(log_ratio)
                    require_finite("PPO importance ratio", ratio, synchronize=True)

                    # compute approximate KL divergence
                    with torch.no_grad():
                        ratio_detached = ratio.detach().float()
                        log_ratio_detached = log_ratio.detach().float()
                        kl_divergence = ((ratio_detached - 1) - log_ratio_detached).mean()
                        ratio_clipped_detached = torch.clip(ratio_detached, lower, upper)
                        clip_fraction = (torch.abs(ratio_detached - 1.0) > self.cfg.ratio_clip).float().mean()
                        clip_magnitude = torch.abs(ratio_detached - ratio_clipped_detached).mean()
                        observed_batch_samples = ratio_detached.numel()
                        is_ratio_sum = ratio_detached.sum()
                        is_ratio_sumsq = ratio_detached.pow(2).sum()
                        epoch_kl_sum += kl_divergence.double() * observed_batch_samples
                        epoch_kl_samples += observed_batch_samples

                    observed_minibatches += 1
                    if observed_minibatches == 1:
                        initial_replay_max_abs_log_ratio = log_ratio_detached.abs().max().item()
                    kl_value = kl_divergence.item()
                    observed_samples += observed_batch_samples
                    cumulative_approx_kl += kl_value * observed_batch_samples
                    max_approx_kl = max(max_approx_kl, kl_value)
                    cumulative_clip_fraction += clip_fraction.item() * observed_batch_samples
                    cumulative_clip_magnitude += clip_magnitude.item() * observed_batch_samples
                    cumulative_is_ratio_sum += is_ratio_sum.item()
                    cumulative_is_ratio_sumsq += is_ratio_sumsq.item()

                    # early stopping with KL divergence
                    should_early_stop = False
                    if self.cfg.kl_threshold:
                        early_stop_kl = kl_divergence.detach().clone()
                        # use the global mean so all workers follow the same control flow and the
                        # stopping statistic matches the KL-adaptive scheduler statistic
                        if config.torch.is_distributed:
                            torch.distributed.all_reduce(early_stop_kl, op=torch.distributed.ReduceOp.SUM)
                            early_stop_kl /= config.torch.world_size
                        should_early_stop = bool((early_stop_kl > self.cfg.kl_threshold).item())
                    if should_early_stop:
                        kl_early_stop_count += 1
                        update_early_stop = True
                        break

                    # compute entropy loss
                    entropy_tensor = self.policy.get_entropy(role="policy")
                    entropy = entropy_tensor.mean()
                    entropy_batch_samples = entropy_tensor.numel()
                    if self.cfg.entropy_loss_scale:
                        entropy_loss = -self.cfg.entropy_loss_scale * entropy
                    else:
                        entropy_loss = 0

                    # compute policy loss
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(ratio, lower, upper)

                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # Compute the critic independently from policy AMP. Once its
                    # circuit breaker opens, keep training the actor but freeze
                    # the critic for the rest of this PPO update.
                    if critic_guard_open:
                        critic_skipped_minibatches += 1
                        value_clip_fraction = torch.zeros((), device=sampled_values.device)
                        value_loss = torch.zeros((), device=policy_loss.device)
                        explained_variance_batch_samples = 0
                    else:
                        with torch.autocast(
                            device_type=self._device_type,
                            enabled=self.cfg.value_mixed_precision,
                        ):
                            predicted_values_raw, _ = self.value.act(inputs, role="value")
                        validate_scalar_output(
                            "PPO replay value",
                            predicted_values_raw,
                            sampled_returns.shape[0],
                            synchronize=True,
                        )
                        value_loss, unscaled_value_loss, value_clip_fraction = compute_value_loss_fp32(
                            predicted_values=predicted_values_raw,
                            sampled_values=sampled_values,
                            sampled_returns=sampled_returns,
                            value_clip=self.cfg.value_clip,
                            value_loss_scale=self.cfg.value_loss_scale,
                        )

                        with torch.no_grad():
                            returns_detached = sampled_returns.detach().float()
                            predicted_values_detached = predicted_values_raw.detach().float()
                            residual_detached = returns_detached - predicted_values_detached
                            explained_variance_batch_samples = returns_detached.numel()
                            maximum_unscaled_value_loss = max(
                                maximum_unscaled_value_loss, unscaled_value_loss.item()
                            )
                            maximum_abs_value_prediction = max(
                                maximum_abs_value_prediction,
                                predicted_values_detached.abs().max().item(),
                            )

                        if (
                            self.cfg.value_loss_guard > 0 or self.cfg.value_prediction_guard > 0
                        ) and critic_guard_triggered(
                            predicted_values=predicted_values_raw,
                            unscaled_value_loss=unscaled_value_loss,
                            loss_threshold=self.cfg.value_loss_guard,
                            prediction_threshold=self.cfg.value_prediction_guard,
                        ):
                            critic_guard_open = True
                            critic_guard_activations += 1
                            critic_skipped_minibatches += 1
                            value_loss = torch.zeros((), device=policy_loss.device)
                            value_clip_fraction = torch.zeros((), device=sampled_values.device)
                            explained_variance_batch_samples = 0
                        else:
                            value_effective_minibatches += 1
                            explained_variance_returns_sum += returns_detached.sum().item()
                            explained_variance_returns_sumsq += returns_detached.pow(2).sum().item()
                            explained_variance_residual_sum += residual_detached.sum().item()
                            explained_variance_residual_sumsq += residual_detached.pow(2).sum().item()

                    total_loss = policy_loss + entropy_loss + value_loss
                    require_finite("PPO loss", total_loss, synchronize=True)

                # optimization step
                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value and value_loss.requires_grad:
                        self.value.reduce_parameters()

                self.scaler.unscale_(self.optimizer)
                if self.policy is self.value:
                    grad_parameters = tuple(self.policy.parameters())
                else:
                    grad_parameters = tuple(itertools.chain(self.policy.parameters(), self.value.parameters()))

                if self.cfg.grad_norm_clip > 0:
                    grad_norm = nn.utils.clip_grad_norm_(grad_parameters, self.cfg.grad_norm_clip)
                else:
                    grad_norm = torch.zeros((), device=self.device)
                    for parameter in grad_parameters:
                        if parameter.grad is not None:
                            grad_norm += parameter.grad.detach().pow(2).sum()
                    grad_norm = torch.sqrt(grad_norm)

                # GradScaler owns non-finite gradient recovery under AMP. Without AMP,
                # reject a finite-loss/non-finite-gradient update before it can corrupt
                # model parameters (and synchronize that decision across workers).
                if not self.scaler.is_enabled():
                    require_finite("PPO gradient norm", grad_norm, synchronize=True)

                track_optimizer_step = self.scaler.is_enabled()
                scale_before_step = self.scaler.get_scale() if track_optimizer_step else None
                self.scaler.step(self.optimizer)
                self.scaler.update()
                optimizer_step_succeeded = not track_optimizer_step or self.scaler.get_scale() >= scale_before_step
                if config.torch.is_distributed and track_optimizer_step:
                    optimizer_step_succeeded_tensor = torch.tensor(
                        optimizer_step_succeeded, dtype=torch.int32, device=self.device
                    )
                    torch.distributed.all_reduce(optimizer_step_succeeded_tensor, op=torch.distributed.ReduceOp.MIN)
                    optimizer_step_succeeded = bool(optimizer_step_succeeded_tensor.item())
                if optimizer_step_succeeded:
                    epoch_optimizer_steps += 1
                    optimizer_steps += 1

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self.cfg.entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()
                cumulative_entropy += entropy.item() * entropy_batch_samples
                entropy_samples += entropy_batch_samples
                cumulative_value_clip_fraction += value_clip_fraction.item() * explained_variance_batch_samples
                cumulative_grad_norm += grad_norm.item()
                effective_minibatches += 1
                effective_samples += explained_variance_batch_samples
                explained_variance_samples += explained_variance_batch_samples

            # update learning rate
            if self.scheduler:
                # A terminal KL at the start of an epoch can be the first observation of the
                # preceding epoch's final optimizer step, so report it once to KLAdaptiveLR.
                if isinstance(self.scheduler, KLAdaptiveLR) and (
                    epoch_optimizer_steps or (update_early_stop and optimizer_steps > 0)
                ):
                    # Weight uneven mini-batches by their actual sample counts.
                    if config.torch.is_distributed:
                        stats = torch.stack(
                            (
                                epoch_kl_sum,
                                torch.tensor(float(epoch_kl_samples), dtype=torch.float64, device=self.device),
                            )
                        )
                        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
                        kl = stats[0] / stats[1].clamp_min(1)
                    else:
                        kl = epoch_kl_sum / max(epoch_kl_samples, 1)
                    self.scheduler.step(kl.item())
                elif epoch_optimizer_steps:
                    self.scheduler.step()

            # a KL early-stop applies to the complete PPO update, not only to the current epoch
            if update_early_stop:
                break

        value_lr_before_backoff = self.optimizer.param_groups[-1]["lr"]
        value_lr_after_backoff = value_lr_before_backoff
        if critic_guard_open:
            value_lr_before_backoff, value_lr_after_backoff = backoff_value_learning_rate(
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                factor=self.cfg.value_lr_backoff_factor,
                minimum=self.cfg.value_lr_backoff_min,
            )
            logger.warning(
                "PPO critic circuit breaker opened: "
                f"max unscaled value loss={maximum_unscaled_value_loss:.6g}, "
                f"max abs prediction={maximum_abs_value_prediction:.6g}, "
                f"skipped minibatches={critic_skipped_minibatches}, "
                f"value lr={value_lr_before_backoff:.6g}->{value_lr_after_backoff:.6g}"
            )

        # Keep input normalization fixed from rollout collection through optimization. Only after
        # the update is complete, absorb each rollout sample once for use by the next rollout.
        del inputs
        observations = self.memory.get_tensor_by_name("observations").flatten(0, 1)
        states = self.memory.get_tensor_by_name("states").flatten(0, 1) if self.state_space is not None else None
        sample_count = observations.shape[0]
        if sample_count:
            # Bound the output allocation for large observations while keeping at least two samples
            # per chunk when the rollout contains more than one sample.
            preprocessor_batches = min(mini_batches, max(sample_count // 2, 1))
            observation_batches = torch.tensor_split(observations, preprocessor_batches)
            state_batches = torch.tensor_split(states, preprocessor_batches) if states is not None else None
            observation_update_stats = getattr(self._observation_preprocessor, "update_stats", None)
            state_update_stats = getattr(self._state_preprocessor, "update_stats", None)
            observation_update_stats_distributed = getattr(
                self._observation_preprocessor, "update_stats_distributed", None
            )
            state_update_stats_distributed = getattr(
                self._state_preprocessor, "update_stats_distributed", None
            )
            for i, observation_batch in enumerate(observation_batches):
                if observation_update_stats_distributed is not None:
                    observation_update_stats_distributed(observation_batch)
                elif observation_update_stats is not None:
                    observation_update_stats(observation_batch)
                else:
                    self._observation_preprocessor(observation_batch, train=True)
                if state_batches is not None:
                    if state_update_stats_distributed is not None:
                        state_update_stats_distributed(state_batches[i])
                    elif state_update_stats is not None:
                        state_update_stats(state_batches[i])
                    else:
                        self._state_preprocessor(state_batches[i], train=True)

            value_update_stats = getattr(self._value_preprocessor, "update_stats", None)
            value_update_stats_distributed = getattr(
                self._value_preprocessor, "update_stats_distributed", None
            )
            flat_returns = returns.flatten(0, 1)
            if value_update_stats_distributed is not None:
                value_update_stats_distributed(flat_returns)
            elif value_update_stats is not None:
                value_update_stats(flat_returns)
            else:
                self._value_preprocessor(flat_returns, train=True)

        # record data
        observed_samples_safe = max(observed_samples, 1)
        effective_minibatches_safe = max(effective_minibatches, 1)
        effective_samples_safe = max(effective_samples, 1)
        entropy_samples_safe = max(entropy_samples, 1)
        self.track_data("Loss / Policy loss", cumulative_policy_loss / effective_minibatches_safe)
        self.track_data("Loss / Value loss", cumulative_value_loss / max(value_effective_minibatches, 1))
        if self.cfg.entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / effective_minibatches_safe)

        is_ratio_mean = cumulative_is_ratio_sum / observed_samples_safe
        is_ratio_var = max(cumulative_is_ratio_sumsq / observed_samples_safe - is_ratio_mean * is_ratio_mean, 0.0)
        explained_variance = 0.0
        if explained_variance_samples:
            returns_mean = explained_variance_returns_sum / explained_variance_samples
            returns_var = (
                explained_variance_returns_sumsq / explained_variance_samples - returns_mean * returns_mean
            )
            if returns_var > 1e-8:
                residual_mean = explained_variance_residual_sum / explained_variance_samples
                residual_var = (
                    explained_variance_residual_sumsq / explained_variance_samples - residual_mean * residual_mean
                )
                explained_variance = 1.0 - residual_var / (returns_var + 1e-8)

        self.track_data("Policy / Approx KL", cumulative_approx_kl / observed_samples_safe)
        self.track_data("Policy / Approx KL (max)", max_approx_kl if max_approx_kl > float("-inf") else 0.0)
        self.track_data("Policy / Clip fraction", cumulative_clip_fraction / observed_samples_safe)
        self.track_data("Policy / Clip magnitude", cumulative_clip_magnitude / observed_samples_safe)
        self.track_data("Policy / IS ratio (mean)", is_ratio_mean)
        self.track_data("Policy / IS ratio (std)", is_ratio_var**0.5)
        self.track_data("Policy / Entropy", cumulative_entropy / entropy_samples_safe)
        policy_distribution = self.policy.distribution(role="policy")
        policy_stddev = policy_distribution.stddev.float().mean()
        if torch.isfinite(policy_stddev):
            self.track_data("Policy / Standard deviation", policy_stddev.item())
        elif hasattr(policy_distribution, "probs"):
            self.track_data(
                "Policy / Maximum action probability",
                policy_distribution.probs.float().amax(dim=-1).mean().item(),
            )
        self.track_data("Value / Explained variance", explained_variance)
        self.track_data("Value / Clip fraction", cumulative_value_clip_fraction / effective_samples_safe)
        self.track_data("Optimization / Grad norm", cumulative_grad_norm / effective_minibatches_safe)
        self.track_data("Optimization / KL early-stop count", kl_early_stop_count)
        self.track_data("Optimization / Effective minibatches", effective_minibatches)
        self.track_data("Optimization / Observed minibatches", observed_minibatches)
        self.track_data("Optimization / Successful optimizer steps", optimizer_steps)
        self.track_data("Optimization / Observed samples", observed_samples)
        self.track_data("Optimization / Effective samples", effective_samples)
        self.track_data("Optimization / Critic guard activations", critic_guard_activations)
        self.track_data("Optimization / Critic skipped minibatches", critic_skipped_minibatches)
        self.track_data("Policy / Initial replay max abs log-ratio", initial_replay_max_abs_log_ratio)
        self.track_data("Value / Maximum unscaled loss", maximum_unscaled_value_loss)
        self.track_data("Value / Maximum abs prediction", maximum_abs_value_prediction)
        self.track_data(
            "Learning / Value LR backoff ratio",
            value_lr_after_backoff / value_lr_before_backoff if value_lr_before_backoff else 1.0,
        )

        for name, preprocessor in (
            ("Observation", self._observation_preprocessor),
            ("State", self._state_preprocessor),
            ("Value", self._value_preprocessor),
        ):
            count = getattr(preprocessor, "current_count", None)
            if count is not None:
                self.track_data(f"Preprocessor / {name} sample count", float(count.item()))

        learning_rates = (
            self.scheduler.get_last_lr()
            if self.scheduler
            else [group["lr"] for group in self.optimizer.param_groups]
        )
        self.track_data("Learning / Learning rate", learning_rates[0])
        self.track_data("Learning / Policy learning rate", learning_rates[0])
        self.track_data(
            "Learning / Value learning rate",
            learning_rates[1] if len(learning_rates) > 1 else learning_rates[0],
        )
        self._rollout_consumed = True
