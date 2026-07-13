from __future__ import annotations

from typing import Any

import itertools
import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.utils import ScopedTimer

from .ppo_cfg import PPO_CFG


def compute_gae(
    *,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    discount_factor: float = 0.99,
    lambda_coefficient: float = 0.95,
) -> torch.Tensor:
    """Compute the Generalized Advantage Estimator (GAE).

    :param rewards: Rewards obtained by the agent.
    :param terminated: Signals to indicate that episodes have ended.
    :param values: Values obtained by the agent.
    :param next_values: Next values obtained by the agent.
    :param discount_factor: Discount factor.
    :param lambda_coefficient: Lambda coefficient.

    :return: Generalized Advantage Estimator.
    """
    advantage = 0
    advantages = torch.zeros_like(rewards)
    not_terminated = terminated.logical_not()
    memory_size = rewards.shape[0]

    # advantages computation
    for i in reversed(range(memory_size)):
        next_values = values[i + 1] if i < memory_size - 1 else next_values
        advantage = (
            rewards[i]
            - values[i]
            + discount_factor * not_terminated[i] * (next_values + lambda_coefficient * advantage)
        )
        advantages[i] = advantage
    # returns computation
    returns = advantages + values
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    return returns, advantages


class PPO_RNN(Agent):
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
        """Proximal Policy Optimization (PPO) with support for Recurrent Neural Networks (RNN, GRU, LSTM, etc.).

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

        # set up optimizer and learning rate scheduler
        if self.policy is not None and self.value is not None:
            # - optimizers
            optimizer_cfg = self.cfg.optimizer
            if isinstance(optimizer_cfg, str):
                try:
                    optimizer_cls = getattr(torch.optim, optimizer_cfg)
                except AttributeError as e:
                    raise ValueError(f"Unknown optimizer '{optimizer_cfg}' for PPO_RNN") from e
            else:
                optimizer_cls = optimizer_cfg
            optimizer_kwargs = dict(self.cfg.optimizer_kwargs)
            if self.policy is self.value:
                self.optimizer = optimizer_cls(self.policy.parameters(), lr=self.cfg.learning_rate[0], **optimizer_kwargs)
            else:
                self.optimizer = optimizer_cls(
                    itertools.chain(self.policy.parameters(), self.value.parameters()),
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

            self._tensors_names = [
                "observations",
                "states",
                "actions",
                "terminated",
                "truncated",
                "log_prob",
                "values",
                "returns",
                "advantages",
            ]

        # RNN specifications
        self._rnn = False  # flag to indicate whether RNN is available
        self._rnn_tensors_names = []  # used for sampling during training
        self._rnn_final_states = {"policy": [], "value": []}
        self._rnn_initial_states = {"policy": [], "value": []}
        self._rnn_sequence_length = self.policy.get_specification().get("rnn", {}).get("sequence_length", 1)

        # policy
        for i, size in enumerate(self.policy.get_specification().get("rnn", {}).get("sizes", [])):
            self._rnn = True
            # create tensors in memory
            if self.memory is not None:
                self.memory.create_tensor(
                    name=f"rnn_policy_{i}", size=(size[0], size[2]), dtype=torch.float32, keep_dimensions=True
                )
                self._rnn_tensors_names.append(f"rnn_policy_{i}")
            # default RNN states
            self._rnn_initial_states["policy"].append(torch.zeros(size, dtype=torch.float32, device=self.device))

        # value
        if self.value is not None:
            if self.policy is self.value:
                self._rnn_initial_states["value"] = self._rnn_initial_states["policy"]
            else:
                for i, size in enumerate(self.value.get_specification().get("rnn", {}).get("sizes", [])):
                    self._rnn = True
                    # create tensors in memory
                    if self.memory is not None:
                        self.memory.create_tensor(
                            name=f"rnn_value_{i}", size=(size[0], size[2]), dtype=torch.float32, keep_dimensions=True
                        )
                        self._rnn_tensors_names.append(f"rnn_value_{i}")
                    # default RNN states
                    self._rnn_initial_states["value"].append(torch.zeros(size, dtype=torch.float32, device=self.device))

        # create temporary variables needed for storage and computation
        self._current_next_observations = None
        self._current_next_states = None
        self._current_log_prob = None
        self._rollout = 0

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
        inputs.update({"rnn": self._rnn_initial_states["policy"]} if self._rnn else {})

        # sample random actions
        # TODO, check for stochasticity
        if timestep < self.cfg.random_timesteps:
            return self.policy.random_act(inputs, role="policy")

        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            actions, outputs = self.policy.act(inputs, role="policy")
            self._current_log_prob = outputs["log_prob"]

        if self._rnn:
            self._rnn_final_states["policy"] = outputs.get("rnn", [])

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

        if self.memory is not None:
            self._current_next_observations = next_observations
            self._current_next_states = next_states

            # reward shaping
            if self.cfg.rewards_shaper is not None:
                rewards = self.cfg.rewards_shaper(rewards, timestep, timesteps)

            # compute values
            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                inputs = {
                    "observations": self._observation_preprocessor(observations),
                    "states": self._state_preprocessor(states),
                }
                inputs.update({"rnn": self._rnn_initial_states["value"]} if self._rnn else {})
                values, outputs = self.value.act(inputs, role="value")
                values = self._value_preprocessor(values, inverse=True)

            # time-limit (truncation) bootstrapping
            if self.cfg.time_limit_bootstrap:
                rewards += self.cfg.discount_factor * values * truncated

            # package RNN states
            rnn_states = {}
            if self._rnn:
                rnn_states.update(
                    {f"rnn_policy_{i}": s.transpose(0, 1) for i, s in enumerate(self._rnn_initial_states["policy"])}
                )
                if self.policy is not self.value:
                    rnn_states.update(
                        {f"rnn_value_{i}": s.transpose(0, 1) for i, s in enumerate(self._rnn_initial_states["value"])}
                    )

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
                **rnn_states,
            )

        # update RNN states
        if self._rnn:
            self._rnn_final_states["value"] = (
                self._rnn_final_states["policy"] if self.policy is self.value else outputs.get("rnn", [])
            )

            # reset states if the episodes have ended
            finished_episodes = (terminated | truncated).nonzero(as_tuple=False)
            if finished_episodes.numel():
                for rnn_state in self._rnn_final_states["policy"]:
                    rnn_state[:, finished_episodes[:, 0]] = 0
                if self.policy is not self.value:
                    for rnn_state in self._rnn_final_states["value"]:
                        rnn_state[:, finished_episodes[:, 0]] = 0

            # Keep a distinct container for the states used by the next action. Otherwise, assigning
            # a new final state in act() also overwrites the initial state that record_transition()
            # must store for replay.
            policy_states = list(self._rnn_final_states["policy"])
            self._rnn_initial_states = {
                "policy": policy_states,
                "value": policy_states if self.policy is self.value else list(self._rnn_final_states["value"]),
            }

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Method called before the interaction with the environment.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        pass

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Method called after the interaction with the environment.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
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
        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            inputs = {
                "observations": self._observation_preprocessor(self._current_next_observations),
                "states": self._state_preprocessor(self._current_next_states),
            }
            inputs.update({"rnn": self._rnn_initial_states["value"]} if self._rnn else {})
            self.value.enable_training_mode(False)
            last_values, _ = self.value.act(inputs, role="value")
            self.value.enable_training_mode(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            terminated=self.memory.get_tensor_by_name("terminated"),
            values=values,
            next_values=last_values,
            discount_factor=self.cfg.discount_factor,
            lambda_coefficient=self.cfg.lambda_,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        rnn_policy, rnn_value = {}, {}
        if self._rnn_sequence_length > 1:
            if self.memory.memory_size % self._rnn_sequence_length:
                raise ValueError(
                    f"PPO_RNN rollout length ({self.memory.memory_size}) must be divisible by the RNN sequence "
                    f"length ({self._rnn_sequence_length})"
                )
            sequence_count = len(self.memory) // self._rnn_sequence_length
            mini_batches = max(1, min(self.cfg.mini_batches, sequence_count))
            sequence_indexes = torch.as_tensor(self.memory.all_sequence_indexes, device=self.device).view(
                -1, self._rnn_sequence_length
            )
            index_batches = [batch.flatten() for batch in torch.tensor_split(sequence_indexes, mini_batches)]
            sampled_batches = [
                self.memory.sample_by_index(names=self._tensors_names, indexes=indexes)[0]
                for indexes in index_batches
            ]
            if self._rnn:
                sampled_rnn_batches = [
                    self.memory.sample_by_index(names=self._rnn_tensors_names, indexes=indexes)[0]
                    for indexes in index_batches
                ]
        else:
            mini_batches = max(1, min(self.cfg.mini_batches, len(self.memory)))
            sampled_tensors = self.memory.sample_all(names=self._tensors_names, mini_batches=1)[0]
            sampled_tensors_batches = [
                torch.tensor_split(tensor, mini_batches) if tensor is not None else [None] * mini_batches
                for tensor in sampled_tensors
            ]
            sampled_batches = [
                [tensor_batches[i] for tensor_batches in sampled_tensors_batches] for i in range(mini_batches)
            ]
            if self._rnn:
                sampled_rnn_tensors = self.memory.sample_all(names=self._rnn_tensors_names, mini_batches=1)[0]
                sampled_rnn_tensors_batches = [
                    torch.tensor_split(tensor, mini_batches) if tensor is not None else [None] * mini_batches
                    for tensor in sampled_rnn_tensors
                ]
                sampled_rnn_batches = [
                    [tensor_batches[i] for tensor_batches in sampled_rnn_tensors_batches]
                    for i in range(mini_batches)
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

        # learning epochs
        for epoch in range(self.cfg.learning_epochs):
            kl_divergences = []
            epoch_optimizer_steps = 0

            # mini-batches loop
            for i, (
                sampled_observations,
                sampled_states,
                sampled_actions,
                sampled_terminated,
                sampled_truncated,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
            ) in enumerate(sampled_batches):

                if self._rnn:
                    if self.policy is self.value:
                        rnn_policy = {
                            "rnn": [s.transpose(0, 1) for s in sampled_rnn_batches[i]],
                            "terminated": sampled_terminated,
                            "truncated": sampled_truncated,
                        }
                        rnn_value = rnn_policy
                    else:
                        rnn_policy = {
                            "rnn": [
                                s.transpose(0, 1)
                                for s, n in zip(sampled_rnn_batches[i], self._rnn_tensors_names)
                                if "policy" in n
                            ],
                            "terminated": sampled_terminated,
                            "truncated": sampled_truncated,
                        }
                        rnn_value = {
                            "rnn": [
                                s.transpose(0, 1)
                                for s, n in zip(sampled_rnn_batches[i], self._rnn_tensors_names)
                                if "value" in n
                            ],
                            "terminated": sampled_terminated,
                            "truncated": sampled_truncated,
                        }

                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    inputs = {
                        "observations": self._observation_preprocessor(sampled_observations),
                        "states": self._state_preprocessor(sampled_states),
                    }

                    _, outputs = self.policy.act(
                        {**inputs, "taken_actions": sampled_actions, **rnn_policy}, role="policy"
                    )
                    next_log_prob = outputs["log_prob"]
                    lower, upper = 1.0 - self.cfg.ratio_clip, 1.0 + self.cfg.ratio_clip
                    log_ratio = next_log_prob - sampled_log_prob
                    ratio = torch.exp(log_ratio)

                    # compute approximate KL divergence
                    with torch.no_grad():
                        ratio_detached = ratio.detach().float()
                        log_ratio_detached = log_ratio.detach().float()
                        kl_divergence = ((ratio_detached - 1) - log_ratio_detached).mean()
                        kl_divergences.append(kl_divergence)
                        ratio_clipped_detached = torch.clip(ratio_detached, lower, upper)
                        clip_fraction = (torch.abs(ratio_detached - 1.0) > self.cfg.ratio_clip).float().mean()
                        clip_magnitude = torch.abs(ratio_detached - ratio_clipped_detached).mean()
                        observed_batch_samples = ratio_detached.numel()
                        is_ratio_sum = ratio_detached.sum()
                        is_ratio_sumsq = ratio_detached.pow(2).sum()

                    observed_minibatches += 1
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

                    # compute value loss
                    predicted_values_raw, _ = self.value.act({**inputs, **rnn_value}, role="value")

                    with torch.no_grad():
                        returns_detached = sampled_returns.detach().float()
                        predicted_values_detached = predicted_values_raw.detach().float()
                        residual_detached = returns_detached - predicted_values_detached
                        explained_variance_batch_samples = returns_detached.numel()
                        explained_variance_returns_sum += returns_detached.sum().item()
                        explained_variance_returns_sumsq += returns_detached.pow(2).sum().item()
                        explained_variance_residual_sum += residual_detached.sum().item()
                        explained_variance_residual_sumsq += residual_detached.pow(2).sum().item()

                    if self.cfg.value_clip > 0:
                        value_delta = predicted_values_raw - sampled_values
                        with torch.no_grad():
                            value_clip_fraction = (torch.abs(value_delta) > self.cfg.value_clip).float().mean()
                        predicted_values = sampled_values + torch.clip(
                            value_delta, min=-self.cfg.value_clip, max=self.cfg.value_clip
                        )
                    else:
                        predicted_values = predicted_values_raw
                        value_clip_fraction = torch.zeros((), device=sampled_values.device)
                    value_loss = self.cfg.value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                # optimization step
                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
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

                track_optimizer_step = self.scheduler is not None and self.scaler.is_enabled()
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
                    kl = torch.stack(kl_divergences).mean()
                    # reduce (collect from all workers/processes) KL in distributed runs
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                elif epoch_optimizer_steps:
                    self.scheduler.step()

            # a KL early-stop applies to the complete PPO update, not only to the current epoch
            if update_early_stop:
                break

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
            for i, observation_batch in enumerate(observation_batches):
                if observation_update_stats is not None:
                    observation_update_stats(observation_batch)
                else:
                    self._observation_preprocessor(observation_batch, train=True)
                if state_batches is not None:
                    if state_update_stats is not None:
                        state_update_stats(state_batches[i])
                    else:
                        self._state_preprocessor(state_batches[i], train=True)

        # record data
        observed_samples_safe = max(observed_samples, 1)
        effective_minibatches_safe = max(effective_minibatches, 1)
        effective_samples_safe = max(effective_samples, 1)
        entropy_samples_safe = max(entropy_samples, 1)
        self.track_data("Loss / Policy loss", cumulative_policy_loss / effective_minibatches_safe)
        self.track_data("Loss / Value loss", cumulative_value_loss / effective_minibatches_safe)
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
        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())
        self.track_data("Value / Explained variance", explained_variance)
        self.track_data("Value / Clip fraction", cumulative_value_clip_fraction / effective_samples_safe)
        self.track_data("Optimization / Grad norm", cumulative_grad_norm / effective_minibatches_safe)
        self.track_data("Optimization / KL early-stop count", kl_early_stop_count)
        self.track_data("Optimization / Effective minibatches", effective_minibatches)
        self.track_data("Optimization / Observed minibatches", observed_minibatches)

        if self.scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
