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
from ._utils import (
    any_rank_true,
    ensure_full_rollout,
    require_finite,
    require_finite_model,
    validate_ppo_setup,
    validate_rnn_output,
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
        validate_ppo_setup(
            name="PPO_RNN", cfg=self.cfg, policy=self.policy, value=self.value, memory=self.memory
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
                    raise ValueError(f"Unknown optimizer '{optimizer_cfg}' for PPO_RNN") from e
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

        policy_spec = self.policy.get_specification().get("rnn", {})
        value_spec = self.value.get_specification().get("rnn", {}) if self.value is not None else {}
        policy_sizes = list(policy_spec.get("sizes", []))
        value_sizes = policy_sizes if self.policy is self.value else list(value_spec.get("sizes", []))
        raw_policy_sequence_length = policy_spec.get("sequence_length", 1)
        raw_value_sequence_length = (
            raw_policy_sequence_length if self.policy is self.value else value_spec.get("sequence_length", 1)
        )
        for role, length in (
            ("policy", raw_policy_sequence_length),
            ("value", raw_value_sequence_length),
        ):
            if not isinstance(length, int) or isinstance(length, bool) or length < 1:
                raise ValueError(f"PPO_RNN {role} sequence length must be a positive integer")
        policy_sequence_length = raw_policy_sequence_length
        value_sequence_length = raw_value_sequence_length

        if bool(policy_sizes) != bool(value_sizes):
            raise ValueError("PPO_RNN requires policy and value to either both be recurrent or both be feed-forward")
        if policy_sizes and policy_sequence_length != value_sequence_length:
            raise ValueError(
                "PPO_RNN policy and value sequence lengths must match "
                f"({policy_sequence_length} != {value_sequence_length})"
            )
        if not policy_sizes and policy_sequence_length != 1:
            raise ValueError("PPO_RNN feed-forward models must use sequence length 1")
        if self.memory is not None:
            if self.memory.memory_size != self.cfg.rollouts:
                raise ValueError(
                    f"PPO_RNN rollout length ({self.cfg.rollouts}) must match memory size "
                    f"({self.memory.memory_size})"
                )
            if policy_sequence_length > 1 and self.memory.memory_size % policy_sequence_length:
                raise ValueError(
                    f"PPO_RNN rollout length ({self.memory.memory_size}) must be divisible by the RNN "
                    f"sequence length ({policy_sequence_length})"
                )
            for role, sizes in (("policy", policy_sizes), ("value", value_sizes)):
                for size in sizes:
                    if len(size) != 3:
                        raise ValueError(f"PPO_RNN {role} state specification must be (layers, num_envs, hidden)")
                    if any(not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1 for dimension in size):
                        raise ValueError(f"PPO_RNN {role} recurrent state dimensions must be positive integers")
                    if int(size[1]) != self.memory.num_envs:
                        raise ValueError(
                            f"PPO_RNN {role} recurrent environment count ({size[1]}) does not match "
                            f"memory num_envs ({self.memory.num_envs})"
                        )

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
        self._rnn_sequence_length = policy_sequence_length

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
        self._current_is_random = False
        self._rollout = 0
        self._rollout_consumed = False

    def reset_rnn_states(self, env_ids: torch.Tensor | list[int] | None = None) -> None:
        """Reset recurrent state after an explicit environment reset.

        Episode terminations observed through ``record_transition`` are reset automatically. This
        method covers lifecycle resets performed outside an environment step, such as starting a
        new train/eval run or manually resetting selected vector environments. Because an explicit
        reset is an episode boundary that is absent from the stored done masks, any partial
        on-policy rollout is discarded to prevent sequence replay from crossing that boundary.
        """

        if self.memory is not None:
            self.memory.reset()
        self._rollout = 0
        self._current_next_observations = None
        self._current_next_states = None
        self._current_log_prob = None
        self._current_is_random = False
        self._rollout_consumed = False
        if not self._rnn:
            return
        indexes = None if env_ids is None else torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        seen: set[int] = set()
        for state_group in (self._rnn_initial_states, self._rnn_final_states):
            for states in state_group.values():
                for state in states:
                    # Shared actor/critic state tensors may appear in more than one container.
                    if id(state) in seen:
                        continue
                    seen.add(id(state))
                    if indexes is None:
                        state.zero_()
                    else:
                        state[:, indexes] = 0

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
        self._current_is_random = timestep < self.cfg.random_timesteps

        # sample random actions
        # TODO, check for stochasticity
        if self._current_is_random:
            actions, outputs = self.policy.random_act(inputs, role="policy")
            if "log_prob" not in outputs or (
                self._rnn and len(outputs.get("rnn", [])) != len(self._rnn_initial_states["policy"])
            ):
                _, likelihood_outputs = self.policy.act(
                    {**inputs, "taken_actions": actions}, role="policy"
                )
                # A custom random_act may provide its own likelihood metadata but omit
                # recurrent state. Always take the policy-evaluated log-probability and
                # RNN output so warm-up advances the same hidden dynamics as normal act().
                outputs = {**outputs, **likelihood_outputs}
            self._current_log_prob = outputs["log_prob"]
            validate_scalar_output(
                "PPO_RNN policy log_prob", self._current_log_prob, observations.shape[0], synchronize=True
            )
            if self._rnn:
                self._rnn_final_states["policy"] = validate_rnn_output(
                    "PPO_RNN policy",
                    outputs,
                    self._rnn_initial_states["policy"],
                    synchronize=True,
                )
            return actions, outputs

        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            actions, outputs = self.policy.act(inputs, role="policy")
            self._current_log_prob = outputs["log_prob"]
            validate_scalar_output(
                "PPO_RNN policy log_prob", self._current_log_prob, observations.shape[0], synchronize=True
            )

        if self._rnn:
            self._rnn_final_states["policy"] = validate_rnn_output(
                "PPO_RNN policy",
                outputs,
                self._rnn_initial_states["policy"],
                synchronize=True,
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

        if self.memory is not None:
            if not self._current_is_random:
                self._current_next_observations = next_observations
                self._current_next_states = next_states

            # reward shaping
            if not self._current_is_random and self.cfg.rewards_shaper is not None:
                rewards = self.cfg.rewards_shaper(rewards, timestep, timesteps)

            # compute values
            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                inputs = {
                    "observations": self._observation_preprocessor(observations),
                    "states": self._state_preprocessor(states),
                }
                inputs.update({"rnn": self._rnn_initial_states["value"]} if self._rnn else {})
                values, value_outputs = self.value.act(inputs, role="value")
                validate_scalar_output("PPO_RNN value", values, observations.shape[0], synchronize=True)
                validated_value_states = (
                    validate_rnn_output(
                        "PPO_RNN value",
                        value_outputs,
                        self._rnn_initial_states["value"],
                        synchronize=True,
                    )
                    if self._rnn
                    else []
                )
                values = self._value_preprocessor(values, inverse=True)

            # time-limit (truncation) bootstrapping
            timeout_mask = truncated & ~terminated
            if (
                not self._current_is_random
                and self.cfg.time_limit_bootstrap
                and any_rank_true(timeout_mask)
            ):
                reset_timeout = (
                    timeout_mask.any()
                    if isinstance(infos, dict) and infos.get("_skrl_autoreset", False)
                    else torch.zeros((), dtype=torch.bool, device=timeout_mask.device)
                )
                if any_rank_true(reset_timeout):
                    raise RuntimeError(
                        "PPO_RNN time-limit bootstrapping requires final pre-reset observations/states, "
                        "but this auto-reset environment only returned reset observations"
                    )
                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    next_inputs = {
                        "observations": self._observation_preprocessor(next_observations),
                        "states": self._state_preprocessor(next_states),
                    }
                    next_inputs.update({"rnn": validated_value_states} if self._rnn else {})
                    timeout_values, _ = self.value.act(next_inputs, role="value")
                    validate_scalar_output(
                        "PPO_RNN timeout value", timeout_values, observations.shape[0], synchronize=True
                    )
                    timeout_values = self._value_preprocessor(timeout_values, inverse=True)
                rewards = rewards + self.cfg.discount_factor * timeout_values * timeout_mask

            # package RNN states
            rnn_states = {}
            if self._rnn and not self._current_is_random:
                rnn_states.update(
                    {f"rnn_policy_{i}": s.transpose(0, 1) for i, s in enumerate(self._rnn_initial_states["policy"])}
                )
                if self.policy is not self.value:
                    rnn_states.update(
                        {f"rnn_value_{i}": s.transpose(0, 1) for i, s in enumerate(self._rnn_initial_states["value"])}
                    )

            # storage transition in memory
            if not self._current_is_random:
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
                self._rollout_consumed = False

        # update RNN states
        if self._rnn:
            if self.policy is self.value:
                self._rnn_final_states["value"] = self._rnn_final_states["policy"]
            elif self.memory is not None:
                self._rnn_final_states["value"] = validated_value_states
            else:
                self._rnn_final_states["value"] = list(self._rnn_initial_states["value"])

            # reset states if the episodes have ended
            finished_episodes = (terminated | truncated).nonzero(as_tuple=False)
            self.track_data("RNN / Reset environments", finished_episodes.shape[0])
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
            for role, states in self._rnn_initial_states.items():
                if states:
                    hidden_norm = torch.stack([state.float().norm() for state in states]).mean().item()
                    self.track_data(f"RNN / {role.capitalize()} hidden norm", hidden_norm)

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Method called before the interaction with the environment.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        if timestep == 0:
            self.reset_rnn_states()

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
        ensure_full_rollout(name="PPO_RNN", memory=self.memory, consumed=self._rollout_consumed)
        require_finite_model("PPO_RNN policy", self.policy, synchronize=True)
        if self.value is not self.policy:
            require_finite_model("PPO_RNN value", self.value, synchronize=True)
        require_finite(
            "PPO_RNN rollout observations", self.memory.get_tensor_by_name("observations"), synchronize=True
        )
        require_finite("PPO_RNN rollout actions", self.memory.get_tensor_by_name("actions"), synchronize=True)
        if "states" in self.memory.tensors:
            require_finite("PPO_RNN rollout states", self.memory.get_tensor_by_name("states"), synchronize=True)
        require_finite("PPO_RNN rollout rewards", self.memory.get_tensor_by_name("rewards"), synchronize=True)
        require_finite("PPO_RNN rollout log_prob", self.memory.get_tensor_by_name("log_prob"), synchronize=True)
        require_finite("PPO_RNN rollout values", self.memory.get_tensor_by_name("values"), synchronize=True)
        require_finite("PPO_RNN next observations", self._current_next_observations, synchronize=True)
        if self._current_next_states is not None:
            require_finite("PPO_RNN next states", self._current_next_states, synchronize=True)

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
            validate_scalar_output(
                "PPO_RNN last value", last_values, self._current_next_observations.shape[0], synchronize=True
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
        require_finite("PPO_RNN returns", returns, synchronize=True)
        require_finite("PPO_RNN advantages", advantages, synchronize=True)

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns))
        self.memory.set_tensor_by_name("advantages", advantages)

        # Build randomized mini-batches on complete recurrent sequence boundaries.
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

            def sample_minibatches():
                permutation = (
                    torch.arange(sequence_indexes.shape[0], device=self.device)
                    if mini_batches == 1
                    else torch.randperm(sequence_indexes.shape[0], device=self.device)
                )
                shuffled_sequences = sequence_indexes[permutation]
                index_batches = [
                    batch.flatten() for batch in torch.tensor_split(shuffled_sequences, mini_batches)
                ]
                sampled = [
                    self.memory.sample_by_index(names=self._tensors_names, indexes=indexes)[0]
                    for indexes in index_batches
                ]
                sampled_rnn = [
                    self.memory.sample_by_index(names=self._rnn_tensors_names, indexes=indexes)[0]
                    for indexes in index_batches
                ] if self._rnn else []
                return sampled, sampled_rnn
        else:
            mini_batches = max(1, min(self.cfg.mini_batches, len(self.memory)))

            def sample_minibatches():
                indexes = (
                    torch.arange(len(self.memory), device=self.device)
                    if mini_batches == 1
                    else torch.randperm(len(self.memory), device=self.device)
                )
                index_batches = torch.tensor_split(indexes, mini_batches)
                sampled = [
                    self.memory.sample_by_index(names=self._tensors_names, indexes=batch)[0]
                    for batch in index_batches
                ]
                sampled_rnn = [
                    self.memory.sample_by_index(names=self._rnn_tensors_names, indexes=batch)[0]
                    for batch in index_batches
                ] if self._rnn else []
                return sampled, sampled_rnn

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

        # learning epochs
        for epoch in range(self.cfg.learning_epochs):
            epoch_kl_sum = torch.zeros((), dtype=torch.float64, device=self.device)
            epoch_kl_samples = 0
            epoch_optimizer_steps = 0
            sampled_batches, sampled_rnn_batches = sample_minibatches()

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
                    validate_scalar_output(
                        "PPO_RNN replay log_prob",
                        next_log_prob,
                        sampled_log_prob.shape[0],
                        synchronize=True,
                    )
                    lower, upper = 1.0 - self.cfg.ratio_clip, 1.0 + self.cfg.ratio_clip
                    log_ratio = next_log_prob - sampled_log_prob
                    ratio = torch.exp(log_ratio)
                    require_finite("PPO_RNN importance ratio", ratio, synchronize=True)

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

                    # compute value loss
                    predicted_values_raw, _ = self.value.act({**inputs, **rnn_value}, role="value")
                    validate_scalar_output(
                        "PPO_RNN replay value",
                        predicted_values_raw,
                        sampled_returns.shape[0],
                        synchronize=True,
                    )

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
                        predicted_values_clipped = sampled_values + torch.clip(
                            value_delta, min=-self.cfg.value_clip, max=self.cfg.value_clip
                        )
                        value_error = (sampled_returns - predicted_values_raw).pow(2)
                        value_error_clipped = (sampled_returns - predicted_values_clipped).pow(2)
                        value_loss = self.cfg.value_loss_scale * torch.maximum(
                            value_error, value_error_clipped
                        ).mean()
                    else:
                        value_clip_fraction = torch.zeros((), device=sampled_values.device)
                        value_loss = self.cfg.value_loss_scale * F.mse_loss(
                            sampled_returns, predicted_values_raw
                        )

                    total_loss = policy_loss + entropy_loss + value_loss
                    require_finite("PPO_RNN loss", total_loss, synchronize=True)

                # optimization step
                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()

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

                if not self.scaler.is_enabled():
                    require_finite("PPO_RNN gradient norm", grad_norm, synchronize=True)

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
        self.track_data("Policy / Initial replay max abs log-ratio", initial_replay_max_abs_log_ratio)

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
