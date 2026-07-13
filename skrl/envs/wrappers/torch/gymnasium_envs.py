from __future__ import annotations

from typing import Any

import gymnasium

import torch

from skrl import config, logger
from skrl.envs.wrappers.torch.base import Wrapper
from skrl.utils.spaces.torch import (
    flatten_tensorized_space,
    tensorize_space,
    unflatten_tensorized_space,
    untensorize_space,
)


class GymnasiumWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Gymnasium environment wrapper.

        :param env: The environment instance to wrap.
        """
        super().__init__(env)

        self._seed = config.torch.key
        self._vectorized = False
        try:
            self._vectorized = self._vectorized or isinstance(env, gymnasium.vector.VectorEnv)
        except Exception as e:
            pass
        try:
            self._vectorized = self._vectorized or isinstance(env, gymnasium.experimental.vector.VectorEnv)
        except Exception as e:
            logger.warning(f"Failed to check for a vectorized environment: {e}")
        if self._vectorized:
            # skrl trainers expect the observation returned with a done transition to be
            # ready for the next action. Gymnasium's NEXT_STEP mode instead emits a
            # synthetic reset step on the following call, which would be recorded as an
            # ordinary PPO transition. Reject that mode rather than silently mixing it
            # into the rollout.
            autoreset_mode = getattr(env, "autoreset_mode", None)
            if autoreset_mode is not None:
                try:
                    same_step_mode = gymnasium.vector.AutoresetMode.SAME_STEP
                except AttributeError:
                    same_step_mode = None
                if same_step_mode is not None and autoreset_mode != same_step_mode:
                    raise ValueError(
                        "Gymnasium vector environments must use AutoresetMode.SAME_STEP with skrl"
                    )
            self._reset_once = True
            self._observation = None
            self._info = None

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space."""
        if self._vectorized:
            return self._env.single_observation_space
        return self._env.observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space."""
        if self._vectorized:
            return self._env.single_action_space
        return self._env.action_space

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        actions = untensorize_space(
            self.action_space,
            unflatten_tensorized_space(self.action_space, actions),
            squeeze_batch_dimension=not self._vectorized,
        )
        if self._vectorized and isinstance(self.action_space, gymnasium.spaces.Discrete):
            actions = actions.flatten()

        observation, reward, terminated, truncated, info = self._env.step(actions)

        # convert response to torch
        observation = flatten_tensorized_space(tensorize_space(self.observation_space, observation, device=self.device))
        reward = torch.tensor(reward, device=self.device, dtype=torch.float32).view(self.num_envs, -1)
        terminated = torch.tensor(terminated, device=self.device, dtype=torch.bool).view(self.num_envs, -1)
        truncated = torch.tensor(truncated, device=self.device, dtype=torch.bool).view(self.num_envs, -1)

        # save observation and info for vectorized envs
        if self._vectorized:
            if isinstance(info, dict):
                info = dict(info)
                # SAME_STEP returns reset observations for completed environments. PPO
                # timeout bootstrap must therefore use an explicitly preserved final state
                # or fail fast instead of evaluating this observation.
                info["_skrl_autoreset"] = True
            self._observation = observation
            self._info = info

        return observation, reward, terminated, truncated, info

    def state(self) -> torch.Tensor | None:
        """Get the environment state.

        :return: State.
        """
        try:
            return flatten_tensorized_space(
                tensorize_space(self.state_space, self._unwrapped.state(), device=self.device)
            )
        except:
            return None

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment.

        :return: Observation, info.
        """
        # handle vectorized environments (vector environments are autoreset)
        if self._vectorized:
            if self._reset_once:
                observation, self._info = self._env.reset(seed=self._seed)
                self._observation = flatten_tensorized_space(
                    tensorize_space(self.observation_space, observation, device=self.device)
                )
                self._reset_once = False
                self._seed = None
            return self._observation, self._info

        observation, info = self._env.reset(seed=self._seed)
        observation = flatten_tensorized_space(tensorize_space(self.observation_space, observation, device=self.device))
        self._seed = None
        return observation, info

    def render(self, *args, **kwargs) -> Any:
        """Render the environment."""
        if self._vectorized:
            return self._env.call("render", *args, **kwargs)
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment."""
        self._env.close()
