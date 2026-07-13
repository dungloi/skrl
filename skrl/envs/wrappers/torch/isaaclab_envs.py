from __future__ import annotations

from typing import Any

import gymnasium

import torch

from skrl import config
from skrl.envs.wrappers.torch.base import MultiAgentEnvWrapper, Wrapper
from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space, unflatten_tensorized_space


class IsaacLabWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Isaac Lab environment wrapper.

        :param env: The environment instance to wrap.
        """
        super().__init__(env)

        self._seed = config.torch.key
        self._reset_once = True
        self._observations = None
        self._states = None
        self._info = {}

    @property
    def state_space(self) -> gymnasium.Space | None:
        """State space."""
        try:
            return self._unwrapped.single_observation_space["critic"]
        except KeyError:
            pass
        try:
            return self._unwrapped.state_space
        except AttributeError:
            return None

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space."""
        try:
            return self._unwrapped.single_observation_space["policy"]
        except:
            return self._unwrapped.observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space."""
        try:
            return self._unwrapped.single_action_space
        except:
            return self._unwrapped.action_space

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        actions = unflatten_tensorized_space(self.action_space, actions)
        with torch.no_grad():
            observations, reward, terminated, truncated, self._info = self._env.step(actions)
        if isinstance(self._info, dict):
            # Isaac Lab computes returned observations after automatically resetting done
            # environments, so they cannot be used as terminal states for timeout bootstrap.
            self._info = dict(self._info)
            self._info["_skrl_autoreset"] = True
        self._observations = flatten_tensorized_space(tensorize_space(self.observation_space, observations["policy"]))
        states = observations.get("critic", None)
        if states is not None:
            self._states = flatten_tensorized_space(tensorize_space(self.state_space, states))
        return self._observations, reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), self._info

    def state(self) -> torch.Tensor | None:
        """Get the environment state.

        :return: State.
        """
        return self._states

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment.

        :return: Observation, info.
        """
        if self._reset_once:
            observations, self._info = self._env.reset(seed=self._seed)
            self._observations = flatten_tensorized_space(
                tensorize_space(self.observation_space, observations["policy"])
            )
            states = observations.get("critic", None)
            if states is not None:
                self._states = flatten_tensorized_space(tensorize_space(self.state_space, states))
            self._reset_once = False
            self._seed = None
        return self._observations, self._info

    def render(self, *args, **kwargs) -> None:
        """Render the environment."""
        return None

    def close(self) -> None:
        """Close the environment."""
        self._env.close()


class IsaacLabMultiAgentWrapper(MultiAgentEnvWrapper):
    def __init__(self, env: Any) -> None:
        """Isaac Lab environment wrapper for multi-agent implementation.

        :param env: The environment instance to wrap.
        """
        super().__init__(env)

        self._seed = config.torch.key
        self._reset_once = True
        self._observations = None
        self._info = {}

        # 判断是否需要从 observations 中获取 state
        if (self._unwrapped.state_space is None) or (self._unwrapped.state_space.shape[0] == 0):
            self._get_state_from_observations = True
            self._states = None
        else:
            self._get_state_from_observations = False

    def step(self, actions: dict[str, torch.Tensor]) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, Any],
    ]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        actions = {k: unflatten_tensorized_space(self.action_spaces[k], v) for k, v in actions.items()}
        with torch.no_grad():
            observations, rewards, terminated, truncated, self._info = self._env.step(actions)
        if self._get_state_from_observations:
            self._observations = {
                k: flatten_tensorized_space(tensorize_space(self.observation_spaces[k], v["policy"]))
                for k, v in observations.items()
            }
            self._states = {
                k: flatten_tensorized_space(tensorize_space(self.state_spaces[k], v["critic"]))
                for k, v in observations.items()
            }
        else:
            self._observations = {
                k: flatten_tensorized_space(tensorize_space(self.observation_spaces[k], v))
                for k, v in observations.items()
            }
        return (
            self._observations,
            {k: v.view(-1, 1) for k, v in rewards.items()},
            {k: v.view(-1, 1) for k, v in terminated.items()},
            {k: v.view(-1, 1) for k, v in truncated.items()},
            self._info,
        )

    def reset(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Reset the environment.

        :return: Observation, info.
        """
        if self._reset_once:
            observations, self._info = self._env.reset(seed=self._seed)
            if self._get_state_from_observations:
                self._observations = {
                    k: flatten_tensorized_space(tensorize_space(self.observation_spaces[k], v["policy"]))
                    for k, v in observations.items()
                }
                self._states = {
                    k: flatten_tensorized_space(tensorize_space(self.state_spaces[k], v["critic"]))
                    for k, v in observations.items()
                }
            else:
                self._observations = {
                    k: flatten_tensorized_space(tensorize_space(self.observation_spaces[k], v))
                    for k, v in observations.items()
                }
            self._reset_once = False
            self._seed = None
        return self._observations, self._info

    def state(self) -> dict[str, torch.Tensor | None]:
        """Get the environment state.

        :return: State.
        """
        if self._get_state_from_observations:
            return self._states
        else:
            try:
                state = self._env.state()
            except AttributeError:  # 'OrderEnforcing' object has no attribute 'state'
                state = self._unwrapped.state()
            if state is not None:
                state = flatten_tensorized_space(tensorize_space(next(iter(self.state_spaces.values())), state))
            return {uid: state for uid in self.possible_agents}

    def render(self, *args, **kwargs) -> None:
        """Render the environment."""
        return None

    def close(self) -> None:
        """Close the environment."""
        self._env.close()

    @property
    def state_spaces(self) -> dict[str, gymnasium.Space | None]:
        """State spaces."""
        if self._get_state_from_observations:
            return {agent: space["critic"] for agent, space in self._unwrapped.observation_spaces.items()}
        else:
            return {agent: self._unwrapped.state_space for agent in self.possible_agents}

    @property
    def observation_spaces(self) -> dict[str, gymnasium.Space]:
        """Observation spaces."""
        if self._get_state_from_observations:
            return {agent: space["policy"] for agent, space in self._unwrapped.observation_spaces.items()}
        else:
            return self._unwrapped.observation_spaces
