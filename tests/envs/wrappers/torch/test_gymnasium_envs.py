import pytest

from collections.abc import Mapping
import gymnasium as gym
import numpy as np

import torch

from skrl.envs.wrappers.torch import wrap_env
from skrl.envs.wrappers.torch.gymnasium_envs import GymnasiumWrapper


def test_env(capsys: pytest.CaptureFixture):
    num_envs = 1
    action = torch.ones((num_envs, 1))

    # check wrapper definition
    assert isinstance(wrap_env(None, "gymnasium"), GymnasiumWrapper)

    # load wrap the environment
    original_env = gym.make("Pendulum-v1")
    env = wrap_env(original_env, "auto")
    assert isinstance(env, GymnasiumWrapper)
    env = wrap_env(original_env, "gymnasium")
    assert isinstance(env, GymnasiumWrapper)

    # check properties
    assert env.state_space is None
    assert isinstance(env.observation_space, gym.Space) and env.observation_space.shape == (3,)
    assert isinstance(env.action_space, gym.Space) and env.action_space.shape == (1,)
    assert isinstance(env.num_envs, int) and env.num_envs == num_envs
    assert isinstance(env.num_agents, int) and env.num_agents == 1
    assert isinstance(env.device, torch.device)
    # check internal properties
    assert env._env is original_env
    assert env._unwrapped is original_env.unwrapped
    # check methods
    for _ in range(2):
        observation, info = env.reset()
        state = env.state()
        assert isinstance(observation, torch.Tensor) and observation.shape == torch.Size([num_envs, 3])
        assert isinstance(info, Mapping)
        assert state is None
        for _ in range(3):
            observation, reward, terminated, truncated, info = env.step(action)
            state = env.state()
            env.render()
            assert isinstance(observation, torch.Tensor) and observation.shape == torch.Size([num_envs, 3])
            assert isinstance(reward, torch.Tensor) and reward.shape == torch.Size([num_envs, 1])
            assert isinstance(terminated, torch.Tensor) and terminated.shape == torch.Size([num_envs, 1])
            assert isinstance(truncated, torch.Tensor) and truncated.shape == torch.Size([num_envs, 1])
            assert isinstance(info, Mapping)
            assert state is None

    env.close()


@pytest.mark.parametrize("vectorization_mode", ["async", "sync"])
def test_vectorized_env(capsys: pytest.CaptureFixture, vectorization_mode: str):
    num_envs = 10
    action = torch.ones((num_envs, 1))

    # check wrapper definition
    assert isinstance(wrap_env(None, "gymnasium"), GymnasiumWrapper)

    # load wrap the environment
    try:
        make_vec_kwargs = {}
        if hasattr(gym.vector, "AutoresetMode"):
            make_vec_kwargs["vector_kwargs"] = {"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP}
        original_env = gym.make_vec(
            "Pendulum-v1",
            num_envs=num_envs,
            vectorization_mode=vectorization_mode,
            **make_vec_kwargs,
        )
    except AttributeError:
        original_env = gym.vector.make("Pendulum-v1", num_envs=num_envs, asynchronous=vectorization_mode == "async")
    env = wrap_env(original_env, "auto")
    assert isinstance(env, GymnasiumWrapper)
    env = wrap_env(original_env, "gymnasium")
    assert isinstance(env, GymnasiumWrapper)

    # check properties
    assert env.state_space is None
    assert isinstance(env.observation_space, gym.Space) and env.observation_space.shape == (3,)
    assert isinstance(env.action_space, gym.Space) and env.action_space.shape == (1,)
    assert isinstance(env.num_envs, int) and env.num_envs == num_envs
    assert isinstance(env.num_agents, int) and env.num_agents == 1
    assert isinstance(env.device, torch.device)
    # check internal properties
    assert env._env is original_env
    assert env._unwrapped is original_env.unwrapped
    assert env._vectorized is True
    # check methods
    for _ in range(2):
        observation, info = env.reset()
        state = env.state()
        observation, info = env.reset()  # edge case: vectorized environments are autoreset
        state = env.state()
        assert isinstance(observation, torch.Tensor) and observation.shape == torch.Size([num_envs, 3])
        assert isinstance(info, Mapping)
        assert state is None
        for _ in range(3):
            observation, reward, terminated, truncated, info = env.step(action)
            state = env.state()
            env.render()
            assert isinstance(observation, torch.Tensor) and observation.shape == torch.Size([num_envs, 3])
            assert isinstance(reward, torch.Tensor) and reward.shape == torch.Size([num_envs, 1])
            assert isinstance(terminated, torch.Tensor) and terminated.shape == torch.Size([num_envs, 1])
            assert isinstance(truncated, torch.Tensor) and truncated.shape == torch.Size([num_envs, 1])
            assert isinstance(info, Mapping)
            assert info["_skrl_autoreset"] is True
            assert state is None

    env.close()


def test_vectorized_next_step_autoreset_is_rejected():
    if not hasattr(gym.vector, "AutoresetMode"):
        pytest.skip("Gymnasium version does not expose configurable autoreset modes")
    original_env = gym.make_vec(
        "Pendulum-v1",
        num_envs=2,
        vectorization_mode="sync",
        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.NEXT_STEP},
    )
    try:
        with pytest.raises(ValueError, match="SAME_STEP"):
            GymnasiumWrapper(original_env)
    finally:
        original_env.close()


def test_same_step_timeout_is_explicitly_marked_as_returning_reset_observations():
    if not hasattr(gym.vector, "AutoresetMode"):
        pytest.skip("Gymnasium version does not expose configurable autoreset modes")
    original_env = gym.make_vec(
        "CartPole-v1",
        num_envs=2,
        vectorization_mode="sync",
        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP},
        max_episode_steps=1,
    )
    env = GymnasiumWrapper(original_env)
    try:
        env.reset()
        observation, _, terminated, truncated, info = env.step(torch.zeros((2, 1), dtype=torch.long))
        assert not terminated.any()
        assert truncated.all()
        assert info["_skrl_autoreset"] is True
        final_observation = torch.as_tensor(np.stack(info["final_obs"]), dtype=observation.dtype)
        assert not torch.equal(observation.cpu(), final_observation)
    finally:
        env.close()
