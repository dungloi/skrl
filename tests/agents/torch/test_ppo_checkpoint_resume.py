from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy

import gymnasium
import pytest
import torch

from skrl.agents.torch.ppo import PPO, PPO_RNN
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model


_ROLLOUTS = 4
_SEQUENCE_LENGTH = 2
_LEARNING_RATE = 3.0e-3


def _tiny_recurrence(model: Model, x: torch.Tensor, hidden: torch.Tensor):
    if model.training:
        x = x.view(-1, _SEQUENCE_LENGTH, 1)
        hidden = hidden.view(1, -1, _SEQUENCE_LENGTH, 1)[:, :, 0, :].contiguous()
        outputs = []
        for step in x.unbind(dim=1):
            hidden = hidden + step.unsqueeze(0)
            outputs.append(hidden.squeeze(0))
        return torch.stack(outputs, dim=1).flatten(0, 1), hidden
    hidden = hidden + x.unsqueeze(0)
    return hidden.squeeze(0), hidden


class _RecurrentPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, state_space, action_space):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device="cpu",
        )
        GaussianMixin.__init__(self, reduction="sum")
        self.gain = torch.nn.Parameter(torch.tensor([0.70]))
        self.bias = torch.nn.Parameter(torch.tensor([0.10]))
        self.log_std_parameter = torch.nn.Parameter(torch.tensor([-0.25]))

    def get_specification(self):
        return {"rnn": {"sequence_length": _SEQUENCE_LENGTH, "sizes": [(1, 1, 1)]}}

    def compute(self, inputs, role=""):
        signal = inputs["observations"][:, :1] * self.gain + self.bias
        output, hidden = _tiny_recurrence(self, signal, inputs["rnn"][0])
        return output, {"log_std": self.log_std_parameter, "rnn": [hidden]}


class _RecurrentValue(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device="cpu",
        )
        DeterministicMixin.__init__(self)
        self.gain = torch.nn.Parameter(torch.tensor([0.35]))
        self.bias = torch.nn.Parameter(torch.tensor([-0.20]))

    def get_specification(self):
        return {"rnn": {"sequence_length": _SEQUENCE_LENGTH, "sizes": [(1, 1, 1)]}}

    def compute(self, inputs, role=""):
        signal = inputs["states"][:, :1] * self.gain + self.bias
        output, hidden = _tiny_recurrence(self, signal, inputs["rnn"][0])
        return output, {"rnn": [hidden]}


def _make_agent(agent_class, *, seed: int):
    torch.manual_seed(seed)
    observation_space = gymnasium.spaces.Box(low=-20, high=20, shape=(2,))
    state_space = gymnasium.spaces.Box(low=-20, high=20, shape=(3,))
    action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(1,))

    if agent_class is PPO_RNN:
        policy = _RecurrentPolicy(observation_space, state_space, action_space)
        value = _RecurrentValue(observation_space, state_space, action_space)
    else:
        policy = gaussian_model(
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device="cpu",
            network=[{"name": "net", "input": "OBSERVATIONS", "layers": [7], "activations": "tanh"}],
            output="ACTIONS",
        )
        value = deterministic_model(
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device="cpu",
            network=[{"name": "net", "input": "STATES", "layers": [7], "activations": "tanh"}],
            output="ONE",
        )

    cfg = {
        "rollouts": _ROLLOUTS,
        "learning_epochs": 2,
        "mini_batches": 2,
        "discount_factor": 0.93,
        "lambda_": 0.81,
        "learning_rate": _LEARNING_RATE,
        "optimizer": "Adam",
        "optimizer_kwargs": {"betas": (0.83, 0.97)},
        "learning_rate_scheduler": torch.optim.lr_scheduler.StepLR,
        "learning_rate_scheduler_kwargs": {"step_size": 1, "gamma": 0.7},
        "observation_preprocessor": RunningStandardScaler,
        "observation_preprocessor_kwargs": {
            "size": observation_space,
            "device": "cpu",
            "clip_threshold": 100.0,
        },
        "state_preprocessor": RunningStandardScaler,
        "state_preprocessor_kwargs": {"size": state_space, "device": "cpu", "clip_threshold": 100.0},
        "value_preprocessor": RunningStandardScaler,
        "value_preprocessor_kwargs": {"size": 1, "device": "cpu", "clip_threshold": 100.0},
        "random_timesteps": 0,
        "learning_starts": 0,
        "grad_norm_clip": 1.0,
        "ratio_clip": 0.2,
        "value_clip": 0.2,
        "entropy_loss_scale": 0.01,
        "value_loss_scale": 0.5,
        "kl_threshold": 0.0,
        "time_limit_bootstrap": False,
        # CPU AMP gives a real, stateful GradScaler without requiring CUDA in CI.
        "mixed_precision": True,
        "experiment": {
            "directory": "",
            "experiment_name": "",
            "write_interval": 0,
            "checkpoint_interval": 0,
            "store_separately": False,
            "wandb": False,
            "wandb_kwargs": {},
        },
    }
    agent = agent_class(
        models={"policy": policy, "value": value},
        memory=RandomMemory(memory_size=_ROLLOUTS, num_envs=1, device="cpu"),
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device="cpu",
        cfg=cfg,
    )
    agent.init()
    return agent


def _fixed_batch(round_index: int):
    offset = 0.17 * round_index
    observations = torch.tensor(
        [[-0.90, 0.20], [-0.25, 0.75], [0.40, -0.55], [1.05, 0.30]], dtype=torch.float32
    )
    states = torch.tensor(
        [[-1.20, 0.10, 0.60], [-0.35, 0.85, -0.15], [0.45, -0.65, 1.30], [1.20, 0.25, -0.80]],
        dtype=torch.float32,
    )
    rewards = torch.tensor([[-0.55], [1.25], [0.35], [2.10]], dtype=torch.float32)
    return {
        "observations": observations + offset,
        "states": states - 0.5 * offset,
        "rewards": rewards + torch.tensor([[offset], [-offset], [0.5 * offset], [0.25 * offset]]),
        # End exactly at the checkpoint boundary. For PPO_RNN this resets both
        # policy and value hidden states before saving.
        "terminated": torch.tensor([[False], [False], [False], [True]]),
    }


def _run_update(agent, *, round_index: int):
    batch = _fixed_batch(round_index)
    actions = []
    log_probabilities = []
    agent.enable_models_training_mode(False)

    for timestep in range(_ROLLOUTS):
        observations = batch["observations"][timestep : timestep + 1]
        states = batch["states"][timestep : timestep + 1]
        next_observations = (
            batch["observations"][timestep + 1 : timestep + 2]
            if timestep + 1 < _ROLLOUTS
            else observations + torch.tensor([[0.20, -0.10]])
        )
        next_states = (
            batch["states"][timestep + 1 : timestep + 2]
            if timestep + 1 < _ROLLOUTS
            else states + torch.tensor([[0.15, -0.05, 0.10]])
        )
        with torch.no_grad():
            action, outputs = agent.act(observations, states, timestep=timestep, timesteps=_ROLLOUTS)
            actions.append(action.detach().clone())
            log_probabilities.append(outputs["log_prob"].detach().clone())
            agent.record_transition(
                observations=observations,
                states=states,
                actions=action,
                rewards=batch["rewards"][timestep : timestep + 1],
                next_observations=next_observations,
                next_states=next_states,
                terminated=batch["terminated"][timestep : timestep + 1],
                truncated=torch.zeros((1, 1), dtype=torch.bool),
                infos={},
                timestep=timestep,
                timesteps=_ROLLOUTS,
            )

    rollout_values = agent.memory.get_tensor_by_name("values").detach().clone()
    agent.enable_models_training_mode(True)
    agent.update(timestep=_ROLLOUTS - 1, timesteps=_ROLLOUTS)
    agent.enable_models_training_mode(False)

    return {
        "actions": torch.cat(actions),
        "log_probabilities": torch.cat(log_probabilities),
        "rollout_values": rollout_values,
        "memory": {name: tensor.detach().clone() for name, tensor in agent.memory.tensors.items()},
        "tracking": {name: values[-1] for name, values in agent.tracking_data.items()},
    }


def _assert_tree_equal(actual, expected, *, path: str = "root"):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), f"{path}: {type(actual)} != Tensor"
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=path)
    elif isinstance(expected, Mapping):
        assert isinstance(actual, Mapping), f"{path}: {type(actual)} != Mapping"
        assert actual.keys() == expected.keys(), f"{path}: {actual.keys()} != {expected.keys()}"
        for key in expected:
            _assert_tree_equal(actual[key], expected[key], path=f"{path}.{key}")
    elif isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        assert isinstance(actual, Sequence) and not isinstance(actual, (str, bytes)), path
        assert len(actual) == len(expected), f"{path}: {len(actual)} != {len(expected)}"
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_tree_equal(actual_item, expected_item, path=f"{path}[{index}]")
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=0, abs=0), f"{path}: {actual} != {expected}"
    else:
        assert actual == expected, f"{path}: {actual} != {expected}"


def _checkpoint_state(agent):
    return {
        name: copy.deepcopy(module.state_dict())
        for name, module in agent.checkpoint_modules.items()
        if hasattr(module, "state_dict")
    }


def _assert_checkpoint_modules_equal(actual, expected):
    assert actual.checkpoint_modules.keys() == expected.checkpoint_modules.keys()
    _assert_tree_equal(_checkpoint_state(actual), _checkpoint_state(expected), path="checkpoint")


_EXPECTED_CHECKPOINT_MODULES = {
    "policy",
    "value",
    "optimizer",
    "scheduler",
    "grad_scaler",
    "observation_preprocessor",
    "state_preprocessor",
    "value_preprocessor",
}


@pytest.mark.parametrize("agent_class", [PPO, PPO_RNN])
def test_checkpoint_round_trip_preserves_every_ppo_training_module(agent_class, tmp_path):
    agent = _make_agent(agent_class, seed=101)
    # The generated feed-forward models contain lazy layers. Run one genuine
    # update before taking the comparison snapshot so every parameter is materialized.
    _run_update(agent, round_index=0)
    initial_policy = copy.deepcopy(agent.policy.state_dict())
    initial_value = copy.deepcopy(agent.value.state_dict())
    _run_update(agent, round_index=1)

    assert set(agent.checkpoint_modules) == _EXPECTED_CHECKPOINT_MODULES
    assert agent.optimizer.state_dict()["state"]
    assert agent.scheduler.last_epoch > 0
    assert agent.optimizer.param_groups[0]["lr"] < _LEARNING_RATE
    assert agent.scaler.is_enabled()
    assert agent.scaler.state_dict()["_growth_tracker"] > 0
    assert agent._observation_preprocessor.current_count.item() > 1
    assert agent._state_preprocessor.current_count.item() > 1
    assert agent._value_preprocessor.current_count.item() > 1
    assert any(not torch.equal(value, initial_policy[name]) for name, value in agent.policy.state_dict().items())
    assert any(not torch.equal(value, initial_value[name]) for name, value in agent.value.state_dict().items())

    checkpoint_path = tmp_path / f"{agent_class.__name__}.pt"
    agent.save(str(checkpoint_path))
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert set(payload) == _EXPECTED_CHECKPOINT_MODULES

    resumed = _make_agent(agent_class, seed=999)
    resumed.load(str(checkpoint_path))
    _assert_checkpoint_modules_equal(resumed, agent)


@pytest.mark.parametrize("agent_class", [PPO, PPO_RNN])
def test_checkpoint_resume_matches_the_next_nonzero_ppo_update(agent_class, tmp_path):
    uninterrupted = _make_agent(agent_class, seed=211)
    _run_update(uninterrupted, round_index=0)
    _run_update(uninterrupted, round_index=1)

    checkpoint_path = tmp_path / f"resume_{agent_class.__name__}.pt"
    uninterrupted.save(str(checkpoint_path))
    resumed = _make_agent(agent_class, seed=733)
    resumed.load(str(checkpoint_path))

    # The generic agent checkpoint intentionally does not own environment or
    # process RNG state. Give both continuations the same explicit RNG state so
    # this test isolates checkpointed PPO training state.
    continuation_rng = torch.get_rng_state().clone()
    torch.set_rng_state(continuation_rng)
    uninterrupted_result = _run_update(uninterrupted, round_index=2)
    rng_after_uninterrupted = torch.get_rng_state().clone()

    torch.set_rng_state(continuation_rng)
    resumed_result = _run_update(resumed, round_index=2)
    rng_after_resumed = torch.get_rng_state().clone()

    _assert_tree_equal(resumed_result, uninterrupted_result, path="continuation")
    _assert_checkpoint_modules_equal(resumed, uninterrupted)
    torch.testing.assert_close(rng_after_resumed, rng_after_uninterrupted, rtol=0, atol=0)
