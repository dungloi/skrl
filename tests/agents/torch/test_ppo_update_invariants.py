import pytest

from unittest.mock import Mock
import gymnasium

import torch

from skrl import config
from skrl.agents.torch.ppo import PPO, PPO_RNN
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler, SelectiveRunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model


class RecordingRunningStandardScaler(RunningStandardScaler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events = []

    def forward(self, x, *, train=False, inverse=False, no_grad=True):
        return super().forward(x, train=train, inverse=inverse, no_grad=no_grad)

    def update_stats(self, x):
        self.events.append(("train", self.current_count.item()))
        super().update_stats(x)


class RecordingSelectiveRunningStandardScaler(SelectiveRunningStandardScaler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events = []

    def forward(self, x, *, train=False, inverse=False, no_grad=True):
        return super().forward(x, train=train, inverse=inverse, no_grad=no_grad)

    def update_stats(self, x):
        self.events.append(("train", self.current_count.item()))
        super().update_stats(x)


class AlwaysSkippingGradScaler:
    def __init__(self):
        self._scale = 2.0

    def is_enabled(self):
        return True

    def get_scale(self):
        return self._scale

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        pass

    def update(self):
        self._scale /= 2


_SEQUENCE_LENGTH = 2


def _tiny_recurrence(model, x, hidden):
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


class TinyRecurrentPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, state_space, action_space):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device="cpu",
        )
        GaussianMixin.__init__(self, reduction="sum")
        self.gain = torch.nn.Parameter(torch.ones(1))
        self.log_std_parameter = torch.nn.Parameter(torch.zeros(1))

    def get_specification(self):
        return {"rnn": {"sequence_length": _SEQUENCE_LENGTH, "sizes": [(1, 1, 1)]}}

    def compute(self, inputs, role=""):
        output, hidden = _tiny_recurrence(self, inputs["observations"][:, :1] * self.gain, inputs["rnn"][0])
        return output, {"log_std": self.log_std_parameter, "rnn": [hidden]}


class TinyRecurrentValue(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device="cpu",
        )
        DeterministicMixin.__init__(self)
        self.gain = torch.nn.Parameter(torch.tensor([0.25]))

    def get_specification(self):
        return {"rnn": {"sequence_length": _SEQUENCE_LENGTH, "sizes": [(1, 1, 1)]}}

    def compute(self, inputs, role=""):
        output, hidden = _tiny_recurrence(self, inputs["states"][:, :1] * self.gain, inputs["rnn"][0])
        return output, {"rnn": [hidden]}


@pytest.mark.parametrize("scaler_class", [RunningStandardScaler, SelectiveRunningStandardScaler])
def test_running_input_scaler_accepts_a_single_rollout_sample(scaler_class):
    scaler = scaler_class(size=2, device="cpu")
    scaler(torch.tensor([[2.0, -3.0]]), train=True)

    assert scaler.current_count.item() == 2
    assert torch.isfinite(scaler.running_mean).all()
    assert torch.isfinite(scaler.running_variance).all()


@pytest.mark.parametrize("scaler_class", [RunningStandardScaler, SelectiveRunningStandardScaler])
def test_running_scaler_distributed_update_matches_global_batch_moments(scaler_class, monkeypatch):
    local = torch.tensor([[1.0, -2.0], [3.0, 4.0]], dtype=torch.float64)
    remote = torch.tensor([[-5.0, 6.0], [7.0, 8.0], [9.0, -10.0]], dtype=torch.float64)
    expected = scaler_class(size=2, device="cpu")
    expected.update_stats(torch.cat((local, remote), dim=0))
    actual = scaler_class(size=2, device="cpu")

    remote64 = remote.double()
    remote_moments = torch.cat(
        (
            torch.tensor([float(remote.shape[0])], dtype=torch.float64),
            remote64.sum(dim=0),
            remote64.square().sum(dim=0),
        )
    )

    def add_remote_moments(moments, op):
        assert op == torch.distributed.ReduceOp.SUM
        moments.add_(remote_moments)

    monkeypatch.setattr(config.torch, "_is_distributed", True)
    monkeypatch.setattr(torch.distributed, "all_reduce", add_remote_moments)
    actual.update_stats_distributed(local)

    torch.testing.assert_close(actual.running_mean, expected.running_mean, rtol=0, atol=1e-12)
    torch.testing.assert_close(actual.running_variance, expected.running_variance, rtol=0, atol=1e-12)
    torch.testing.assert_close(actual.current_count, expected.current_count, rtol=0, atol=0)


def _make_agent(
    agent_class,
    *,
    learning_epochs,
    kl_threshold=0.0,
    scheduler=False,
    learning_rate=0.0,
    recurrent=False,
    input_preprocessor=RecordingRunningStandardScaler,
    rollouts=4,
    mini_batches=2,
):
    torch.manual_seed(0)

    observation_space = gymnasium.spaces.Box(low=-100, high=100, shape=(2,))
    state_space = gymnasium.spaces.Box(low=-100, high=100, shape=(3,))
    action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(1,))

    if recurrent:
        policy = TinyRecurrentPolicy(observation_space, state_space, action_space)
        value = TinyRecurrentValue(observation_space, state_space, action_space)
    else:
        policy = gaussian_model(
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device="cpu",
            network=[{"name": "net", "input": "OBSERVATIONS", "layers": [8], "activations": "tanh"}],
            output="ACTIONS",
        )
        value = deterministic_model(
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device="cpu",
            network=[{"name": "net", "input": "STATES", "layers": [8], "activations": "tanh"}],
            output="ONE",
        )
    memory = RandomMemory(memory_size=rollouts, num_envs=1, device="cpu")
    cfg = {
        "rollouts": rollouts,
        "learning_epochs": learning_epochs,
        "mini_batches": mini_batches,
        "learning_rate": learning_rate,
        "learning_rate_scheduler": KLAdaptiveLR if scheduler else None,
        "learning_rate_scheduler_kwargs": {"kl_threshold": 0.01} if scheduler else {},
        "observation_preprocessor": input_preprocessor,
        "observation_preprocessor_kwargs": {
            "size": observation_space,
            "device": "cpu",
            "clip_threshold": 100.0,
        },
        "state_preprocessor": input_preprocessor,
        "state_preprocessor_kwargs": {"size": state_space, "device": "cpu", "clip_threshold": 100.0},
        "value_preprocessor": None,
        "random_timesteps": 0,
        "learning_starts": 0,
        "grad_norm_clip": 0.0,
        "ratio_clip": 0.2,
        "value_clip": 0.2,
        "entropy_loss_scale": 0.0,
        "value_loss_scale": 0.5,
        "kl_threshold": kl_threshold,
        "mixed_precision": False,
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
        memory=memory,
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device="cpu",
        cfg=cfg,
    )
    agent.init()
    return agent


def _collect_rollout(agent, sample_count=4):
    observations = torch.tensor([[1.0, -1.5], [2.0, -0.5], [3.0, 0.5], [4.0, 1.5], [5.0, 2.5], [6.0, 3.5]])
    states = torch.tensor(
        [
            [-2.0, 0.0, 2.0],
            [-1.0, 1.0, 3.0],
            [0.0, 2.0, 4.0],
            [1.0, 3.0, 5.0],
            [2.0, 4.0, 6.0],
            [3.0, 5.0, 7.0],
        ]
    )
    observations = observations[:sample_count]
    states = states[:sample_count]
    for timestep, (observation, state) in enumerate(zip(observations, states)):
        observation = observation.unsqueeze(0)
        state = state.unsqueeze(0)
        with torch.no_grad():
            actions, _ = agent.act(observation, state, timestep=timestep, timesteps=sample_count)
            agent.record_transition(
                observations=observation,
                states=state,
                actions=actions,
                rewards=torch.zeros((1, 1)),
                next_observations=observation + 0.25,
                next_states=state + 0.25,
                terminated=torch.zeros((1, 1), dtype=torch.bool),
                truncated=torch.zeros((1, 1), dtype=torch.bool),
                infos={},
                timestep=timestep,
                timesteps=sample_count,
            )
    return observations, states


def test_feedforward_ppo_rnn_new_run_discards_partial_rollout():
    agent = _make_agent(PPO_RNN, learning_epochs=1, recurrent=False, rollouts=4)
    assert not agent._rnn
    _collect_rollout(agent, sample_count=2)
    agent._rollout = 2
    assert len(agent.memory) == 2
    assert agent._current_next_observations is not None

    agent.pre_interaction(timestep=0, timesteps=4)

    assert len(agent.memory) == 0
    assert agent._rollout == 0
    assert agent._current_next_observations is None
    assert agent._current_next_states is None
    assert agent._current_log_prob is None


@pytest.mark.parametrize("agent_class", [PPO, PPO_RNN])
@pytest.mark.parametrize("input_preprocessor", [RunningStandardScaler, SelectiveRunningStandardScaler])
def test_single_sample_ppo_update_stays_finite(agent_class, input_preprocessor):
    agent = _make_agent(
        agent_class,
        learning_epochs=1,
        input_preprocessor=input_preprocessor,
        rollouts=1,
        mini_batches=2,
    )
    _collect_rollout(agent, sample_count=1)
    agent.enable_models_training_mode(True)
    agent.update(timestep=0, timesteps=1)

    assert agent._observation_preprocessor.current_count.item() == 2
    assert agent._state_preprocessor.current_count.item() == 2
    for model in (agent.policy, agent.value):
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_ppo_minibatches_include_rollout_remainder_samples():
    agent = _make_agent(PPO, learning_epochs=1, rollouts=5, mini_batches=2)
    _collect_rollout(agent, sample_count=5)

    replay_samples = 0
    original_policy_act = agent.policy.act

    def recording_policy_act(inputs, *, role=""):
        nonlocal replay_samples
        if "taken_actions" in inputs:
            replay_samples += inputs["taken_actions"].shape[0]
        return original_policy_act(inputs, role=role)

    agent.policy.act = recording_policy_act
    agent.enable_models_training_mode(True)
    agent.update(timestep=4, timesteps=5)

    assert replay_samples == 5
    assert agent.tracking_data["Optimization / Observed minibatches"][-1] == 2
    assert torch.isfinite(agent._observation_preprocessor.running_variance).all()


@pytest.mark.parametrize("agent_class", [PPO, PPO_RNN])
@pytest.mark.parametrize(
    "input_preprocessor", [RecordingRunningStandardScaler, RecordingSelectiveRunningStandardScaler]
)
def test_input_scalers_are_frozen_until_the_ppo_update_finishes(agent_class, input_preprocessor):
    agent = _make_agent(agent_class, learning_epochs=2, input_preprocessor=input_preprocessor)
    observations, states = _collect_rollout(agent)

    events = agent._observation_preprocessor.events
    original_policy_act = agent.policy.act

    def recording_policy_act(inputs, role):
        if "taken_actions" in inputs:
            events.append(("policy", agent._observation_preprocessor.current_count.item()))
        return original_policy_act(inputs, role=role)

    agent.policy.act = recording_policy_act
    parameters_before = [parameter.detach().clone() for parameter in agent.policy.parameters()]

    agent.enable_models_training_mode(True)
    agent.update(timestep=3, timesteps=4)

    event_names = [name for name, _ in events]
    assert event_names[:4] == ["policy"] * 4
    assert event_names[4:] and set(event_names[4:]) == {"train"}
    assert all(count == 1 for name, count in events if name == "policy")
    assert agent.tracking_data["Policy / Approx KL"][-1] == pytest.approx(0.0, abs=1e-7)
    assert agent.tracking_data["Policy / Clip fraction"][-1] == 0.0
    assert agent.tracking_data["Policy / IS ratio (mean)"][-1] == pytest.approx(1.0, abs=1e-7)

    assert agent._observation_preprocessor.current_count.item() == 5
    assert agent._state_preprocessor.current_count.item() == 5
    assert torch.allclose(agent._observation_preprocessor.running_mean.float(), observations.sum(dim=0) / 5)
    assert torch.allclose(agent._state_preprocessor.running_mean.float(), states.sum(dim=0) / 5)
    for parameter_before, parameter_after in zip(parameters_before, agent.policy.parameters()):
        assert torch.equal(parameter_before, parameter_after)


def test_recurrent_replay_uses_the_rollout_scaler_and_pre_action_hidden_states():
    agent = _make_agent(PPO_RNN, learning_epochs=2, recurrent=True, mini_batches=4)
    _collect_rollout(agent)

    policy_hidden = agent.memory.get_tensor_by_name("rnn_policy_0").flatten(0, 1).flatten()
    value_hidden = agent.memory.get_tensor_by_name("rnn_value_0").flatten(0, 1).flatten()
    assert torch.equal(policy_hidden, torch.tensor([0.0, 1.0, 3.0, 6.0]))
    assert torch.equal(value_hidden, torch.tensor([0.0, -0.5, -0.75, -0.75]))

    parameters_before = [
        parameter.detach().clone() for model in (agent.policy, agent.value) for parameter in model.parameters()
    ]
    agent.enable_models_training_mode(True)
    agent.update(timestep=3, timesteps=4)

    assert agent.tracking_data["Policy / Approx KL"][-1] == pytest.approx(0.0, abs=1e-7)
    assert agent.tracking_data["Policy / Clip fraction"][-1] == 0.0
    assert agent.tracking_data["Policy / IS ratio (mean)"][-1] == pytest.approx(1.0, abs=1e-7)
    assert agent._observation_preprocessor.current_count.item() == 5
    assert agent._state_preprocessor.current_count.item() == 5
    parameters_after = [parameter for model in (agent.policy, agent.value) for parameter in model.parameters()]
    assert all(torch.equal(before, after) for before, after in zip(parameters_before, parameters_after))


def test_recurrent_minibatches_are_split_on_complete_sequence_boundaries():
    agent = _make_agent(PPO_RNN, learning_epochs=1, recurrent=True, rollouts=6, mini_batches=2)
    _collect_rollout(agent, sample_count=6)
    agent.enable_models_training_mode(True)
    agent.update(timestep=5, timesteps=6)

    assert agent.tracking_data["Optimization / Observed minibatches"][-1] == 2
    assert agent.tracking_data["Policy / Approx KL"][-1] == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("agent_class", [PPO, PPO_RNN])
@pytest.mark.parametrize(
    ("stop_on_call", "expected_effective_minibatches", "expected_observed_minibatches", "expected_scheduler_calls"),
    [(1, 0, 1, 0), (2, 1, 2, 1)],
)
def test_kl_early_stop_ends_all_epochs_and_does_not_reschedule_without_an_optimizer_step(
    agent_class,
    stop_on_call,
    expected_effective_minibatches,
    expected_observed_minibatches,
    expected_scheduler_calls,
):
    agent = _make_agent(
        agent_class,
        learning_epochs=5,
        kl_threshold=0.01,
        scheduler=True,
        learning_rate=1e-3,
    )
    _collect_rollout(agent)

    replay_calls = 0
    original_policy_act = agent.policy.act

    def policy_act_with_controlled_kl(inputs, *, role=""):
        nonlocal replay_calls
        actions, outputs = original_policy_act(inputs, role=role)
        if "taken_actions" in inputs:
            replay_calls += 1
            if replay_calls == stop_on_call:
                outputs["log_prob"] = outputs["log_prob"] + 5.0
        return actions, outputs

    agent.policy.act = policy_act_with_controlled_kl

    scheduler_step = Mock(wraps=agent.scheduler.step)
    agent.scheduler.step = scheduler_step
    agent.enable_models_training_mode(True)
    agent.update(timestep=3, timesteps=4)

    assert agent.tracking_data["Optimization / KL early-stop count"][-1] == 1
    assert agent.tracking_data["Optimization / Effective minibatches"][-1] == expected_effective_minibatches
    assert agent.tracking_data["Optimization / Observed minibatches"][-1] == expected_observed_minibatches
    assert scheduler_step.call_count == expected_scheduler_calls
    assert agent._observation_preprocessor.current_count.item() == 5
    assert agent._state_preprocessor.current_count.item() == 5


@pytest.mark.parametrize("agent_class", [PPO, PPO_RNN])
def test_kl_scheduler_observes_a_terminal_kl_caused_by_the_previous_epoch(agent_class):
    agent = _make_agent(
        agent_class,
        learning_epochs=5,
        kl_threshold=0.01,
        scheduler=True,
        learning_rate=0.0,
    )
    _collect_rollout(agent)

    replay_calls = 0
    original_policy_act = agent.policy.act

    def policy_act_with_terminal_kl(inputs, *, role=""):
        nonlocal replay_calls
        actions, outputs = original_policy_act(inputs, role=role)
        if "taken_actions" in inputs:
            replay_calls += 1
            if replay_calls == 3:
                outputs["log_prob"] = outputs["log_prob"] + 5.0
        return actions, outputs

    agent.policy.act = policy_act_with_terminal_kl
    scheduler_step = Mock(wraps=agent.scheduler.step)
    agent.scheduler.step = scheduler_step
    agent.enable_models_training_mode(True)
    agent.update(timestep=3, timesteps=4)

    assert replay_calls == 3
    assert agent.tracking_data["Optimization / KL early-stop count"][-1] == 1
    assert agent.tracking_data["Optimization / Effective minibatches"][-1] == 2
    assert agent.tracking_data["Optimization / Observed minibatches"][-1] == 3
    # Once for the completed first epoch and once for the newly observed terminal KL.
    assert scheduler_step.call_count == 2


def test_non_kl_scheduler_does_not_advance_a_terminal_epoch_without_an_optimizer_step():
    agent = _make_agent(PPO, learning_epochs=5, kl_threshold=0.01, learning_rate=0.0)
    _collect_rollout(agent)

    replay_calls = 0
    original_policy_act = agent.policy.act

    def policy_act_with_terminal_kl(inputs, *, role=""):
        nonlocal replay_calls
        actions, outputs = original_policy_act(inputs, role=role)
        if "taken_actions" in inputs:
            replay_calls += 1
            if replay_calls == 3:
                outputs["log_prob"] = outputs["log_prob"] + 5.0
        return actions, outputs

    agent.policy.act = policy_act_with_terminal_kl
    agent.scheduler = torch.optim.lr_scheduler.ConstantLR(agent.optimizer, factor=0.5, total_iters=5)
    scheduler_step = Mock(wraps=agent.scheduler.step)
    agent.scheduler.step = scheduler_step
    agent.enable_models_training_mode(True)
    agent.update(timestep=3, timesteps=4)

    assert replay_calls == 3
    assert agent.tracking_data["Optimization / Effective minibatches"][-1] == 2
    # The scheduler advances for the completed first epoch, but not for the terminal second epoch.
    assert scheduler_step.call_count == 1


@pytest.mark.parametrize("agent_class", [PPO, PPO_RNN])
def test_scheduler_does_not_advance_when_grad_scaler_skips_all_optimizer_steps(agent_class):
    agent = _make_agent(agent_class, learning_epochs=1, learning_rate=1e-3)
    _collect_rollout(agent)

    parameters_before = [
        parameter.detach().clone() for model in (agent.policy, agent.value) for parameter in model.parameters()
    ]
    agent.scaler = AlwaysSkippingGradScaler()
    agent.scheduler = torch.optim.lr_scheduler.ConstantLR(agent.optimizer, factor=0.5, total_iters=5)
    scheduler_step = Mock(wraps=agent.scheduler.step)
    agent.scheduler.step = scheduler_step
    agent.enable_models_training_mode(True)
    agent.update(timestep=3, timesteps=4)

    parameters_after = [parameter for model in (agent.policy, agent.value) for parameter in model.parameters()]
    assert all(torch.equal(before, after) for before, after in zip(parameters_before, parameters_after))
    assert scheduler_step.call_count == 0
