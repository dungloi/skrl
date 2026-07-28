import datetime
from unittest.mock import Mock

import gymnasium
import pytest

import torch

from skrl import config
from skrl.agents.torch.ppo import PPO, PPO_RNN
from skrl.agents.torch.ppo._utils import synchronized_grad_scaler_step
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
    def __init__(self, observation_space, state_space, action_space, device="cpu"):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
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
    def __init__(self, observation_space, state_space, action_space, device="cpu"):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
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
    device="cpu",
    mixed_precision=False,
):
    torch.manual_seed(0)

    observation_space = gymnasium.spaces.Box(low=-100, high=100, shape=(2,))
    state_space = gymnasium.spaces.Box(low=-100, high=100, shape=(3,))
    action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(1,))

    if recurrent:
        policy = TinyRecurrentPolicy(observation_space, state_space, action_space, device=device)
        value = TinyRecurrentValue(observation_space, state_space, action_space, device=device)
    else:
        policy = gaussian_model(
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            network=[{"name": "net", "input": "OBSERVATIONS", "layers": [8], "activations": "tanh"}],
            output="ACTIONS",
        )
        value = deterministic_model(
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            network=[{"name": "net", "input": "STATES", "layers": [8], "activations": "tanh"}],
            output="ONE",
        )
    memory = RandomMemory(memory_size=rollouts, num_envs=1, device=device)
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
            "device": device,
            "clip_threshold": 100.0,
        },
        "state_preprocessor": input_preprocessor,
        "state_preprocessor_kwargs": {"size": state_space, "device": device, "clip_threshold": 100.0},
        "value_preprocessor": None,
        "random_timesteps": 0,
        "learning_starts": 0,
        "grad_norm_clip": 0.0,
        "ratio_clip": 0.2,
        "value_clip": 0.2,
        "entropy_loss_scale": 0.0,
        "value_loss_scale": 0.5,
        "kl_threshold": kl_threshold,
        "mixed_precision": mixed_precision,
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
        device=device,
        cfg=cfg,
    )
    agent.init()
    return agent


def _collect_rollout(agent, sample_count=4):
    observations = torch.tensor(
        [[1.0, -1.5], [2.0, -0.5], [3.0, 0.5], [4.0, 1.5], [5.0, 2.5], [6.0, 3.5]],
        device=agent.device,
    )
    states = torch.tensor(
        [
            [-2.0, 0.0, 2.0],
            [-1.0, 1.0, 3.0],
            [0.0, 2.0, 4.0],
            [1.0, 3.0, 5.0],
            [2.0, 4.0, 6.0],
            [3.0, 5.0, 7.0],
        ],
        device=agent.device,
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
                rewards=torch.zeros((1, 1), device=agent.device),
                next_observations=observation + 0.25,
                next_states=state + 0.25,
                terminated=torch.zeros((1, 1), dtype=torch.bool, device=agent.device),
                truncated=torch.zeros((1, 1), dtype=torch.bool, device=agent.device),
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


def test_grad_scaler_gates_an_overflow_created_while_unscaling():
    """A scale below one can turn a finite scaled gradient into Inf during unscale_."""

    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD((parameter,), lr=1.0)
    scaler = torch.amp.GradScaler(device="cpu", init_scale=0.5, enabled=True)
    scaler.scale(parameter.sum())  # lazily initialize the scale tensor
    parameter.grad = torch.full_like(parameter, torch.finfo(parameter.dtype).max)
    scaler.unscale_(optimizer)
    assert torch.isinf(parameter.grad).all()

    scale_before = scaler.get_scale()
    optimizer_step_succeeded = synchronized_grad_scaler_step(
        scaler=scaler, optimizer=optimizer, grad_norm=parameter.grad.norm()
    )

    assert not optimizer_step_succeeded
    assert torch.equal(parameter, torch.zeros_like(parameter))
    assert scaler.get_scale() == pytest.approx(scale_before * 0.5)


def test_distributed_grad_scaler_scale_mismatch_fails_before_optimizer_step(monkeypatch):
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD((parameter,), lr=1.0)
    scaler = torch.amp.GradScaler(device="cpu", init_scale=8.0, enabled=True)
    scaler.scale(parameter.square().sum()).backward()
    scaler.unscale_(optimizer)
    parameter_before = parameter.detach().clone()

    monkeypatch.setattr(config.torch, "_is_distributed", True)

    def emulate_smaller_remote_scale(step_control, op):
        assert op == torch.distributed.ReduceOp.MIN
        # Local scale is 8. The combined MIN reduction for a remote scale of 4 has
        # min(scale)=4 and min(-scale)=-8, exposing the mismatch on every rank.
        step_control[2] = 4.0
        step_control[3] = -8.0

    monkeypatch.setattr(torch.distributed, "all_reduce", emulate_smaller_remote_scale)
    with pytest.raises(RuntimeError, match="scale differs across distributed ranks"):
        synchronized_grad_scaler_step(
            scaler=scaler, optimizer=optimizer, grad_norm=parameter.grad.norm()
        )

    assert torch.equal(parameter, parameter_before)
    assert scaler.get_scale() == 8.0
    assert not optimizer.state


@pytest.mark.parametrize(
    ("agent_class", "recurrent"),
    [(PPO, False), (PPO_RNN, True)],
    ids=["ppo", "ppo-rnn"],
)
def test_remote_amp_overflow_is_synchronized_before_any_optimizer_step(
    agent_class, recurrent, monkeypatch
):
    """A remote-only overflow must skip parameters, scaler growth and scheduler on this rank."""

    agent = _make_agent(
        agent_class,
        learning_epochs=1,
        learning_rate=1e-3,
        recurrent=recurrent,
        mixed_precision=True,
    )
    _collect_rollout(agent)
    parameters_before = [
        parameter.detach().clone() for model in (agent.policy, agent.value) for parameter in model.parameters()
    ]
    scale_before = agent.scaler.get_scale()
    agent.scheduler = torch.optim.lr_scheduler.ConstantLR(agent.optimizer, factor=0.5, total_iters=5)
    scheduler_step = Mock(wraps=agent.scheduler.step)
    agent.scheduler.step = scheduler_step

    # Gradient reduction itself is outside this control-flow test. Simulate an otherwise
    # identical remote worker whose post-unscale finite flag is false on every minibatch.
    agent.policy.reduce_parameters = Mock()
    agent.value.reduce_parameters = Mock()
    monkeypatch.setattr(config.torch, "_is_distributed", True)
    monkeypatch.setattr(config.torch, "_world_size", 2)
    min_reductions = 0

    def emulate_remote_overflow(tensor, op):
        nonlocal min_reductions
        if op == torch.distributed.ReduceOp.MIN:
            min_reductions += 1
            tensor[0] = 0
        elif op == torch.distributed.ReduceOp.SUM:
            # End-of-update scaler moments see an identical remote rollout.
            tensor.mul_(2)

    monkeypatch.setattr(torch.distributed, "all_reduce", emulate_remote_overflow)
    agent.enable_models_training_mode(True)
    agent.update(timestep=3, timesteps=4)

    parameters_after = [parameter for model in (agent.policy, agent.value) for parameter in model.parameters()]
    assert all(torch.equal(before, after) for before, after in zip(parameters_before, parameters_after))
    assert min_reductions == 2
    assert agent.scaler.get_scale() == pytest.approx(scale_before * (0.5**min_reductions))
    assert agent.tracking_data["Optimization / Gradient overflow count"][-1] == min_reductions
    assert agent.tracking_data["Optimization / Successful optimizer steps"][-1] == 0
    assert scheduler_step.call_count == 0


def _run_two_rank_amp_overflow_update(rank, world_size, init_method):
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=30),
    )
    previous_distributed = config.torch._is_distributed
    previous_world_size = config.torch._world_size
    previous_rank = config.torch._rank
    previous_local_rank = config.torch._local_rank
    config.torch._is_distributed = True
    config.torch._world_size = world_size
    config.torch._rank = rank
    config.torch._local_rank = rank

    try:
        agent = _make_agent(
            PPO,
            learning_epochs=1,
            learning_rate=1e-3,
            mixed_precision=True,
        )
        _collect_rollout(agent)
        parameters_before = [
            parameter.detach().clone()
            for model in (agent.policy, agent.value)
            for parameter in model.parameters()
        ]
        scale_before = agent.scaler.get_scale()
        agent.scheduler = torch.optim.lr_scheduler.ConstantLR(
            agent.optimizer, factor=0.5, total_iters=5
        )
        scheduler_epoch_before = agent.scheduler.last_epoch

        # PPO has already reduced the scaled gradients when it calls unscale_. Inject a
        # rank-local overflow at that boundary to exercise the exact case in which a
        # post-step success reduction would be too late: rank 0 is ready to step while
        # rank 1's GradScaler must skip.
        original_unscale = agent.scaler.unscale_

        def unscale_with_rank_local_overflow(optimizer):
            if rank == 1 and optimizer is agent.optimizer:
                parameter_with_gradient = next(
                    parameter
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                    if parameter.grad is not None
                )
                parameter_with_gradient.grad.fill_(float("inf"))
            return original_unscale(optimizer)

        agent.scaler.unscale_ = unscale_with_rank_local_overflow
        agent.enable_models_training_mode(True)
        agent.update(timestep=3, timesteps=4)

        parameters_after = [
            parameter.detach()
            for model in (agent.policy, agent.value)
            for parameter in model.parameters()
        ]
        assert all(
            torch.equal(before, after)
            for before, after in zip(parameters_before, parameters_after)
        )
        assert agent.scaler.get_scale() == pytest.approx(scale_before * 0.25)
        assert agent.tracking_data["Optimization / Gradient overflow count"][-1] == 2
        assert agent.tracking_data["Optimization / Successful optimizer steps"][-1] == 0
        assert agent.scheduler.last_epoch == scheduler_epoch_before
        assert not agent.optimizer.state

        # Verify the final state across the real process group, not only against each
        # worker's local snapshot.
        flattened_parameters = torch.cat([parameter.flatten() for parameter in parameters_after])
        gathered_parameters = [
            torch.empty_like(flattened_parameters) for _ in range(world_size)
        ]
        torch.distributed.all_gather(gathered_parameters, flattened_parameters)
        assert all(
            torch.equal(gathered_parameters[0], parameters)
            for parameters in gathered_parameters[1:]
        )

        local_step_state = torch.tensor(
            [
                agent.scaler.get_scale(),
                agent.tracking_data["Optimization / Gradient overflow count"][-1],
                agent.tracking_data["Optimization / Successful optimizer steps"][-1],
            ],
            dtype=torch.float64,
        )
        gathered_step_states = [
            torch.empty_like(local_step_state) for _ in range(world_size)
        ]
        torch.distributed.all_gather(gathered_step_states, local_step_state)
        assert all(
            torch.equal(gathered_step_states[0], step_state)
            for step_state in gathered_step_states[1:]
        )
    finally:
        config.torch._is_distributed = previous_distributed
        config.torch._world_size = previous_world_size
        config.torch._rank = previous_rank
        config.torch._local_rank = previous_local_rank
        torch.distributed.destroy_process_group()


def test_two_rank_amp_overflow_skips_all_optimizer_steps_and_synchronizes_scalers(
    tmp_path,
):
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("PyTorch Gloo distributed backend is not available")
    if torch.distributed.is_initialized():
        pytest.skip("The parent pytest process already owns a distributed process group")

    world_size = 2
    torch.multiprocessing.spawn(
        _run_two_rank_amp_overflow_update,
        args=(world_size, (tmp_path / "two_rank_amp_init").as_uri()),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    ("agent_class", "recurrent"),
    [(PPO, False), (PPO_RNN, True)],
    ids=["ppo", "ppo-rnn"],
)
def test_cuda_amp_overflow_skips_parameters_scaler_and_scheduler_steps(agent_class, recurrent):
    """Exercise real CUDA GradScaler overflow handling with a finite PPO loss."""

    agent = _make_agent(
        agent_class,
        learning_epochs=1,
        learning_rate=1e-3,
        recurrent=recurrent,
        device="cuda:0",
        mixed_precision=True,
    )
    _collect_rollout(agent)

    assert agent.scaler.is_enabled()
    parameters_before = [
        parameter.detach().clone() for model in (agent.policy, agent.value) for parameter in model.parameters()
    ]
    scale_before = agent.scaler.get_scale()
    agent.scheduler = torch.optim.lr_scheduler.ConstantLR(agent.optimizer, factor=0.5, total_iters=5)
    scheduler_epoch_before = agent.scheduler.last_epoch
    scheduler_step = Mock(wraps=agent.scheduler.step)
    agent.scheduler.step = scheduler_step

    # The hook runs during backward, after the finite-loss guard and before GradScaler.unscale_.
    # One non-finite gradient must make GradScaler skip the complete optimizer transaction.
    hooked_parameter = next(parameter for parameter in agent.policy.parameters() if parameter.requires_grad)
    gradient_hook = hooked_parameter.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
    try:
        agent.enable_models_training_mode(True)
        agent.update(timestep=3, timesteps=4)
    finally:
        gradient_hook.remove()

    expected_overflows = 2  # one for each complete minibatch in the single learning epoch
    parameters_after = [parameter for model in (agent.policy, agent.value) for parameter in model.parameters()]
    assert all(torch.equal(before, after) for before, after in zip(parameters_before, parameters_after))
    assert agent.scaler.get_scale() == pytest.approx(scale_before * (0.5**expected_overflows))
    assert agent.tracking_data["Optimization / Gradient overflow count"][-1] == expected_overflows
    assert agent.tracking_data["Optimization / Successful optimizer steps"][-1] == 0
    assert agent.tracking_data["Optimization / Effective minibatches"][-1] == expected_overflows
    assert agent.tracking_data["Optimization / Grad norm"][-1] == 0.0
    assert scheduler_step.call_count == 0
    assert agent.scheduler.last_epoch == scheduler_epoch_before
    assert not agent.optimizer.state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_amp_ppo_executes_real_mixed_precision_optimizer_steps():
    """Smoke-test the successful CUDA autocast/GradScaler path for feed-forward PPO."""

    agent = _make_agent(
        PPO,
        learning_epochs=1,
        learning_rate=1e-3,
        device="cuda:0",
        mixed_precision=True,
    )
    # Avoid making this success-path smoke depend on GradScaler's initial overflow calibration.
    agent.scaler = torch.amp.GradScaler(device="cuda", init_scale=128.0, growth_interval=1000, enabled=True)
    _collect_rollout(agent)

    parameters_before = [
        parameter.detach().clone() for model in (agent.policy, agent.value) for parameter in model.parameters()
    ]
    scale_before = agent.scaler.get_scale()
    agent.scheduler = torch.optim.lr_scheduler.ConstantLR(agent.optimizer, factor=0.5, total_iters=5)
    scheduler_step = Mock(wraps=agent.scheduler.step)
    agent.scheduler.step = scheduler_step
    agent.enable_models_training_mode(True)
    agent.update(timestep=3, timesteps=4)

    parameters_after = [parameter for model in (agent.policy, agent.value) for parameter in model.parameters()]
    assert any(not torch.equal(before, after) for before, after in zip(parameters_before, parameters_after))
    assert all(torch.isfinite(parameter).all() for parameter in parameters_after)
    assert agent.scaler.is_enabled()
    assert agent.scaler.get_scale() == scale_before
    assert agent.tracking_data["Optimization / Gradient overflow count"][-1] == 0
    assert agent.tracking_data["Optimization / Successful optimizer steps"][-1] == 2
    assert agent.tracking_data["Optimization / Grad norm"][-1] > 0
    assert scheduler_step.call_count == 1
