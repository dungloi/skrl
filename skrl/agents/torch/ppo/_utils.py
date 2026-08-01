from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch

from skrl import config


def _finite_float(name: str, value: Any) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from error
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return converted


def validate_ppo_setup(*, name: str, cfg: Any, policy: Any, value: Any, memory: Any) -> None:
    """Validate PPO contracts that would otherwise fail later with silent NaNs or stale samples."""

    if policy is None or value is None:
        raise ValueError(f"{name} requires both a policy and a value model")

    integer_fields = {
        "rollouts": cfg.rollouts,
        "learning_epochs": cfg.learning_epochs,
        "mini_batches": cfg.mini_batches,
    }
    for field, value_ in integer_fields.items():
        if not isinstance(value_, int) or isinstance(value_, bool) or value_ < 1:
            raise ValueError(f"{name} {field} must be a positive integer, got {value_!r}")

    for field in ("random_timesteps", "learning_starts"):
        value_ = getattr(cfg, field)
        if not isinstance(value_, int) or isinstance(value_, bool) or value_ < 0:
            raise ValueError(f"{name} {field} must be a non-negative integer, got {value_!r}")
    for field in ("time_limit_bootstrap", "mixed_precision"):
        value_ = getattr(cfg, field)
        if not isinstance(value_, bool):
            raise ValueError(f"{name} {field} must be a boolean, got {value_!r}")
    if cfg.numerics_check_mode not in ("strict", "update", "off"):
        raise ValueError(
            f"{name} numerics_check_mode must be 'strict', 'update' or 'off', "
            f"got {cfg.numerics_check_mode!r}"
        )

    discount_factor = _finite_float(f"{name} discount_factor", cfg.discount_factor)
    lambda_ = _finite_float(f"{name} lambda_", cfg.lambda_)
    if not 0 <= discount_factor <= 1:
        raise ValueError(f"{name} discount_factor must be in [0, 1]")
    if not 0 <= lambda_ <= 1:
        raise ValueError(f"{name} lambda_ must be in [0, 1]")

    ratio_clip = _finite_float(f"{name} ratio_clip", cfg.ratio_clip)
    _finite_float(f"{name} value_clip", cfg.value_clip)
    kl_threshold = _finite_float(f"{name} kl_threshold", cfg.kl_threshold)
    if ratio_clip < 0 or kl_threshold < 0:
        raise ValueError(f"{name} ratio_clip and kl_threshold must be non-negative")

    entropy_loss_scale = _finite_float(f"{name} entropy_loss_scale", cfg.entropy_loss_scale)
    value_loss_scale = _finite_float(f"{name} value_loss_scale", cfg.value_loss_scale)
    _finite_float(f"{name} grad_norm_clip", cfg.grad_norm_clip)
    if entropy_loss_scale < 0 or value_loss_scale < 0:
        raise ValueError(f"{name} loss scales must be non-negative")

    if len(cfg.learning_rate) != 2:
        raise ValueError(f"{name} learning_rate must contain exactly policy and value rates")
    if any(_finite_float(f"{name} learning_rate", lr) < 0 for lr in cfg.learning_rate):
        raise ValueError(f"{name} learning rates must be finite and non-negative")
    if policy is value and float(cfg.learning_rate[0]) != float(cfg.learning_rate[1]):
        raise ValueError(f"{name} cannot apply different policy/value learning rates to a shared model")
    if len(cfg.learning_rate_scheduler) != 2 or len(cfg.learning_rate_scheduler_kwargs) != 2:
        raise ValueError(f"{name} scheduler configuration must contain policy and value entries")
    if cfg.learning_rate_scheduler[0] is not cfg.learning_rate_scheduler[1]:
        raise ValueError(f"{name} currently requires the same scheduler for policy and value parameter groups")
    if cfg.learning_rate_scheduler_kwargs[0] != cfg.learning_rate_scheduler_kwargs[1]:
        raise ValueError(f"{name} currently requires identical policy/value scheduler arguments")

    if memory is not None and memory.memory_size != cfg.rollouts:
        raise ValueError(
            f"{name} rollout length ({cfg.rollouts}) must match memory size ({memory.memory_size})"
        )


def ensure_full_rollout(*, name: str, memory: Any, consumed: bool = False) -> None:
    if memory is None:
        raise RuntimeError(f"{name} cannot update without rollout memory")
    expected = memory.memory_size * memory.num_envs
    at_rollout_boundary = memory.filled and memory.memory_index == 0 and memory.env_index == 0
    local_fresh_failure = len(memory) != expected or not at_rollout_boundary
    failure_code = torch.tensor(
        2 * int(local_fresh_failure) + int(consumed), dtype=torch.int32, device=memory.device
    )
    if config.torch.is_distributed:
        torch.distributed.all_reduce(failure_code, op=torch.distributed.ReduceOp.MAX)
    global_failure_code = int(failure_code.item())
    if global_failure_code >= 2:
        raise RuntimeError(
            f"{name} requires a fresh full rollout before update ({len(memory)} / {expected} samples)"
        )
    if global_failure_code:
        raise RuntimeError(f"{name} rollout has already been consumed by an update")


def _any_rank_failed(local_failure: torch.Tensor, *, synchronize: bool) -> tuple[bool, bool]:
    """Return (global failure, local failure), optionally synchronizing all workers."""

    local_failed = bool(local_failure.item())
    if synchronize and config.torch.is_distributed:
        torch.distributed.all_reduce(local_failure, op=torch.distributed.ReduceOp.MAX)
        return bool(local_failure.item()), local_failed
    return local_failed, local_failed


def _raise_if_any_rank_failed(name: str, local_failure: torch.Tensor, *, synchronize: bool) -> None:
    """Raise on every worker when a numerical contract fails on any synchronized worker."""

    failed, local_failed = _any_rank_failed(local_failure, synchronize=synchronize)
    if failed:
        location = "" if local_failed else " on another distributed rank"
        raise FloatingPointError(f"{name} contains NaN or Inf{location}")


def require_finite(name: str, tensor: torch.Tensor, *, synchronize: bool = False) -> None:
    _raise_if_any_rank_failed(
        name,
        torch.logical_not(torch.isfinite(tensor).all()).to(dtype=torch.int32),
        synchronize=synchronize,
    )


def require_finite_many(
    name: str, tensors: Iterable[torch.Tensor], *, synchronize: bool = False
) -> None:
    """Validate a tensor collection with one device-to-host decision.

    This is intended for update-boundary diagnostics, where reporting the exact
    first tensor is less useful than avoiding a separate CUDA synchronization for
    every model parameter and rollout buffer.
    """

    tensors = tuple(tensor for tensor in tensors if isinstance(tensor, torch.Tensor))
    if not tensors:
        return
    device = tensors[0].device
    local_failure = torch.zeros((), dtype=torch.int32, device=device)
    for tensor in tensors:
        tensor_failed = torch.logical_not(torch.isfinite(tensor).all())
        local_failure = torch.maximum(
            local_failure,
            tensor_failed.to(device=device, dtype=torch.int32),
        )
    _raise_if_any_rank_failed(name, local_failure, synchronize=synchronize)


def require_finite_model(name: str, model: Any, *, synchronize: bool = False) -> None:
    checked_tensors: list[tuple[str, torch.Tensor]] = []
    for parameter_name, parameter in model.named_parameters():
        checked_tensors.append((f"parameter {parameter_name}", parameter))
    for buffer_name, buffer in model.named_buffers():
        if torch.is_floating_point(buffer):
            checked_tensors.append((f"buffer {buffer_name}", buffer))
    if not checked_tensors:
        return

    device = checked_tensors[0][1].device
    local_failure = torch.zeros((), dtype=torch.int32, device=device)
    for _, tensor in checked_tensors:
        tensor_failed = torch.logical_not(torch.isfinite(tensor).all())
        local_failure = torch.maximum(local_failure, tensor_failed.to(device=device, dtype=torch.int32))
    first_failed_name = None
    if local_failure.item():
        for tensor_name, tensor in checked_tensors:
            if not torch.isfinite(tensor).all().item():
                first_failed_name = tensor_name
                break
    failure_name = f"{name} {first_failed_name}" if first_failed_name is not None else name
    _raise_if_any_rank_failed(failure_name, local_failure, synchronize=synchronize)


def any_rank_true(value: torch.Tensor) -> bool:
    """Return whether a boolean condition is true on any distributed worker."""

    local_failure = value.bool().any().to(dtype=torch.int32)
    failed, _ = _any_rank_failed(local_failure, synchronize=True)
    return failed


def synchronized_grad_scaler_step(
    *, scaler: Any, optimizer: torch.optim.Optimizer, grad_norm: torch.Tensor
) -> bool | torch.Tensor:
    """Execute an optimizer step only when every distributed worker has finite gradients.

    The caller must unscale the optimizer and compute ``grad_norm`` before calling this
    helper. A post-step synchronization is too late: one worker may already have changed
    its parameters while another worker's GradScaler skipped the step. Instead, synchronize
    the finite-gradient decision first and skip the real optimizer on every worker if any
    worker reports an overflow.

    On a worker whose local gradients are finite, GradScaler would otherwise treat a remote
    overflow as a successful iteration and grow (or retain) its scale. A throwaway optimizer
    carrying an infinite sentinel gradient records the global overflow through GradScaler's
    public API, so all workers apply the same backoff and reset their growth tracker.

    On a single device, PyTorch's native GradScaler already records ``found_inf``
    on-device. Return a scalar tensor in that common path so callers can batch
    optimizer-success accounting with their epoch statistics instead of forcing
    another host decision per mini-batch. Distributed execution still returns a
    Python boolean because every rank must take the same branch before mutation.

    :return: Whether the real optimizer step was executed successfully, as a
        device boolean on the native single-device AMP path and a Python boolean
        otherwise.
    """

    scaler_enabled = scaler.is_enabled()
    if not scaler_enabled:
        # Non-AMP callers already use the synchronized finite-gradient guard before
        # entering this helper, so no additional collective is needed here.
        scaler.step(optimizer)
        scaler.update()
        return True

    native_grad_scaler_types = tuple(
        scaler_type
        for scaler_type in (
            getattr(torch.amp, "GradScaler", None),
            getattr(torch.cuda.amp, "GradScaler", None),
        )
        if isinstance(scaler_type, type)
    )
    if not config.torch.is_distributed and isinstance(scaler, native_grad_scaler_types):
        # ``unscale_`` has already populated GradScaler's per-optimizer
        # ``found_inf`` tensors. Native GradScaler consumes those tensors in
        # ``step``/``update``; fused optimizers keep the complete skip/backoff
        # decision on the GPU. The total gradient norm is finite exactly when
        # the unscaled gradient collection is finite and doubles as an async
        # success metric for the caller.
        optimizer_step_allowed = torch.isfinite(grad_norm).all().detach()
        optimizer_state = getattr(scaler, "_per_optimizer_states", {}).get(id(optimizer))
        found_inf_per_device = (
            optimizer_state.get("found_inf_per_device", {})
            if isinstance(optimizer_state, dict)
            else {}
        )
        if not found_inf_per_device:
            raise RuntimeError(
                "Native GradScaler did not expose found_inf state after unscale_; "
                "call scaler.unscale_(optimizer) before synchronized_grad_scaler_step"
            )
        # GradScaler checks scaled gradients while unscaling. With a scale below
        # one, a finite scaled gradient can itself overflow during division. Fold
        # the post-unscale norm decision back into every found_inf tensor so both
        # fused and regular optimizers skip that step without a host branch.
        for found_inf in found_inf_per_device.values():
            optimizer_step_allowed = torch.logical_and(
                optimizer_step_allowed,
                torch.logical_not(found_inf.bool().any()).to(device=optimizer_step_allowed.device),
            )
        overflow = torch.logical_not(optimizer_step_allowed)
        for found_inf in found_inf_per_device.values():
            found_inf.copy_(
                torch.maximum(
                    found_inf,
                    overflow.to(device=found_inf.device, dtype=found_inf.dtype),
                )
            )
        scaler.step(optimizer)
        scaler.update()
        return optimizer_step_allowed

    scale_before_step = float(scaler.get_scale())
    local_scale_is_valid = math.isfinite(scale_before_step) and scale_before_step > 0
    optimizer_step_allowed = bool(torch.isfinite(grad_norm).all().item())
    if config.torch.is_distributed:
        safe_scale = scale_before_step if local_scale_is_valid else 0.0
        # A single MIN collective obtains the global finite-gradient decision, scale
        # validity, minimum scale and (through the negated value) maximum scale. Different
        # scales mean backward/unscale used incompatible coordinates, so every worker must
        # fail before any real optimizer can mutate parameters.
        step_control = torch.tensor(
            (float(optimizer_step_allowed), float(local_scale_is_valid), safe_scale, -safe_scale),
            dtype=torch.float64,
            device=grad_norm.device,
        )
        torch.distributed.all_reduce(step_control, op=torch.distributed.ReduceOp.MIN)
        if not bool(step_control[1].item()):
            raise RuntimeError("GradScaler scale must be finite and positive on every distributed rank")
        minimum_scale = step_control[2].item()
        maximum_scale = -step_control[3].item()
        if minimum_scale != maximum_scale:
            raise RuntimeError(
                "GradScaler scale differs across distributed ranks "
                f"(minimum {minimum_scale}, maximum {maximum_scale})"
            )
        optimizer_step_allowed = bool(step_control[0].item())
    elif not local_scale_is_valid:
        raise RuntimeError("GradScaler scale must be finite and positive")

    globally_finite = optimizer_step_allowed
    if globally_finite:
        scaler.step(optimizer)
    else:
        # ``unscale_(optimizer)`` has already run, so changing a real gradient now would not
        # update GradScaler's recorded found_inf state. Register the global overflow on a
        # separate zero-LR optimizer instead. ``scaler.step`` is intentionally invoked for
        # the sentinel optimizer and is skipped because its gradient is infinite.
        overflow_marker = torch.nn.Parameter(torch.zeros((), device=grad_norm.device))
        overflow_marker.grad = torch.full_like(overflow_marker, float("inf"))
        overflow_optimizer = torch.optim.SGD((overflow_marker,), lr=0.0)
        scaler.unscale_(overflow_optimizer)
        scaler.step(overflow_optimizer)

    scaler.update()
    if not globally_finite:
        return False
    # GradScaler normally skips exactly when its recorded found_inf is non-zero. Keep this
    # check for custom/fake scalers and as a defensive guard around future PyTorch behavior.
    return scaler.get_scale() >= scale_before_step


def validate_scalar_output(
    name: str,
    tensor: torch.Tensor,
    batch_size: int,
    *,
    synchronize: bool = False,
    check_finite: bool = True,
) -> None:
    expected_shape = (batch_size, 1)
    local_shape_failed = tensor.shape != expected_shape
    if not check_finite and not (synchronize and config.torch.is_distributed):
        if local_shape_failed:
            raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}")
        return

    if not check_finite:
        failure_code = torch.tensor(
            2 * int(local_shape_failed), dtype=torch.int32, device=tensor.device
        )
        torch.distributed.all_reduce(failure_code, op=torch.distributed.ReduceOp.MAX)
        global_failure_code = int(failure_code.item())
        if global_failure_code:
            actual = tuple(tensor.shape) if local_shape_failed else "invalid on another distributed rank"
            raise ValueError(f"{name} must have shape {expected_shape}, got {actual}")
        return

    local_finite_failed = not torch.isfinite(tensor).all().item()
    # One collective covers both contracts. Shape errors have priority because a
    # malformed scalar output may also make a finite check misleading.
    failure_code = torch.tensor(
        2 * int(local_shape_failed) + int(local_finite_failed),
        dtype=torch.int32,
        device=tensor.device,
    )
    if synchronize and config.torch.is_distributed:
        torch.distributed.all_reduce(failure_code, op=torch.distributed.ReduceOp.MAX)
    global_failure_code = int(failure_code.item())
    if global_failure_code >= 2:
        local_failed = local_shape_failed
        actual = tuple(tensor.shape) if local_failed else "invalid on another distributed rank"
        raise ValueError(f"{name} must have shape {expected_shape}, got {actual}")
    if global_failure_code:
        location = "" if local_finite_failed else " on another distributed rank"
        raise FloatingPointError(f"{name} contains NaN or Inf{location}")


def validate_rnn_output(
    name: str,
    outputs: dict[str, Any],
    expected_states: list[torch.Tensor],
    *,
    synchronize: bool = False,
    check_finite: bool = True,
) -> list[torch.Tensor]:
    """Validate the number, shape and finiteness of live recurrent states."""

    if not expected_states:
        return []
    actual_states = outputs.get("rnn", [])
    structure_failed = not isinstance(actual_states, (list, tuple)) or len(actual_states) != len(expected_states)
    shape_failed = structure_failed
    finite_failed = False
    if not structure_failed:
        for actual, expected in zip(actual_states, expected_states):
            if not isinstance(actual, torch.Tensor) or actual.shape != expected.shape:
                shape_failed = True
                continue
            if check_finite and not torch.isfinite(actual).all().item():
                finite_failed = True

    if not check_finite and not (synchronize and config.torch.is_distributed):
        if shape_failed:
            raise ValueError(f"{name} has an invalid recurrent-state count or shape")
        return list(actual_states)

    failure_code = torch.tensor(
        2 * int(shape_failed) + int(finite_failed),
        dtype=torch.int32,
        device=expected_states[0].device,
    )
    if synchronize and config.torch.is_distributed:
        torch.distributed.all_reduce(failure_code, op=torch.distributed.ReduceOp.MAX)
    global_failure_code = int(failure_code.item())
    if global_failure_code >= 2:
        location = "" if shape_failed else " on another distributed rank"
        raise ValueError(f"{name} has an invalid recurrent-state count or shape{location}")
    if global_failure_code:
        location = "" if finite_failed else " on another distributed rank"
        raise FloatingPointError(f"{name} recurrent state contains NaN or Inf{location}")
    return list(actual_states)
