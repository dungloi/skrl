from __future__ import annotations

import math
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
    for field in ("time_limit_bootstrap", "mixed_precision", "value_mixed_precision"):
        value_ = getattr(cfg, field)
        if not isinstance(value_, bool):
            raise ValueError(f"{name} {field} must be a boolean, got {value_!r}")
    if cfg.value_mixed_precision and not cfg.mixed_precision:
        raise ValueError(f"{name} value_mixed_precision requires mixed_precision")

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

    value_loss_guard = _finite_float(f"{name} value_loss_guard", cfg.value_loss_guard)
    value_prediction_guard = _finite_float(
        f"{name} value_prediction_guard", cfg.value_prediction_guard
    )
    value_lr_backoff_factor = _finite_float(
        f"{name} value_lr_backoff_factor", cfg.value_lr_backoff_factor
    )
    value_lr_backoff_min = _finite_float(
        f"{name} value_lr_backoff_min", cfg.value_lr_backoff_min
    )
    if value_loss_guard < 0 or value_prediction_guard < 0 or value_lr_backoff_min < 0:
        raise ValueError(f"{name} value guards and backoff minimum must be non-negative")
    if not 0 < value_lr_backoff_factor <= 1:
        raise ValueError(f"{name} value_lr_backoff_factor must be in (0, 1]")
    if (
        policy is value
        and (value_loss_guard > 0 or value_prediction_guard > 0 or value_lr_backoff_factor < 1)
    ):
        raise ValueError(f"{name} critic circuit breaker requires separate policy and value models")

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


def compute_value_loss_fp32(
    *,
    predicted_values: torch.Tensor,
    sampled_values: torch.Tensor,
    sampled_returns: torch.Tensor,
    value_clip: float,
    value_loss_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute PPO's value objective in FP32, even when the model uses AMP.

    Gradient scaling only protects the backward pass. Squaring a large FP16
    residual can overflow during the forward loss computation, so the value
    objective is deliberately promoted before subtraction and squaring.

    :return: ``(scaled_loss, unscaled_loss, clip_fraction)``.
    """

    predicted_values_fp32 = predicted_values.float()
    sampled_values_fp32 = sampled_values.float()
    sampled_returns_fp32 = sampled_returns.float()
    if value_clip > 0:
        value_delta = predicted_values_fp32 - sampled_values_fp32
        with torch.no_grad():
            value_clip_fraction = (torch.abs(value_delta) > value_clip).float().mean()
        predicted_values_clipped = sampled_values_fp32 + torch.clip(
            value_delta, min=-value_clip, max=value_clip
        )
        value_error = (sampled_returns_fp32 - predicted_values_fp32).square()
        value_error_clipped = (sampled_returns_fp32 - predicted_values_clipped).square()
        unscaled_loss = torch.maximum(value_error, value_error_clipped).mean()
    else:
        value_clip_fraction = torch.zeros((), device=sampled_values.device)
        unscaled_loss = (sampled_returns_fp32 - predicted_values_fp32).square().mean()
    return value_loss_scale * unscaled_loss, unscaled_loss, value_clip_fraction


def critic_guard_triggered(
    *,
    predicted_values: torch.Tensor,
    unscaled_value_loss: torch.Tensor,
    loss_threshold: float,
    prediction_threshold: float,
) -> bool:
    """Return a distributed-consistent critic circuit-breaker decision."""

    triggered = torch.zeros((), dtype=torch.bool, device=predicted_values.device)
    if loss_threshold > 0:
        triggered |= unscaled_value_loss.detach() >= loss_threshold
    if prediction_threshold > 0:
        triggered |= predicted_values.detach().float().abs().max() >= prediction_threshold
    return any_rank_true(triggered)


def backoff_value_learning_rate(
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    factor: float,
    minimum: float,
) -> tuple[float, float]:
    """Back off the value parameter group's learning rate once."""

    group_index = 1 if len(optimizer.param_groups) > 1 else 0
    group = optimizer.param_groups[group_index]
    previous = float(group["lr"])
    current = min(previous, max(previous * factor, minimum))
    group["lr"] = current
    if scheduler is not None and hasattr(scheduler, "_last_lr"):
        scheduler._last_lr = [float(item["lr"]) for item in optimizer.param_groups]
    return previous, current


def validate_scalar_output(
    name: str, tensor: torch.Tensor, batch_size: int, *, synchronize: bool = False
) -> None:
    expected_shape = (batch_size, 1)
    local_shape_failed = tensor.shape != expected_shape
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
            if not torch.isfinite(actual).all().item():
                finite_failed = True

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
