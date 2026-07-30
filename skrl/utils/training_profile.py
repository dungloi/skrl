from __future__ import annotations

from contextlib import nullcontext
from typing import Any, ContextManager


_controller: Any | None = None


def set_training_profile_controller(controller: Any | None) -> None:
    """Install the process-local training profiling controller.

    The controller is intentionally process-local. Distributed training launches one
    Python process per rank, and every rank therefore owns an independent profiler
    and output file.
    """

    global _controller
    _controller = controller


def get_training_profile_controller() -> Any | None:
    """Return the active training profiling controller, if any."""

    return _controller


def training_profile_event(name: str, /, **payload: Any) -> None:
    """Forward a semantic training event to the active controller."""

    controller = _controller
    if controller is not None:
        controller.event(name, **payload)


def training_profile_range(name: str, /, **metadata: Any) -> ContextManager[Any]:
    """Return a profiler range or a no-op context manager when profiling is disabled."""

    controller = _controller
    if controller is None:
        return nullcontext()
    return controller.range(name, **metadata)
