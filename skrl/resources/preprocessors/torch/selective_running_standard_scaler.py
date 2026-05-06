from __future__ import annotations

from collections.abc import Sequence

import gymnasium

import torch
import torch.nn as nn

from skrl import config
from skrl.utils.spaces.torch import compute_space_size


class SelectiveRunningStandardScaler(nn.Module):
    def __init__(
        self,
        size: int | list[int] | gymnasium.Space,
        *,
        exclude_keys: Sequence[str] = ("image",),
        epsilon: float = 1e-8,
        clip_threshold: float = 5.0,
        device: str | torch.device | None = None,
    ) -> None:
        """Standardize selected flat observation columns while passing excluded keys through unchanged.

        skrl wrappers flatten Dict observations before preprocessors are called. This preprocessor
        reconstructs only the top-level flat slices from the Dict space metadata and keeps the
        runtime path as Tensor slice operations.
        """

        super().__init__()

        self.epsilon = epsilon
        self.clip_threshold = clip_threshold
        self.device = config.torch.parse_device(device)

        self._flat_size = compute_space_size(size, occupied_size=True)
        self._selected_slices = self._build_selected_slices(size, set(exclude_keys))
        self._selected_size = sum(end - start for start, end in self._selected_slices)
        self._all_selected = self._selected_size == self._flat_size

        self.register_buffer(
            "running_mean",
            torch.zeros(self._selected_size, dtype=torch.float64, device=self.device),
        )
        self.register_buffer(
            "running_variance",
            torch.ones(self._selected_size, dtype=torch.float64, device=self.device),
        )
        self.register_buffer("current_count", torch.ones((), dtype=torch.float64, device=self.device))

    def _build_selected_slices(
        self,
        size: int | list[int] | gymnasium.Space,
        exclude_keys: set[str],
    ) -> list[tuple[int, int]]:
        if not isinstance(size, gymnasium.spaces.Dict):
            return [(0, self._flat_size)]

        selected: list[tuple[int, int]] = []
        start = 0
        for key in sorted(size.keys()):
            end = start + compute_space_size(size[key], occupied_size=True)
            if key not in exclude_keys:
                selected.append((start, end))
            start = end
        return selected

    def _parallel_variance(self, input_mean: torch.Tensor, input_var: torch.Tensor, input_count: int) -> None:
        delta = input_mean - self.running_mean
        total_count = self.current_count + input_count
        m2 = (
            (self.running_variance * self.current_count)
            + (input_var * input_count)
            + delta**2 * self.current_count * input_count / total_count
        )

        self.running_mean = self.running_mean + delta * input_count / total_count
        self.running_variance = m2 / total_count
        self.current_count = total_count

    def _gather_selected(self, x: torch.Tensor) -> torch.Tensor:
        if self._all_selected:
            return x
        return torch.cat([x[..., start:end] for start, end in self._selected_slices], dim=-1)

    def _scatter_selected(self, x: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        if self._all_selected:
            return selected

        output = x.clone()
        offset = 0
        for start, end in self._selected_slices:
            width = end - start
            output[..., start:end] = selected[..., offset : offset + width]
            offset += width
        return output

    def _compute(self, x: torch.Tensor, *, train: bool = False, inverse: bool = False) -> torch.Tensor:
        if self._selected_size == 0:
            return x

        selected = self._gather_selected(x)

        if train:
            reduce_dims = tuple(range(selected.dim() - 1))
            input_count = 1
            for dim in selected.shape[:-1]:
                input_count *= int(dim)
            self._parallel_variance(
                torch.mean(selected, dim=reduce_dims),
                torch.var(selected, dim=reduce_dims),
                input_count,
            )

        if inverse:
            selected = (
                torch.sqrt(self.running_variance.float())
                * torch.clamp(selected, min=-self.clip_threshold, max=self.clip_threshold)
                + self.running_mean.float()
            )
        else:
            selected = torch.clamp(
                (selected - self.running_mean.float()) / (torch.sqrt(self.running_variance.float()) + self.epsilon),
                min=-self.clip_threshold,
                max=self.clip_threshold,
            )
        return self._scatter_selected(x, selected)

    def forward(
        self,
        x: torch.Tensor | None,
        *,
        train: bool = False,
        inverse: bool = False,
        no_grad: bool = True,
    ) -> torch.Tensor | None:
        if x is None:
            return None
        if no_grad:
            with torch.no_grad():
                return self._compute(x, train=train, inverse=inverse)
        return self._compute(x, train=train, inverse=inverse)
