# Copyright (c) 2026, Tang yucheng
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

"""纯 MLP 模型族定义。

该模块包含一组共享同一配置结构的模型实现：

- ``MLPPolicy``：Gaussian policy，输出动作均值与可学习 log std
- ``MLPValue``：deterministic value 网络，输出标量 value

网络结构通过 ``network.mlp`` 配置驱动，适用于扁平向量输入场景。
"""

from __future__ import annotations

from typing import Literal

import gymnasium
import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space

from .utils import (
    _build_mlp,
    _normalize_mlp_cfg,
    _normalize_network_root,
    _print_block_footer,
    _print_block_header,
    _print_deterministic_settings,
    _print_gaussian_settings,
    _print_inputs,
    _print_kv,
    _print_section,
    _print_subconfig,
)

# 当前模块要求的 network 一级配置 key
_MLP_REQUIRED_KEYS = {"mlp"}
# 当前模块输入空间打印优先级（MLP 输入通常为扁平 Box，无需额外排序）
_MLP_SPACE_PRIORITY: dict[str, tuple[str, ...]] = {}


class MLPPolicy(GaussianMixin, Model):
    """基于纯 MLP backbone 的 Gaussian policy。

    期望的网络配置:
        network.mlp
    """

    def __init__(
        self,
        *,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        clip_actions: bool = False,
        clip_mean_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20,
        max_log_std: float = 2,
        reduction: Literal["mean", "sum", "prod", "none"] = "sum",
        role: str = "",
        initial_log_std: float = 0,
        fixed_log_std: bool = False,
        **kwargs
    ):
        """初始化 MLP Gaussian policy。

        Args:
            observation_space: 扁平 observation space。
            state_space: State space（为接口一致性保留）。
            action_space: 连续动作空间。
            device: 目标 torch device。
            clip_actions: 是否裁剪采样动作。
            clip_mean_actions: 是否裁剪 Gaussian mean 输出。
            clip_log_std: 是否裁剪 log standard deviation。
            min_log_std: log standard deviation 下界。
            max_log_std: log standard deviation 上界。
            reduction: GaussianMixin 使用的 reduction 模式。
            role: skrl 使用的模型角色名。
            initial_log_std: log std 参数初始值。
            fixed_log_std: 是否固定 log std（不可训练）。
            **kwargs: 额外参数，必须包含 `network`，且其中包含
                `mlp` 配置段。
        """
        # ----------------------------------------
        # 基类初始化
        # ----------------------------------------
        Model.__init__(
            self,
            observation_space=observation_space, state_space=state_space,
            action_space=action_space, device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions, clip_mean_actions=clip_mean_actions, clip_log_std=clip_log_std,
            min_log_std=min_log_std, max_log_std=max_log_std, reduction=reduction, role=role,
        )

        # ----------------------------------------
        # 网络结构定义
        # ----------------------------------------
        # 1. 解析输入空间维度
        action_shape = action_space.shape
        observation_dim = observation_space.shape[0]
        action_dim  = action_shape[0]

        # 2. 从 YAML 读取并严格校验网络结构参数
        network_cfg = _normalize_network_root(
            kwargs.get("network", None), required_keys=_MLP_REQUIRED_KEYS, model_name=self.__class__.__name__
        )
        mlp_cfg = _normalize_mlp_cfg(network_cfg.get("mlp"))

        # 3. 按配置构建 MLP (Observation特征 -> Action)
        self.mlp = _build_mlp(
            input_dim=observation_dim,
            hidden_dims=mlp_cfg["hidden_dims"],
            output_dim=action_dim,
            use_layernorm=mlp_cfg["use_layernorm"],
            activation_factory=mlp_cfg["activation_factory"],
        )

        # 4. 定义 log_std 参数
        self.log_std_parameter = nn.Parameter(
            torch.full(size=action_space.shape, fill_value=float(initial_log_std), dtype=torch.float32),
            requires_grad=not fixed_log_std
        )

        # ----------------------------------------
        # 打印信息
        # ----------------------------------------
        _print_block_header(f"[{self.__class__.__name__}] Initialization")
        _print_inputs(
            observation_space,
            state_space,
            action_space,
            device,
            space_priority=_MLP_SPACE_PRIORITY,
        )
        _print_gaussian_settings(
            clip_actions=clip_actions,
            clip_mean_actions=clip_mean_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
            initial_log_std=initial_log_std,
            fixed_log_std=fixed_log_std,
        )

        _print_section("Parsed Shapes")
        _print_kv("observation_shape", observation_space.shape)
        _print_kv("action_shape", action_shape)

        _print_section("Network Config")
        _print_subconfig(
            "mlp",
            [
                ("hidden_dims", mlp_cfg["hidden_dims"]),
                ("use_layernorm", mlp_cfg["use_layernorm"]),
                ("activation", mlp_cfg["activation"]),
            ],
        )

        _print_section("Derived Dims")
        _print_kv("observation_dim", observation_dim)
        _print_kv("action_dim", action_dim)
        _print_block_footer(f"[{self.__class__.__name__}] Ready")

    def compute(self, inputs, role=""):
        """执行前向计算并返回 Gaussian mean 与 log std。

        Args:
            inputs: skrl 模型输入字典，使用 `inputs["observations"]`。
            role: 模型角色名（当前前向计算未使用）。

        Returns:
            `(mean_actions, extras)` 二元组，其中 `extras` 包含
            `{"log_std": self.log_std_parameter}`.
        """
        # 1. 准备输入
        observations = unflatten_tensorized_space(self.observation_space, inputs.get("observations"))

        # 2. 通过 MLP 计算动作均值
        output = self.mlp(observations)

        return output, {"log_std": self.log_std_parameter}


class MLPValue(DeterministicMixin, Model):
    """基于纯 MLP backbone 的 deterministic value 模型。

    期望的网络配置:
        network.mlp
    """

    def __init__(
        self,
        *,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        clip_actions: bool = False,
        role: str = "",
        **kwargs
    ):
        """初始化 MLP value 模型。

        Args:
            observation_space: Observation space（为接口一致性保留）。
            state_space: 扁平 state space。
            action_space: skrl 接口使用的动作空间。
            device: 目标 torch device。
            clip_actions: deterministic mixin 中是否裁剪动作。
            role: skrl 使用的模型角色名。
            **kwargs: 额外参数，必须包含 `network`，且其中包含
                `mlp` 配置段。
        """
        # ----------------------------------------
        # 基类初始化
        # ----------------------------------------
        Model.__init__(
            self,
            observation_space=observation_space, state_space=state_space,
            action_space=action_space, device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=clip_actions, role=role)

        # ----------------------------------------
        # 网络结构定义
        # ----------------------------------------

        # 1. 解析输入空间维度
        action_shape = action_space.shape
        state_dim = state_space.shape[0]

        # 2. 从 YAML 读取并严格校验网络结构参数
        network_cfg = _normalize_network_root(
            kwargs.get("network", None), required_keys=_MLP_REQUIRED_KEYS, model_name=self.__class__.__name__
        )
        mlp_cfg = _normalize_mlp_cfg(network_cfg.get("mlp"))

        # 3. 按配置构建 MLP (State特征 -> Value)
        self.mlp = _build_mlp(
            input_dim=state_dim,
            hidden_dims=mlp_cfg["hidden_dims"],
            output_dim=1,
            use_layernorm=mlp_cfg["use_layernorm"],
            activation_factory=mlp_cfg["activation_factory"],
        )

        # ----------------------------------------
        # 打印信息
        # ----------------------------------------
        _print_block_header(f"[{self.__class__.__name__}] Initialization")
        _print_inputs(
            observation_space,
            state_space,
            action_space,
            device,
            space_priority=_MLP_SPACE_PRIORITY,
        )
        _print_deterministic_settings(clip_actions=clip_actions, role=role)

        _print_section("Parsed Shapes")
        _print_kv("state_shape", state_space.shape)
        _print_kv("action_shape", action_shape)

        _print_section("Network Config")
        _print_subconfig(
            "mlp",
            [
                ("hidden_dims", mlp_cfg["hidden_dims"]),
                ("use_layernorm", mlp_cfg["use_layernorm"]),
                ("activation", mlp_cfg["activation"]),
            ],
        )

        _print_section("Derived Dims")
        _print_kv("state_dim", state_dim)
        _print_kv("output_dim", 1)
        _print_block_footer(f"[{self.__class__.__name__}] Ready")

    def compute(self, inputs, role=""):
        """执行前向计算并返回状态 value 预测。

        Args:
            inputs: skrl 模型输入字典，使用 `inputs["states"]`。
            role: 模型角色名（当前前向计算未使用）。

        Returns:
            `(value, extras)` 二元组，其中 `extras` 为空字典。
        """
        # 1. 准备输入
        states = unflatten_tensorized_space(self.state_space, inputs.get("states"))

        # 2. 通过 MLP 计算 value
        output = self.mlp(states)

        return output, {}


__all__ = ["MLPPolicy", "MLPValue"]
