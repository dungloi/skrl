# Copyright (c) 2026, Tang yucheng
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

"""CNN + MLP 模型族定义。

该模块聚焦“图像 + 状态向量”输入融合场景，包含：

- ``CNNMLPPolicy``：先做 CNN 特征提取，再与状态拼接输出动作均值
- ``CNNMLPValue``：共享相同融合思路，输出标量 value

网络结构由 ``network.cnn`` 与 ``network.mlp`` 两段配置共同驱动。
"""

from __future__ import annotations

from typing import Literal

import gymnasium
import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space

from .utils import (
    _build_cnn,
    _build_mlp,
    _normalize_cnn_cfg,
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
_CNN_MLP_REQUIRED_KEYS = {"cnn", "mlp"}
# 当前模块输入空间打印优先级（其余键，如动态 other_*，交由自然排序）
_CNN_MLP_SPACE_PRIORITY: dict[str, tuple[str, ...]] = {
    "observation_space": ("image", "state"),
    "state_space": ("image", "state"),
}


class CNNMLPPolicy(GaussianMixin, Model):
    """使用 CNN 图像编码与 MLP 融合头的 Gaussian policy。

    期望的网络配置:
        network.cnn + network.mlp
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
        """初始化 CNN-MLP Gaussian policy。

        Args:
            observation_space: 包含 `image` 与 `state` 的 Dict space。
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
                `cnn` 与 `mlp` 配置段。
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
        # observation_space 是一个 Dict，包含 "image" 和 "state"
        # image shape 为 (C, H, W)
        image_shape  = observation_space["image"].shape
        state_shape  = observation_space["state"].shape
        action_shape = action_space.shape
        
        in_channels = image_shape[0]
        state_dim   = state_shape[0]
        action_dim  = action_shape[0]

        # 2. 从 YAML 读取并严格校验网络结构参数
        network_cfg = _normalize_network_root(
            kwargs.get("network", None), required_keys=_CNN_MLP_REQUIRED_KEYS, model_name=self.__class__.__name__
        )
        cnn_cfg = _normalize_cnn_cfg(network_cfg.get("cnn"))
        mlp_cfg = _normalize_mlp_cfg(network_cfg.get("mlp"))

        # 3. 按配置构建 CNN
        self.cnn = _build_cnn(in_channels, cnn_cfg)

        # 4. 计算 CNN 输出维度 (通过一次 dummy forward)
        with torch.no_grad():
            # 创建全0的 dummy 输入来推断 flatten 后的维度
            dummy_img = torch.zeros((1, *image_shape))
            cnn_out = self.cnn(dummy_img)
            cnn_out_dim = cnn_out.shape[1]
        combined_input_dim = cnn_out_dim + state_dim

        # 5. 按配置构建 MLP (CNN特征 + State特征 -> Action)
        self.mlp = _build_mlp(
            input_dim=combined_input_dim,
            hidden_dims=mlp_cfg["hidden_dims"],
            output_dim=action_dim,
            use_layernorm=mlp_cfg["use_layernorm"],
            activation_factory=mlp_cfg["activation_factory"],
        )

        # 6. 定义 log_std 参数
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
            space_priority=_CNN_MLP_SPACE_PRIORITY,
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
        _print_kv("image_shape", image_shape)
        _print_kv("state_shape", state_shape)
        _print_kv("action_shape", action_shape)

        _print_section("Network Config")
        _print_subconfig(
            "cnn",
            [
                ("channels", cnn_cfg["channels"]),
                ("kernels", cnn_cfg["kernels"]),
                ("strides", cnn_cfg["strides"]),
                ("paddings", cnn_cfg["paddings"]),
                ("activation", cnn_cfg["activation"]),
            ],
        )
        _print_subconfig(
            "mlp",
            [
                ("hidden_dims", mlp_cfg["hidden_dims"]),
                ("use_layernorm", mlp_cfg["use_layernorm"]),
                ("activation", mlp_cfg["activation"]),
            ],
        )

        _print_section("Derived Dims")
        _print_kv("cnn_out_dim", cnn_out_dim)
        _print_kv("combined_input_dim", combined_input_dim)
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
        img = observations["image"]
        state = observations["state"]

        # 2. 提取图像特征
        img_features = self.cnn(img)

        # 3. 拼接图像特征和状态特征
        combined_features = torch.cat([img_features, state], dim=1)

        # 4. 通过 MLP 计算动作均值
        output = self.mlp(combined_features)

        return output, {"log_std": self.log_std_parameter}


class CNNMLPValue(DeterministicMixin, Model):
    """使用 CNN 图像编码与 MLP 头的 deterministic value 模型。

    期望的网络配置:
        network.cnn + network.mlp
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
        """初始化 CNN-MLP value 模型。

        Args:
            observation_space: Observation space（为接口一致性保留）。
            state_space: 包含 `image` 与 `state` 的 Dict space。
            action_space: skrl 接口使用的动作空间。
            device: 目标 torch device。
            clip_actions: deterministic mixin 中是否裁剪动作。
            role: skrl 使用的模型角色名。
            **kwargs: 额外参数，必须包含 `network`，且其中包含
                `cnn` 与 `mlp` 配置段。
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
        # state_space 是一个 Dict，包含 "image" 和 "state"
        # image shape 为 (C, H, W)
        image_shape  = state_space["image"].shape
        state_shape  = state_space["state"].shape
        
        in_channels = image_shape[0]
        state_dim   = state_shape[0]

        # 2. 从 YAML 读取并严格校验网络结构参数
        network_cfg = _normalize_network_root(
            kwargs.get("network", None), required_keys=_CNN_MLP_REQUIRED_KEYS, model_name=self.__class__.__name__
        )
        cnn_cfg = _normalize_cnn_cfg(network_cfg.get("cnn"))
        mlp_cfg = _normalize_mlp_cfg(network_cfg.get("mlp"))

        # 3. 按配置构建 CNN
        self.cnn = _build_cnn(in_channels, cnn_cfg)

        # 4. 计算 CNN 输出维度 (通过一次 dummy forward)
        with torch.no_grad():
            # 创建全0的 dummy 输入来推断 flatten 后的维度
            dummy_img = torch.zeros((1, *image_shape))
            cnn_out = self.cnn(dummy_img)
            cnn_out_dim = cnn_out.shape[1]
        combined_input_dim = cnn_out_dim + state_dim

        # 5. 按配置构建 MLP (CNN特征 + State特征 -> Value)
        self.mlp = _build_mlp(
            input_dim=combined_input_dim,
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
            space_priority=_CNN_MLP_SPACE_PRIORITY,
        )
        _print_deterministic_settings(clip_actions=clip_actions, role=role)

        _print_section("Parsed Shapes")
        _print_kv("image_shape", image_shape)
        _print_kv("state_shape", state_shape)

        _print_section("Network Config")
        _print_subconfig(
            "cnn",
            [
                ("channels", cnn_cfg["channels"]),
                ("kernels", cnn_cfg["kernels"]),
                ("strides", cnn_cfg["strides"]),
                ("paddings", cnn_cfg["paddings"]),
                ("activation", cnn_cfg["activation"]),
            ],
        )
        _print_subconfig(
            "mlp",
            [
                ("hidden_dims", mlp_cfg["hidden_dims"]),
                ("use_layernorm", mlp_cfg["use_layernorm"]),
                ("activation", mlp_cfg["activation"]),
            ],
        )

        _print_section("Derived Dims")
        _print_kv("cnn_out_dim", cnn_out_dim)
        _print_kv("combined_input_dim", combined_input_dim)
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
        img = states["image"]
        state = states["state"]

        # 2. 提取图像特征
        img_features = self.cnn(img)

        # 3. 拼接图像特征和状态特征
        combined_features = torch.cat([img_features, state], dim=1)

        # 4. 通过 MLP 计算 value
        output = self.mlp(combined_features)

        return output, {}


__all__ = ["CNNMLPPolicy", "CNNMLPValue"]
