# Copyright (c) 2026, Tang yucheng
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

"""网络构建工具。

该模块接收经过 ``config`` 子模块校验与标准化后的配置，
并生成可直接前向计算的 ``torch.nn`` 结构，包含：

- CNN 构建
- MLP 构建
- embedding projection 构建
"""

from __future__ import annotations

from typing import Any, Callable

import torch.nn as nn


def _build_cnn(in_channels: int, cfg: dict[str, Any]) -> nn.Sequential:
    """根据标准化配置构建 CNN 特征提取器。

    Args:
        in_channels: 输入图像通道数。
        cfg: `_normalize_cnn_cfg` 返回的标准化 CNN 配置。

    Returns:
        以 `nn.Flatten()` 结尾的 `nn.Sequential` CNN。
    """
    modules: list[nn.Module] = []
    current_channels = in_channels
    activation_factory = cfg["activation_factory"]
    # 逐层堆叠 Conv + Activation
    for out_channels, kernel_size, stride, padding in zip(
        cfg["channels"], cfg["kernels"], cfg["strides"], cfg["paddings"]
    ):
        modules.append(
            nn.Conv2d(
                current_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        )
        modules.append(activation_factory())
        current_channels = out_channels
    # 统一在尾部展平，方便后续拼接/接 MLP
    modules.append(nn.Flatten())
    return nn.Sequential(*modules)


def _build_mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    use_layernorm: bool,
    activation_factory: Callable[[], nn.Module],
) -> nn.Sequential:
    """构建可配置的 MLP 网络。

    Args:
        input_dim: 输入特征维度。
        hidden_dims: 隐藏层维度列表。
        output_dim: 输出特征维度。
        use_layernorm: 是否在每层隐藏层后添加 `LayerNorm`。
        activation_factory: activation 模块构造器。

    Returns:
        `nn.Sequential` 形式的 MLP。
    """
    modules: list[nn.Module] = []
    current_dim = input_dim
    # 逐层构建 Linear + (LayerNorm) + Activation
    for hidden_dim in hidden_dims:
        modules.append(nn.Linear(current_dim, hidden_dim))
        if use_layernorm:
            modules.append(nn.LayerNorm(hidden_dim))
        modules.append(activation_factory())
        current_dim = hidden_dim
    # 输出层不附加激活，交由上层策略决定
    modules.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*modules)


def _build_embedding(
    input_dim: int,
    output_dim: int,
    *,
    use_layernorm: bool,
    activation_factory: Callable[[], nn.Module],
) -> nn.Sequential:
    """构建结构化输入的 embedding projection 模块。

    Args:
        input_dim: 输入特征维度。
        output_dim: 输出 embedding 维度。
        use_layernorm: 是否使用 `LayerNorm`。
        activation_factory: activation 模块构造器。

    Returns:
        `nn.Sequential` 形式的 projection 模块。
    """
    # embedding 投影采用单层线性映射
    modules: list[nn.Module] = [nn.Linear(input_dim, output_dim)]
    if use_layernorm:
        modules.append(nn.LayerNorm(output_dim))
    modules.append(activation_factory())
    return nn.Sequential(*modules)
