# Copyright (c) 2026, Tang yucheng
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

"""配置解析与校验工具。

该模块负责把 ``network`` 原始配置转换为可直接用于构建网络的标准化字典，
并在入口阶段执行严格的字段与数值校验，避免误配置在训练中后期才暴露。

当前覆盖的配置段包括：
- 顶层 ``network`` 根配置
- ``network.attention``
- ``network.cnn``
- ``network.embedding``
- ``network.gru``
- ``network.mlp``
"""

from __future__ import annotations

from typing import Any, Callable

import torch.nn as nn


def _to_2d_tuple(value: Any, field_name: str) -> tuple[int, int]:
    """将卷积参数标准化为 2D tuple。

    Args:
        value: 标量 int，或包含 2 个 int 的 list/tuple。
        field_name: 配置字段名，用于报错信息定位。

    Returns:
        `(height, width)` 形式的 tuple。

    Raises:
        ValueError: 当 `value` 不是 int 或合法 2 元整数序列时抛出。
    """
    # 标量 int 直接扩展成 (v, v)
    if isinstance(value, int):
        return value, value
    # 二元整数序列按 (h, w) 解释
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(v, int) for v in value):
        return int(value[0]), int(value[1])
    raise ValueError(f"Invalid `{field_name}` item: {value}. Expected int or [int, int]")


def _check_unknown_keys(cfg: dict[str, Any], allowed_keys: set[str], field_name: str) -> None:
    """校验配置字典中是否包含未知 key。

    Args:
        cfg: 待校验的配置映射。
        allowed_keys: 允许出现的 key 集合。
        field_name: 配置路径，用于报错信息定位。

    Raises:
        ValueError: 当 `cfg` 出现 `allowed_keys` 之外的 key 时抛出。
    """
    # 通过集合差快速找出非法字段
    unknown_keys = sorted(set(cfg.keys()) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"Invalid `{field_name}` keys: {unknown_keys}. Allowed keys: {sorted(allowed_keys)}")


def _normalize_activation_name(value: Any, field_name: str) -> str:
    """校验并标准化 activation 名称。

    Args:
        value: 配置中的 activation 名称。
        field_name: 配置路径，用于报错信息定位。

    Returns:
        小写形式的 activation 名称。

    Raises:
        ValueError: 当类型非法或 activation 不受支持时抛出。
    """
    # 先做类型校验，避免后续 lower() 报错不清晰
    if not isinstance(value, str):
        raise ValueError(f"Invalid `{field_name}` type: {type(value)}. Expected string")
    name = value.lower()
    supported = {"relu", "elu", "gelu", "tanh", "silu", "leaky_relu"}
    if name not in supported:
        raise ValueError(f"Invalid `{field_name}`: {value}. Supported: {sorted(supported)}")
    return name


def _get_activation_factory(name: str) -> Callable[[], nn.Module]:
    """将 activation 名称映射为 torch 模块构造器。

    Args:
        name: 小写 activation 名称。

    Returns:
        无参 callable，调用后返回对应 activation 模块。
    """
    # 名称到 torch 模块构造器的静态映射
    mapping: dict[str, Callable[[], nn.Module]] = {
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
        "leaky_relu": nn.LeakyReLU,
    }
    return mapping[name]


def _normalize_network_root(
    raw_cfg: Any,
    *,
    required_keys: set[str],
    model_name: str,
    optional_keys: set[str] | None = None,
) -> dict[str, Any]:
    """校验模型级 `network` 配置。

    Args:
        raw_cfg: 原始 `network` 配置对象。
        required_keys: 必需的一级模块 key 集合。
        model_name: 模型名，用于报错信息上下文。
        optional_keys: 当前模型允许但不强制提供的一级模块 key 集合。

    Returns:
        通过校验的 `network` 映射。

    Raises:
        ValueError: 当配置类型不合法、缺少必需 key、或包含未知 key 时抛出。
    """
    # 顶层 network 必须是映射类型
    if not isinstance(raw_cfg, dict):
        raise ValueError(
            f"[{model_name}] Missing or invalid `network` config: expected keys {sorted(required_keys)}"
        )
    # 顶层只允许当前模型声明的必需或可选模块 key
    allowed_keys = required_keys | (optional_keys or set())
    _check_unknown_keys(raw_cfg, allowed_keys, f"{model_name}.network")
    missing_keys = sorted(required_keys - set(raw_cfg.keys()))
    if missing_keys:
        raise ValueError(f"[{model_name}] Missing required `network` keys: {missing_keys}")
    return raw_cfg


def _normalize_ego_cfg(raw_cfg: Any, *, default_fusion_dim: int) -> dict[str, int]:
    """校验 ego 直接融合分支配置，并为旧配置提供兼容默认值。"""
    if raw_cfg is None:
        return {"fusion_dim": default_fusion_dim}
    if not isinstance(raw_cfg, dict):
        raise ValueError("Invalid `network.ego` config: expected a mapping")

    required_keys = {"fusion_dim"}
    _check_unknown_keys(raw_cfg, required_keys, "network.ego")
    missing_keys = sorted(required_keys - set(raw_cfg.keys()))
    if missing_keys:
        raise ValueError(f"Missing required `network.ego` keys: {missing_keys}")

    fusion_dim = raw_cfg["fusion_dim"]
    if not isinstance(fusion_dim, int) or fusion_dim <= 0:
        raise ValueError(
            f"Invalid `network.ego.fusion_dim`: expected a positive integer, got {fusion_dim}"
        )
    return {"fusion_dim": fusion_dim}


def _normalize_cnn_cfg(raw_cfg: Any) -> dict[str, Any]:
    """校验并标准化 CNN 配置。

    Args:
        raw_cfg: 原始 `network.cnn` 配置。

    Returns:
        标准化后的 CNN 配置，包含 tuple 化空间参数、输出特征维度和
        activation factory。

    Raises:
        ValueError: 当缺少必需字段、存在未知字段、或字段类型/形状非法时抛出。
    """
    # CNN 子配置必须是映射类型
    if not isinstance(raw_cfg, dict):
        raise ValueError("Invalid `network.cnn` config: expected a mapping")
    # 先校验字段完整性与合法性，再做值校验
    required_keys = {"channels", "kernels", "strides", "paddings", "activation"}
    allowed_keys = required_keys | {"output_dim"}
    _check_unknown_keys(raw_cfg, allowed_keys, "network.cnn")
    missing_keys = sorted(required_keys - set(raw_cfg.keys()))
    if missing_keys:
        raise ValueError(f"Missing required `network.cnn` keys: {missing_keys}")

    # 读取字段并做基础归一化
    channels = raw_cfg["channels"]
    kernels = raw_cfg["kernels"]
    strides = raw_cfg["strides"]
    paddings = raw_cfg["paddings"]
    activation = _normalize_activation_name(raw_cfg["activation"], "network.cnn.activation")
    output_dim = raw_cfg.get("output_dim", 64)

    if not isinstance(channels, (list, tuple)) or len(channels) == 0:
        raise ValueError("Invalid `network.cnn.channels`: expected a non-empty list of positive integers")
    if not isinstance(kernels, (list, tuple)) or not isinstance(strides, (list, tuple)) or not isinstance(
        paddings, (list, tuple)
    ):
        raise ValueError("Invalid `network.cnn`: `kernels`, `strides`, and `paddings` must be lists")
    if not all(isinstance(ch, int) and ch > 0 for ch in channels):
        raise ValueError("Invalid `network.cnn.channels`: all channel values must be positive integers")
    if not isinstance(output_dim, int) or output_dim <= 0:
        raise ValueError(
            f"Invalid `network.cnn.output_dim`: expected a positive integer, got {output_dim}"
        )
    if not all(isinstance(v, (int, list, tuple)) for v in list(kernels) + list(strides) + list(paddings)):
        raise ValueError("Invalid `network.cnn` shape fields: use int or [int, int] per layer")

    # 每一层的 channels / kernel / stride / padding 必须一一对应
    if not (len(channels) == len(kernels) == len(strides) == len(paddings)):
        raise ValueError(
            "Invalid `network.cnn` config: `channels`, `kernels`, `strides`, and `paddings` must have same length"
        )

    # 返回可直接用于构建网络的标准化配置
    return {
        "channels": [int(ch) for ch in channels],
        "kernels": [_to_2d_tuple(v, "network.cnn.kernels") for v in kernels],
        "strides": [_to_2d_tuple(v, "network.cnn.strides") for v in strides],
        "paddings": [_to_2d_tuple(v, "network.cnn.paddings") for v in paddings],
        "output_dim": output_dim,
        "activation": activation,
        "activation_factory": _get_activation_factory(activation),
    }


def _normalize_attention_cfg(raw_cfg: Any) -> dict[str, int]:
    """校验并标准化 attention 配置。

    Args:
        raw_cfg: 原始 `network.attention` 配置。

    Returns:
        标准化后的 attention 配置，包含 `embed_dim` 与 `num_heads`。

    Raises:
        ValueError: 当缺少必需字段、类型非法、数值非正、或
            `embed_dim` 不能被 `num_heads` 整除时抛出。
    """
    # attention 子配置必须是映射类型
    if not isinstance(raw_cfg, dict):
        raise ValueError("Invalid `network.attention` config: expected a mapping")
    required_keys = {"embed_dim", "num_heads"}
    _check_unknown_keys(raw_cfg, required_keys, "network.attention")
    missing_keys = sorted(required_keys - set(raw_cfg.keys()))
    if missing_keys:
        raise ValueError(f"Missing required `network.attention` keys: {missing_keys}")
    if not isinstance(raw_cfg["embed_dim"], int) or not isinstance(raw_cfg["num_heads"], int):
        raise ValueError("Invalid `network.attention`: `embed_dim` and `num_heads` must be integers")

    # 读取字段后再做数值约束
    embed_dim = raw_cfg["embed_dim"]
    num_heads = raw_cfg["num_heads"]
    if embed_dim <= 0 or num_heads <= 0:
        raise ValueError("Invalid `network.attention`: `embed_dim` and `num_heads` must be positive integers")
    if embed_dim % num_heads != 0:
        raise ValueError("Invalid `network.attention`: `embed_dim` must be divisible by `num_heads`")
    return {"embed_dim": embed_dim, "num_heads": num_heads}


def _normalize_gru_cfg(raw_cfg: Any) -> dict[str, Any]:
    """校验并标准化 GRU 配置。

    Args:
        raw_cfg: 原始 `network.gru` 配置。

    Returns:
        标准化后的 GRU 配置，包含 `hidden_size`、`num_layers`、`sequence_length`
        与可选的 `separate_feature_projection`。

    Raises:
        ValueError: 当缺少必需字段、存在未知字段、类型非法或数值非正时抛出。
    """
    # GRU 子配置必须是映射类型
    if not isinstance(raw_cfg, dict):
        raise ValueError("Invalid `network.gru` config: expected a mapping")

    # 先校验字段完整性与合法性，再做值校验
    required_keys = {"hidden_size", "num_layers", "sequence_length"}
    allowed_keys = required_keys | {"separate_feature_projection"}
    _check_unknown_keys(raw_cfg, allowed_keys, "network.gru")
    missing_keys = sorted(required_keys - set(raw_cfg.keys()))
    if missing_keys:
        raise ValueError(f"Missing required `network.gru` keys: {missing_keys}")

    # 读取字段后再做数值约束
    cfg = {
        "hidden_size": raw_cfg["hidden_size"],
        "num_layers": raw_cfg["num_layers"],
        "sequence_length": raw_cfg["sequence_length"],
        "separate_feature_projection": raw_cfg.get("separate_feature_projection", False),
    }
    for key in required_keys:
        value = cfg[key]
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"Invalid `network.gru.{key}`: expected a positive integer, got {value}")
    if not isinstance(cfg["separate_feature_projection"], bool):
        raise ValueError(
            "Invalid `network.gru.separate_feature_projection`: expected a boolean, "
            f"got {cfg['separate_feature_projection']!r}"
        )
    return cfg


def _normalize_mlp_cfg(raw_cfg: Any) -> dict[str, Any]:
    """校验并标准化 MLP 配置。

    Args:
        raw_cfg: 原始 `network.mlp` 配置。

    Returns:
        标准化后的 MLP 配置，包含维度列表和 activation factory。

    Raises:
        ValueError: 当缺少必需字段、存在未知字段、或字段类型/形状非法时抛出。
    """
    # MLP 子配置必须是映射类型
    if not isinstance(raw_cfg, dict):
        raise ValueError("Invalid `network.mlp` config: expected a mapping")
    required_keys = {"hidden_dims", "use_layernorm", "activation"}
    _check_unknown_keys(raw_cfg, required_keys, "network.mlp")
    missing_keys = sorted(required_keys - set(raw_cfg.keys()))
    if missing_keys:
        raise ValueError(f"Missing required `network.mlp` keys: {missing_keys}")

    # 读取字段并做语义校验
    hidden_dims = raw_cfg["hidden_dims"]
    use_layernorm = raw_cfg["use_layernorm"]
    activation = _normalize_activation_name(raw_cfg["activation"], "network.mlp.activation")

    if not isinstance(hidden_dims, (list, tuple)) or len(hidden_dims) == 0:
        raise ValueError("Invalid `network.mlp.hidden_dims`: expected a non-empty list of positive integers")
    if not all(isinstance(dim, int) and dim > 0 for dim in hidden_dims):
        raise ValueError("Invalid `network.mlp.hidden_dims`: all dimensions must be positive integers")
    if not isinstance(use_layernorm, bool):
        raise ValueError("Invalid `network.mlp.use_layernorm`: expected boolean")

    # 返回可直接用于构建网络的标准化配置
    return {
        "hidden_dims": [int(dim) for dim in hidden_dims],
        "use_layernorm": use_layernorm,
        "activation": activation,
        "activation_factory": _get_activation_factory(activation),
    }


def _normalize_embedding_cfg(raw_cfg: Any) -> dict[str, Any]:
    """校验并标准化 embedding projection 配置。

    Args:
        raw_cfg: 原始 `network.embedding` 配置。

    Returns:
        标准化后的 embedding 配置，包含 activation factory。

    Raises:
        ValueError: 当缺少必需字段、存在未知字段、或字段类型非法时抛出。
    """
    # embedding 子配置必须是映射类型
    if not isinstance(raw_cfg, dict):
        raise ValueError("Invalid `network.embedding` config: expected a mapping")
    required_keys = {"use_layernorm", "activation"}
    _check_unknown_keys(raw_cfg, required_keys, "network.embedding")
    missing_keys = sorted(required_keys - set(raw_cfg.keys()))
    if missing_keys:
        raise ValueError(f"Missing required `network.embedding` keys: {missing_keys}")

    # 读取并校验 embedding 相关字段
    use_layernorm = raw_cfg["use_layernorm"]
    activation = _normalize_activation_name(raw_cfg["activation"], "network.embedding.activation")
    if not isinstance(use_layernorm, bool):
        raise ValueError("Invalid `network.embedding.use_layernorm`: expected boolean")

    # 返回可直接用于构建网络的标准化配置
    return {
        "use_layernorm": use_layernorm,
        "activation": activation,
        "activation_factory": _get_activation_factory(activation),
    }
