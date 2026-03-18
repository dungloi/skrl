# Copyright (c) 2026, Tang yucheng
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

"""custom_models 工具函数统一导出入口。

该子包按职责拆分为三类工具：

- config：YAML 配置解析与严格校验
- builders：根据标准化配置构建 torch 网络模块
- printing：模型初始化阶段的结构化打印

模型文件通常只从本入口导入工具函数，以减少跨文件依赖噪声。
"""

# builders：网络结构构建
from .builders import (
    _build_cnn,
    _build_embedding,
    _build_mlp,
)
# config：配置解析与校验
from .config import (
    _normalize_activation_name,
    _normalize_attention_cfg,
    _normalize_cnn_cfg,
    _normalize_embedding_cfg,
    _normalize_mlp_cfg,
    _normalize_network_root,
)
# printing：初始化打印与调试输出
from .printing import (
    _print_block_footer,
    _print_block_header,
    _print_deterministic_settings,
    _print_gaussian_settings,
    _print_inputs,
    _print_kv,
    _print_section,
    _print_subconfig,
)

__all__ = [
    # builders
    "_build_cnn",
    "_build_embedding",
    "_build_mlp",
    # config
    "_normalize_activation_name",
    "_normalize_attention_cfg",
    "_normalize_cnn_cfg",
    "_normalize_embedding_cfg",
    "_normalize_mlp_cfg",
    "_normalize_network_root",
    # printing
    "_print_block_footer",
    "_print_block_header",
    "_print_deterministic_settings",
    "_print_gaussian_settings",
    "_print_inputs",
    "_print_kv",
    "_print_section",
    "_print_subconfig",
]
