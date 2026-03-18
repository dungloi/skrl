# Copyright (c) 2026, Tang yucheng
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

"""custom_models 的统一导出入口。

该包聚焦于可通过 YAML 配置驱动的自定义 policy / value 网络，
向外统一导出以下模型族：

- MLP：纯向量输入的 MLP policy/value
- CNN + MLP：图像与状态融合的 policy/value
- Attention + MLP：ego-other 交互建模的 policy/value
- CNN + Attention + MLP：图像与交互特征联合建模的 policy/value

外部模块通常只需要从本入口导入模型类，无需感知子模块拆分细节。
"""

# Attention + MLP：ego-other 交互建模
from .attention_mlp import AttentionMLPPolicy, AttentionMLPValue
# CNN + Attention + MLP：图像与交互联合建模
from .cnn_attention_mlp import CNNAttentionMLPPolicy, CNNAttentionMLPValue
# CNN + MLP：图像与状态特征融合
from .cnn_mlp import CNNMLPPolicy, CNNMLPValue
# MLP：纯向量输入
from .mlp import MLPPolicy, MLPValue

__all__ = [
    # MLP
    "MLPPolicy",
    "MLPValue",
    # CNN + MLP
    "CNNMLPPolicy",
    "CNNMLPValue",
    # Attention + MLP
    "AttentionMLPPolicy",
    "AttentionMLPValue",
    # CNN + Attention + MLP
    "CNNAttentionMLPPolicy",
    "CNNAttentionMLPValue",
]
