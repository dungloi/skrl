# Copyright (c) 2026, Tang yucheng
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

"""CNN + GRU + Attention + MLP 模型族定义。

该模块面向图像历史需要通过 recurrent 方式建模、且多体交互共同参与决策的场景，包含：

- ``CNNGRUAttentionMLPPolicy``：融合 CNN 图像特征、GRU 时序记忆与 attention 交互特征输出动作均值
- ``CNNGRUAttentionMLPValue``：复用同一融合路径并输出标量 value

网络结构由 ``network.cnn``、``network.attention``、``network.gru``、``network.mlp``
四段配置驱动。
"""

from __future__ import annotations

from typing import Any, Literal

import gymnasium
import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space

from .utils import (
    _build_cnn,
    _build_mlp,
    _normalize_attention_cfg,
    _normalize_cnn_cfg,
    _normalize_gru_cfg,
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
_CNN_GRU_ATTENTION_MLP_REQUIRED_KEYS = {"cnn", "attention", "gru", "mlp"}
# 当前模块输入空间打印优先级（其余键，如动态 other_*，交由自然排序）
_CNN_GRU_ATTENTION_MLP_SPACE_PRIORITY: dict[str, tuple[str, ...]] = {
    "observation_space": ("image", "ego", "others_mask"),
    "state_space": ("image", "ego", "others_mask"),
}


def _parse_other_shape(other_shape: tuple[int, ...]) -> tuple[int, int]:
    """解析 `other_*` 输入 shape，返回 `(history_length, other_input_dim)`。"""
    if len(other_shape) != 2:
        raise ValueError(
            "Invalid `other_*` shape: expected (H, D), where H=1 means no history, "
            f"got {other_shape}"
        )

    history_length = int(other_shape[0])
    other_input_dim = int(other_shape[1])
    if history_length < 1 or other_input_dim < 1:
        raise ValueError(
            "Invalid `other_*` shape values: "
            f"history_length={history_length}, other_input_dim={other_input_dim}"
        )

    return history_length, other_input_dim


def _merge_done_flags(inputs: dict[str, Any]) -> torch.Tensor | None:
    """合并 recurrent 训练中使用的终止信号。"""
    terminated = inputs.get("terminated")
    truncated = inputs.get("truncated")
    if terminated is None:
        return truncated
    if truncated is None:
        return terminated
    return terminated | truncated


class _CNNGRUAttentionMLPCommon:
    """共享 CNN-attention-GRU 主干逻辑。"""

    def _build_common_modules(
        self,
        *,
        input_space: gymnasium.Space,
        output_dim: int,
        num_envs: int,
        network_raw_cfg: Any,
        model_name: str,
    ) -> None:
        # ----------------------------------------
        # 网络结构定义
        # ----------------------------------------
        # 1. 解析输入空间维度（input_space 是一个 Dict，包含 "image"、"ego" 和 "other_i"）
        # 解析 "other_i" 的数量和 shape，假设它们都是同样的 shape
        self.other_keys = sorted([key for key in input_space.keys() if key.startswith("other_")])
        other_shape = input_space[self.other_keys[0]].shape
        self.other_history_length, other_input_dim = _parse_other_shape(other_shape)

        # 解析 "ego" 的 shape
        ego_shape = input_space["ego"].shape
        # 解析 “image” 的 shape
        image_shape = input_space["image"].shape

        self.num_envs = int(num_envs)
        if self.num_envs <= 0:
            raise ValueError(f"[{model_name}] `num_envs` must be a positive integer, got {num_envs}")

        # 模型参数定义
        in_channels = image_shape[0]
        ego_input_dim = ego_shape[0]

        # 2. 从 YAML 读取并严格校验网络结构参数
        network_cfg = _normalize_network_root(
            network_raw_cfg,
            required_keys=_CNN_GRU_ATTENTION_MLP_REQUIRED_KEYS,
            model_name=model_name,
        )
        cnn_cfg = _normalize_cnn_cfg(network_cfg.get("cnn"))
        attention_cfg = _normalize_attention_cfg(network_cfg.get("attention"))
        gru_cfg = _normalize_gru_cfg(network_cfg.get("gru"))
        mlp_cfg = _normalize_mlp_cfg(network_cfg.get("mlp"))

        embed_dim = attention_cfg["embed_dim"]
        num_heads = attention_cfg["num_heads"]
        self.sequence_length = gru_cfg["sequence_length"]
        self.num_layers = gru_cfg["num_layers"]
        self.hidden_size = gru_cfg["hidden_size"]

        # 3. 按配置构建 CNN
        self.cnn = _build_cnn(in_channels, cnn_cfg)
        with torch.no_grad():
            # 创建全0的 dummy 输入来推断 flatten 后的维度
            dummy_img = torch.zeros((1, *image_shape))
            cnn_out = self.cnn(dummy_img)
            cnn_out_dim = cnn_out.shape[1]

        # 4. 定义图像分支 GRU（用于处理跨时间步的图像记忆）
        self.image_gru = nn.GRU(
            input_size=cnn_out_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        # 5. 定义 Attention 部分 (用于处理 "other_i" 的特征交互)
        # 这里我们使用一个简单的 MultiheadAttention 来处理 "other_i" 特征之间的关系
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # 6. 定义 "ego", "other" 的特征映射层
        self.ego_embedding = nn.Sequential(
            nn.Linear(ego_input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
        self.other_embedding = nn.Sequential(
            nn.Linear(other_input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

        # 7. 定义 MLP 部分 (融合特征 -> Action / Value)
        fused_input_dim = self.hidden_size + ego_input_dim + embed_dim * self.other_history_length
        self.mlp = _build_mlp(
            input_dim=fused_input_dim,
            hidden_dims=mlp_cfg["hidden_dims"],
            output_dim=output_dim,
            use_layernorm=mlp_cfg["use_layernorm"],
            activation_factory=mlp_cfg["activation_factory"],
        )

        self._image_shape = image_shape
        self._ego_shape = ego_shape
        self._other_shape = other_shape
        self._cnn_cfg = cnn_cfg
        self._attention_cfg = attention_cfg
        self._mlp_cfg = mlp_cfg
        self._gru_cfg = gru_cfg
        self._cnn_out_dim = cnn_out_dim
        self._fused_input_dim = fused_input_dim

    def get_specification(self) -> dict[str, Any]:
        """返回 skrl recurrent agent 所需的 RNN 规格说明。"""
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [(self.num_layers, self.num_envs, self.hidden_size)],
            }
        }

    def _run_image_gru(
        self, img_features: torch.Tensor, inputs: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """执行图像分支 GRU 前向，并按 skrl recurrent 接口处理 hidden state。"""
        # 1. 读取输入中的 recurrent state；若没有则使用全零初始状态
        rnn_states = inputs.get("rnn")
        if not rnn_states:
            zero_hidden = torch.zeros(
                self.num_layers,
                img_features.shape[0],
                self.hidden_size,
                device=img_features.device,
                dtype=img_features.dtype,
            )
            rnn_output, _ = self.image_gru(img_features.unsqueeze(1), zero_hidden)
            return rnn_output.squeeze(1), None

        # 2. rollout 与 training 的 hidden state 组织方式不同，分别处理
        hidden_states = rnn_states[0]
        if self.training:
            # training 阶段要求 batch 能够按 sequence_length 还原成序列
            if img_features.shape[0] % self.sequence_length != 0:
                raise ValueError(
                    "Invalid recurrent batch size: expected number of samples to be divisible by "
                    f"`sequence_length={self.sequence_length}`, got {img_features.shape[0]}"
                )

            # 将平铺 batch 还原成 (batch_size, sequence_length, feature_dim)
            rnn_input = img_features.view(-1, self.sequence_length, img_features.shape[-1])
            hidden_states = hidden_states.view(
                self.num_layers, -1, self.sequence_length, hidden_states.shape[-1]
            )
            # 只保留每段序列起始位置对应的 hidden state
            hidden_states = hidden_states[:, :, 0, :].contiguous()

            # 若序列中间存在 done，则需要分段运行 GRU 并重置对应 hidden state
            done = _merge_done_flags(inputs)
            if done is not None and torch.any(done):
                rnn_outputs = []
                done = done.view(-1, self.sequence_length)
                indexes = (
                    [0]
                    + (done[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist()
                    + [self.sequence_length]
                )

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, hidden_states = self.image_gru(rnn_input[:, i0:i1, :], hidden_states)
                    hidden_states[:, done[:, i1 - 1], :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_output = torch.cat(rnn_outputs, dim=1)
            # 若序列中没有 done，则可以直接整段运行 GRU
            else:
                rnn_output, hidden_states = self.image_gru(rnn_input, hidden_states)
        # rollout 阶段每次只处理一个时间步，因此 sequence_length = 1
        else:
            rnn_input = img_features.view(-1, 1, img_features.shape[-1])
            rnn_output, hidden_states = self.image_gru(rnn_input, hidden_states)

        # 3. 将 GRU 输出重新展平成后续 MLP 需要的二维 batch
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)
        return rnn_output, hidden_states

    def _compute_features(
        self,
        *,
        space: gymnasium.Space,
        tensor_name: str,
        inputs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # 1. 准备输入
        data = unflatten_tensorized_space(space, inputs.get(tensor_name))

        # 2. 拆分输入
        img = data["image"]
        ego = data["ego"]
        other = torch.stack([data[key] for key in self.other_keys], dim=1)

        # 校验 other 维度
        if other.dim() != 4:
            raise ValueError(
                "Invalid `other_*` tensor shape: expected (batch_size, num_other, history_length, other_dim), "
                "where history_length=1 means no history, "
                f"got {tuple(other.shape)}"
            )

        batch_size, num_other, other_history_length, other_dim = other.shape
        if other_history_length != self.other_history_length:
            raise ValueError(
                "Temporal `other_*` history length mismatch: expected "
                f"{self.other_history_length}, got {other_history_length}"
            )

        # 3. CNN + GRU 处理图像输入
        img_features = self.cnn(img)
        img_features, hidden_states = self._run_image_gru(img_features, inputs)

        # 4. 计算 "ego" 与 "other" 的 embedding
        ego_embedded = self.ego_embedding(ego)
        other_embedded = self.other_embedding(
            other.reshape(batch_size * num_other * other_history_length, other_dim)
        ).reshape(batch_size, num_other, other_history_length, -1)

        # 5. 通过 Cross Attention 处理 "ego" 与 "other" 的交互特征
        other_key_value = other_embedded.permute(0, 2, 1, 3).reshape(
            batch_size * other_history_length, num_other, -1
        )
        attention_output, _ = self.attention(
            query=ego_embedded.unsqueeze(1)
            .expand(-1, other_history_length, -1)
            .reshape(batch_size * other_history_length, 1, -1),
            key=other_key_value,
            value=other_key_value,
        )
        attention_output = attention_output.squeeze(1).reshape(batch_size, other_history_length, -1).reshape(
            batch_size, -1
        )

        # 6. 融合图像特征与交互特征后输入 MLP
        combined = torch.cat([img_features, ego, attention_output], dim=1)
        return self.mlp(combined), hidden_states


class CNNGRUAttentionMLPPolicy(_CNNGRUAttentionMLPCommon, GaussianMixin, Model):
    """融合 CNN 图像、GRU 时序记忆与 attention 特征的 Gaussian policy。

    期望的网络配置:
        network.cnn + network.attention + network.gru + network.mlp
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
        num_envs: int = 1,
        **kwargs,
    ):
        """初始化 CNN-GRU-attention Gaussian policy。

        Args:
            observation_space: 包含 `image`、`ego`、`other_*` 的 Dict space。
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
            num_envs: recurrent 模型对应的环境数，用于构造 hidden state 规格。
            **kwargs: 额外参数，必须包含 `network`，且其中包含
                `cnn`、`attention`、`gru`、`mlp` 配置段。
        """
        # ----------------------------------------
        # 基类初始化
        # ----------------------------------------
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_mean_actions=clip_mean_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
            role=role,
        )

        # ----------------------------------------
        # 网络结构定义
        # ----------------------------------------
        action_shape = action_space.shape
        self._build_common_modules(
            input_space=observation_space,
            output_dim=action_shape[0],
            num_envs=num_envs,
            network_raw_cfg=kwargs.get("network"),
            model_name=self.__class__.__name__,
        )

        # 定义 log_std 参数
        self.log_std_parameter = nn.Parameter(
            torch.full(size=action_space.shape, fill_value=float(initial_log_std), dtype=torch.float32),
            requires_grad=not fixed_log_std,
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
            space_priority=_CNN_GRU_ATTENTION_MLP_SPACE_PRIORITY,
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
        _print_kv("image_shape", self._image_shape)
        _print_kv("ego_shape", self._ego_shape)
        _print_kv("other_shape", self._other_shape)
        _print_kv("num_other", len(self.other_keys))
        _print_kv("action_shape", action_shape)
        _print_kv("num_envs", self.num_envs)

        _print_section("Network Config")
        _print_subconfig(
            "cnn",
            [
                ("channels", self._cnn_cfg["channels"]),
                ("kernels", self._cnn_cfg["kernels"]),
                ("strides", self._cnn_cfg["strides"]),
                ("paddings", self._cnn_cfg["paddings"]),
                ("activation", self._cnn_cfg["activation"]),
            ],
        )
        _print_subconfig(
            "attention",
            [
                ("embed_dim", self._attention_cfg["embed_dim"]),
                ("num_heads", self._attention_cfg["num_heads"]),
            ],
        )
        _print_subconfig(
            "gru",
            [
                ("hidden_size", self._gru_cfg["hidden_size"]),
                ("num_layers", self._gru_cfg["num_layers"]),
                ("sequence_length", self._gru_cfg["sequence_length"]),
            ],
        )
        _print_subconfig(
            "mlp",
            [
                ("hidden_dims", self._mlp_cfg["hidden_dims"]),
                ("use_layernorm", self._mlp_cfg["use_layernorm"]),
                ("activation", self._mlp_cfg["activation"]),
            ],
        )

        _print_section("Derived Dims")
        _print_kv("cnn_out_dim", self._cnn_out_dim)
        _print_kv("gru_hidden_size", self.hidden_size)
        _print_kv("fused_input_dim", self._fused_input_dim)
        _print_block_footer(f"[{self.__class__.__name__}] Ready")

    def compute(self, inputs, role=""):
        """执行前向计算并返回 Gaussian mean、log std 与可选 RNN state。

        Args:
            inputs: skrl 模型输入字典，使用 `inputs["observations"]`。
            role: 模型角色名（当前前向计算未使用）。

        Returns:
            `(mean_actions, extras)` 二元组，其中 `extras` 至少包含
            `{"log_std": self.log_std_parameter}`；若存在 recurrent state，
            还会包含 `{"rnn": [hidden_states]}`。
        """
        # 1. 通过共享主干提取融合特征
        output, hidden_states = self._compute_features(
            space=self.observation_space,
            tensor_name="observations",
            inputs=inputs,
        )

        # 2. 组织 skrl 所需返回格式
        extras: dict[str, Any] = {"log_std": self.log_std_parameter}
        if hidden_states is not None:
            extras["rnn"] = [hidden_states]
        return output, extras


class CNNGRUAttentionMLPValue(_CNNGRUAttentionMLPCommon, DeterministicMixin, Model):
    """融合 CNN 图像、GRU 时序记忆与 attention 特征的 deterministic value 模型。

    期望的网络配置:
        network.cnn + network.attention + network.gru + network.mlp
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
        num_envs: int = 1,
        **kwargs,
    ):
        """初始化 CNN-GRU-attention value 模型。

        Args:
            observation_space: Observation space（为接口一致性保留）。
            state_space: 包含 `image`、`ego`、`other_*` 的 Dict space。
            action_space: skrl 接口使用的动作空间。
            device: 目标 torch device。
            clip_actions: deterministic mixin 中是否裁剪动作。
            role: skrl 使用的模型角色名。
            num_envs: recurrent 模型对应的环境数，用于构造 hidden state 规格。
            **kwargs: 额外参数，必须包含 `network`，且其中包含
                `cnn`、`attention`、`gru`、`mlp` 配置段。
        """
        # ----------------------------------------
        # 基类初始化
        # ----------------------------------------
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=clip_actions, role=role)

        # ----------------------------------------
        # 网络结构定义
        # ----------------------------------------
        self._build_common_modules(
            input_space=state_space,
            output_dim=1,
            num_envs=num_envs,
            network_raw_cfg=kwargs.get("network"),
            model_name=self.__class__.__name__,
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
            space_priority=_CNN_GRU_ATTENTION_MLP_SPACE_PRIORITY,
        )
        _print_deterministic_settings(clip_actions=clip_actions, role=role)

        _print_section("Parsed Shapes")
        _print_kv("image_shape", self._image_shape)
        _print_kv("ego_shape", self._ego_shape)
        _print_kv("other_shape", self._other_shape)
        _print_kv("num_other", len(self.other_keys))
        _print_kv("num_envs", self.num_envs)

        _print_section("Network Config")
        _print_subconfig(
            "cnn",
            [
                ("channels", self._cnn_cfg["channels"]),
                ("kernels", self._cnn_cfg["kernels"]),
                ("strides", self._cnn_cfg["strides"]),
                ("paddings", self._cnn_cfg["paddings"]),
                ("activation", self._cnn_cfg["activation"]),
            ],
        )
        _print_subconfig(
            "attention",
            [
                ("embed_dim", self._attention_cfg["embed_dim"]),
                ("num_heads", self._attention_cfg["num_heads"]),
            ],
        )
        _print_subconfig(
            "gru",
            [
                ("hidden_size", self._gru_cfg["hidden_size"]),
                ("num_layers", self._gru_cfg["num_layers"]),
                ("sequence_length", self._gru_cfg["sequence_length"]),
            ],
        )
        _print_subconfig(
            "mlp",
            [
                ("hidden_dims", self._mlp_cfg["hidden_dims"]),
                ("use_layernorm", self._mlp_cfg["use_layernorm"]),
                ("activation", self._mlp_cfg["activation"]),
            ],
        )

        _print_section("Derived Dims")
        _print_kv("cnn_out_dim", self._cnn_out_dim)
        _print_kv("gru_hidden_size", self.hidden_size)
        _print_kv("fused_input_dim", self._fused_input_dim)
        _print_kv("output_dim", 1)
        _print_block_footer(f"[{self.__class__.__name__}] Ready")

    def compute(self, inputs, role=""):
        """执行前向计算并返回状态 value 预测与可选 RNN state。

        Args:
            inputs: skrl 模型输入字典，使用 `inputs["states"]`。
            role: 模型角色名（当前前向计算未使用）。

        Returns:
            `(value, extras)` 二元组，其中 `extras` 在 recurrent 模式下
            包含 `{"rnn": [hidden_states]}`，否则为空字典。
        """
        # 1. 通过共享主干提取融合特征
        output, hidden_states = self._compute_features(
            space=self.state_space,
            tensor_name="states",
            inputs=inputs,
        )

        # 2. 组织 skrl 所需返回格式
        extras: dict[str, Any] = {}
        if hidden_states is not None:
            extras["rnn"] = [hidden_states]
        return output, extras


__all__ = ["CNNGRUAttentionMLPPolicy", "CNNGRUAttentionMLPValue"]
