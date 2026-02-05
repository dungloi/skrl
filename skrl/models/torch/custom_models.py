from __future__ import annotations

from typing import Any, Literal

import gymnasium

import torch
import torch.nn as nn

from skrl.models.torch import Model, DeterministicMixin, GaussianMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space




class CNNMLPPolicy(GaussianMixin, Model):
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
        # 调用基类初始化
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

        # 打印配置信息
        print("\n")
        print("============================ CNNMLPPolicy ============================")
        print("###### [Model Initialization]")
        print("observation_space:", observation_space)
        print("state_space:", state_space)
        print("action_space:", action_space)
        print("device:", device)
        print("\n")
        print("###### [GaussianMixin Initialization]")
        print("clip_actions:", clip_actions)
        print("clip_mean_actions:", clip_mean_actions)
        print("clip_log_std:", clip_log_std)
        print("min_log_std:", min_log_std)
        print("max_log_std:", max_log_std)
        print("reduction:", reduction)
        print("role:", role)
        print("\n")
        print("###### [Others]")
        print("initial_log_std:", initial_log_std)
        print("fixed_log_std:", fixed_log_std)
        print("======================================================================\n")

        # ------------------- 网络结构定义 -------------------

        # 1. 解析输入空间维度
        # observation_space 是一个 Dict，包含 "image" 和 "state"
        # image shape 为 (C, H, W)
        image_shape  = observation_space["image"].shape
        state_shape  = observation_space["state"].shape
        action_shape = action_space.shape
        
        in_channels = image_shape[0]
        state_dim   = state_shape[0]
        action_dim  = action_shape[0]

        # 2. 定义 CNN 部分 (用于处理 observation['image'])
        # 这是一个简单的 3 层 Conv 结构
        # 针对 32(H) x 256(W) 长条形图像的特殊优化结构
        self.cnn = nn.Sequential(
            # 第一层：非对称压缩
            # Kernel=5 (比8小以适应短边), Stride=(2, 4) (高度/2, 宽度/4)
            # H: (32-5)/2 + 1 = 14
            # W: (256-5)/4 + 1 = 63
            # Out: 32 x 14 x 63
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=(2, 4)),
            nn.ReLU(),
            # 第二层：均匀压缩
            # Kernel=3, Stride=2
            # H: (14-3)/2 + 1 = 6
            # W: (63-3)/2 + 1 = 31
            # Out: 64 x 6 x 31
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            # 第三层：只压缩宽度，保留高度细节
            # Kernel=3, Stride=(1, 2) (高度不减，宽度/2)
            # H: (6-3)/1 + 1 = 4
            # W: (31-3)/2 + 1 = 15
            # Out: 64 x 4 x 15
            nn.Conv2d(64, 64, kernel_size=3, stride=(1, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # 3. 计算 CNN 输出维度 (通过一次 dummy forward)
        with torch.no_grad():
            # 创建全0的 dummy 输入来推断 flatten 后的维度
            dummy_img = torch.zeros((1, *image_shape))
            cnn_out = self.cnn(dummy_img)
            cnn_out_dim = cnn_out.shape[1]

        print(f"Constructed CNN. Input Channels: {in_channels}, Flattened Output Dim: {cnn_out_dim}")

        # 4. 定义 MLP 部分 (CNN特征 + State特征 -> Action)
        self.mlp = nn.Sequential(
            nn.Linear(cnn_out_dim + state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, action_dim)  # 输出动作均值
        )

        self.log_std_parameter = nn.Parameter(
            torch.full(size=action_space.shape, fill_value=float(initial_log_std), dtype=torch.float32),
            requires_grad=not fixed_log_std
        )


    def compute(self, inputs, role=""):

        # 把扁平的 tensor 还原回字典结构
        observations = unflatten_tensorized_space(self.observation_space, inputs.get("observations"))
        # states = unflatten_tensorized_space(self.state_space, inputs.get("states"))
        # taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))

        # # 测试用代码段
        # print("\n")
        # print("============================ CNNMLPPolicy ============================")
        # print("observations [image]:", observations["image"].shape)
        # print("observations [state]:", observations["state"].shape)
        # print("states [image]:", states["image"].shape)
        # print("states [state]:", states["state"].shape)
        # if taken_actions is not None:
        #     print("taken_actions:", taken_actions.shape)
        # output = torch.zeros((observations["image"].shape[0], 4), dtype=observations["image"].dtype, device=observations["image"].device)

        # 前向传播
        img = observations["image"]
        state = observations["state"]
        # 1. 提取特征
        img_features = self.cnn(img)
        # 2. 拼接特征 (Dim 1 是特征维度)
        combined_features = torch.cat([img_features, state], dim=1)
        # 3. 通过 MLP 计算动作均值
        output = self.mlp(combined_features)
        # 这里不需要处理 taken_actions，因为这是 compute forward
        
        return output, {"log_std": self.log_std_parameter}




class CNNMLPValue(DeterministicMixin, Model):
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
        # 调用基类初始化
        Model.__init__(
            self,
            observation_space=observation_space, state_space=state_space,
            action_space=action_space, device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=clip_actions, role=role)

        # 打印配置信息
        print("\n")
        print("============================ CNNMLPValue =============================")
        print("###### [Model Initialization]")
        print("observation_space:", observation_space)
        print("state_space:", state_space)
        print("action_space:", action_space)
        print("device:", device)
        print("\n")
        print("###### [DeterministicMixin Initialization]")
        print("clip_actions:", clip_actions)
        print("role:", role)
        print("======================================================================\n")

        # ------------------- 网络结构定义 -------------------

        # 1. 解析输入空间维度
        # observation_space 是一个 Dict，包含 "image" 和 "state"
        # image shape 为 (C, H, W)
        image_shape  = state_space["image"].shape
        state_shape  = state_space["state"].shape
        
        in_channels = image_shape[0]
        state_dim   = state_shape[0]

        # 2. 定义 CNN 部分 (用于处理 observation['image'])
        # 这是一个简单的 3 层 Conv 结构
        # 针对 32(H) x 256(W) 长条形图像的特殊优化结构
        self.cnn = nn.Sequential(
            # 第一层：非对称压缩
            # Kernel=5 (比8小以适应短边), Stride=(2, 4) (高度/2, 宽度/4)
            # H: (32-5)/2 + 1 = 14
            # W: (256-5)/4 + 1 = 63
            # Out: 32 x 14 x 63
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=(2, 4)),
            nn.ReLU(),
            # 第二层：均匀压缩
            # Kernel=3, Stride=2
            # H: (14-3)/2 + 1 = 6
            # W: (63-3)/2 + 1 = 31
            # Out: 64 x 6 x 31
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            # 第三层：只压缩宽度，保留高度细节
            # Kernel=3, Stride=(1, 2) (高度不减，宽度/2)
            # H: (6-3)/1 + 1 = 4
            # W: (31-3)/2 + 1 = 15
            # Out: 64 x 4 x 15
            nn.Conv2d(64, 64, kernel_size=3, stride=(1, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # 3. 计算 CNN 输出维度 (通过一次 dummy forward)
        with torch.no_grad():
            # 创建全0的 dummy 输入来推断 flatten 后的维度
            dummy_img = torch.zeros((1, *image_shape))
            cnn_out = self.cnn(dummy_img)
            cnn_out_dim = cnn_out.shape[1]

        print(f"Constructed CNN. Input Channels: {in_channels}, Flattened Output Dim: {cnn_out_dim}")

        # 4. 定义 MLP 部分 (CNN特征 + State特征 -> Action)
        self.mlp = nn.Sequential(
            nn.Linear(cnn_out_dim + state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出 value
        )


    def compute(self, inputs, role=""):

        # 把扁平的 tensor 还原回字典结构
        # observations = unflatten_tensorized_space(self.observation_space, inputs.get("observations"))
        states = unflatten_tensorized_space(self.state_space, inputs.get("states"))
        # taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))

        # # 测试用代码段
        # print("\n")
        # print("============================ CNNMLPValue =============================")
        # print("observations [image]:", observations["image"].shape)
        # print("observations [state]:", observations["state"].shape)
        # print("states [image]:", states["image"].shape)
        # print("states [state]:", states["state"].shape)
        # if taken_actions is not None:
        #     print("taken_actions:", taken_actions.shape)
        # output = torch.zeros((observations["image"].shape[0], 1), dtype=observations["image"].dtype, device=observations["image"].device)  # 占位符

        # 前向传播
        img = states["image"]
        state = states["state"]
        # 1. 提取特征
        img_features = self.cnn(img)
        # 2. 拼接特征 (Dim 1 是特征维度)
        combined_features = torch.cat([img_features, state], dim=1)
        # 3. 通过 MLP 计算动作均值
        output = self.mlp(combined_features)
        # 这里不需要处理 taken_actions，因为这是 compute forward
        
        return output, {}