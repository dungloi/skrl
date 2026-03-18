# Copyright (c) 2026, Tang yucheng
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

"""初始化打印工具。

该模块提供一组统一的格式化输出函数，用于在模型构建阶段打印：

- 输入空间信息
- Gaussian / deterministic 关键参数
- 分组网络配置
- 派生维度信息

目标是让不同模型家族在日志中保持一致的可读性。
"""

from __future__ import annotations

import re
from typing import Any


def _natural_name_key(name: str) -> tuple[tuple[int, int | str], ...]:
    """为 Dict space 的 key 生成自然排序键。

    Args:
        name: 待排序的字段名。

    Returns:
        由 token 组成的排序键。数字 token 与字符串 token 会被拆分并标准化，
        以保证自然排序效果（如 ``other_2`` 在 ``other_10`` 前）。

    Notes:
        不依赖硬编码 key 名，适配新增字段。
    """
    # 按数字片段拆分并统一为可比较的 token 序列
    # 例如: "other_10" -> [("other_", str), (10, int)]
    chunks = re.split(r"(\d+)", name)
    tokens: list[tuple[int, int | str]] = []
    for chunk in chunks:
        if chunk == "":
            continue
        if chunk.isdigit():
            tokens.append((0, int(chunk)))
        else:
            tokens.append((1, chunk.lower()))
    return tuple(tokens)


def _space_sort_key(name: str, priority_order: dict[str, int]) -> tuple[int, int | tuple[tuple[int, int | str], ...]]:
    """为 key 生成最终排序键（可选优先级 + 自然排序）。

    Args:
        name: 待排序的字段名。
        priority_order: 优先级映射（key -> 顺序索引）。

    Returns:
        排序键。命中优先级的 key 会排在前面，其余 key 使用自然排序。
    """
    if name in priority_order:
        return 0, priority_order[name]
    return 1, _natural_name_key(name)


def _extract_dict_space(space: Any) -> dict[str, Any] | None:
    """提取 Dict space 的子空间映射。

    Args:
        space: 待解析的 space 对象，或映射对象。

    Returns:
        若可解析为 Dict 子空间映射则返回 ``dict[str, Any]``，否则返回 ``None``。
    """
    # gymnasium.spaces.Dict 的子空间在 .spaces 中
    spaces_attr = getattr(space, "spaces", None)
    if isinstance(spaces_attr, dict):
        return spaces_attr
    # 兼容直接传入 mapping 的场景
    if isinstance(space, dict):
        return space
    return None


def _print_space(name: str, space: Any, *, priority_keys: tuple[str, ...] | list[str] | None = None) -> None:
    """按统一风格打印 space。

    Args:
        name: space 名称（如 ``observation_space``）。
        space: space 对象，支持普通 space 与 Dict space。
        priority_keys: 可选优先打印 key 列表。命中的 key 会先按给定顺序输出，
            其余 key 按自然排序输出。
    """
    subspaces = _extract_dict_space(space)
    if subspaces is None:
        _print_kv(name, space)
        return

    # 可选优先级：命中的 key 先打印，剩余 key 按自然排序。
    priority_order = {key: idx for idx, key in enumerate(priority_keys or ())}

    print(f"  {name}:")
    for key, subspace in sorted(subspaces.items(), key=lambda item: _space_sort_key(item[0], priority_order)):
        _print_kv(key, subspace, indent=4)


def _print_block_header(title: str, width: int = 80) -> None:
    """打印分块头部。

    Args:
        title: 分块标题。
        width: 横线宽度。
    """
    # 先换行，避免和上一个块粘连
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def _print_block_footer(title: str, width: int = 80) -> None:
    """打印分块尾部。

    Args:
        title: 分块标题。
        width: 横线宽度。
    """
    print("=" * width)
    print(title)
    # 结尾补空行，便于区分后续日志
    print("=" * width + "\n")


def _print_section(title: str) -> None:
    """打印子分段分隔线。

    Args:
        title: 子分段标题。
    """
    print(f"-------------------- {title} --------------------")


def _print_kv(key: str, value: Any, *, indent: int = 2, key_width: int = 24) -> None:
    """打印对齐的 key-value 行。

    Args:
        key: 字段名。
        value: 字段值。
        indent: 左侧缩进空格数。
        key_width: key 对齐宽度。
    """
    # 通过固定 key 宽度提升列对齐可读性
    print(f"{' ' * indent}{key:<{key_width}}: {value}")


def _print_inputs(
    observation_space: Any,
    state_space: Any,
    action_space: Any,
    device: Any,
    *,
    space_priority: dict[str, tuple[str, ...] | list[str]] | None = None,
) -> None:
    """打印模型输入空间信息。

    Args:
        observation_space: Observation space 定义。
        state_space: State space 定义。
        action_space: Action space 定义。
        device: 模型运行的 torch device。
        space_priority: 可选打印优先级配置。可用 key 为
            `"observation_space"` / `"state_space"` / `"action_space"`，
            value 为该空间内希望优先打印的子 key 列表。
    """
    # 统一打印输入相关元信息；Dict 空间逐行展开，提高可读性
    _print_section("Inputs")
    space_priority = space_priority or {}
    _print_space(
        "observation_space",
        observation_space,
        priority_keys=space_priority.get("observation_space"),
    )
    _print_space(
        "state_space",
        state_space,
        priority_keys=space_priority.get("state_space"),
    )
    _print_space(
        "action_space",
        action_space,
        priority_keys=space_priority.get("action_space"),
    )
    _print_kv("device", device)


def _print_gaussian_settings(
    *,
    clip_actions: bool,
    clip_mean_actions: bool,
    clip_log_std: bool,
    min_log_std: float,
    max_log_std: float,
    reduction: str,
    initial_log_std: float,
    fixed_log_std: bool,
) -> None:
    """打印 Gaussian policy 相关配置。

    Args:
        clip_actions: 是否裁剪动作。
        clip_mean_actions: 是否裁剪动作均值。
        clip_log_std: 是否裁剪 log std。
        min_log_std: log std 下界。
        max_log_std: log std 上界。
        reduction: Gaussian 输出的 reduction 模式。
        initial_log_std: log std 初始值。
        fixed_log_std: 是否固定 log std（不可训练）。
    """
    # 统一打印 Gaussian 策略关键超参数
    _print_section("Gaussian")
    _print_kv("clip_actions", clip_actions)
    _print_kv("clip_mean_actions", clip_mean_actions)
    _print_kv("clip_log_std", clip_log_std)
    _print_kv("min_log_std", min_log_std)
    _print_kv("max_log_std", max_log_std)
    _print_kv("reduction", reduction)
    _print_kv("initial_log_std", initial_log_std)
    _print_kv("fixed_log_std", fixed_log_std)


def _print_deterministic_settings(*, clip_actions: bool, role: str) -> None:
    """打印 deterministic 模型相关配置。

    Args:
        clip_actions: 是否裁剪动作。
        role: 模型角色名。
    """
    # 统一打印 deterministic 模型关键参数
    _print_section("Deterministic")
    _print_kv("clip_actions", clip_actions)
    _print_kv("role", role)


def _print_subconfig(name: str, items: list[tuple[str, Any]]) -> None:
    """打印分组配置块。

    Args:
        name: 分组名。
        items: 待打印的 key-value 列表。
    """
    # 分组名称单独一行，组内字段缩进显示
    print(f"  {name}:")
    for key, value in items:
        _print_kv(key, value, indent=4)
