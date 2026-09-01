"""量化配置与全局开关。

量化算子需要在 haiku 追踪网络时就知道配置，而 hk.Module 的构造过程夹在
create_diffv2_net 内部、没法透传参数，所以这里用一个模块级全局配置。
调用顺序必须是 set_quant_config() 先于建网。
"""

from dataclasses import dataclass, replace
from typing import Optional

MODE_OFF = "off"
MODE_INT = "int"
MODE_FAKE = "fake"
VALID_MODES = (MODE_OFF, MODE_INT, MODE_FAKE)


@dataclass(frozen=True)
class QuantConfig:
    """MLP 层的 INT8 量化配置。

    mode:
        off  - 不量化，走原始 FP32 matmul
        int  - int8 x int8 -> int32 真整数累加，再反量化
        fake - 量化-反量化后走 FP32 matmul，用于隔离数值问题
    weight_per_channel:
        权重按输出通道分别取 scale。实测 per-tensor 会让回报掉约 21%。
    act_symmetric:
        激活是否用对称量化。mish 输出下界恒为 -0.3088、分布单侧偏斜，
        所以默认用非对称（带 zero-point）以免浪费一半码字。
    quantize_act:
        关掉则只量化权重，用于消融。
    act_per_token:
        激活 scale 只沿最后一维统计（每个样本/粒子各自一个 scale），而不是整个
        张量共享。这一项对本模型至关重要：Q 网首层输入是 concat(obs, act)，
        obs 的 absmax 约 20 而动作只在 [-1, 1]，共享 scale 会把动作压到只剩
        几个码字，而 Q 网的职责恰恰是区分 32 个候选动作的细微差别。
    skip_first_layer:
        输入层保持 FP32。首层直接吃原始 obs（量纲混杂且未归一化），是量化误差
        的主要来源，跳过它的收益通常远大于代价。
    fp32_modules:
        额外保持 FP32 的模块名（完整 haiku module_name，如 "q_net/linear_3"）。
        用于逐层敏感度分析，以及给个别敏感层开后门。
    act_group_size:
        大于 0 时启用分组激活量化：输入维每 group_size 个分量共享一个 scale，
        组内 int8 累加、组间 FP32 求和。实测这是 INT8 下的决定性一招 ——
        整层共享 scale 时回报掉 46%，而仅把激活放宽到 16bit 只掉 3.2%，
        说明误差几乎全来自激活 scale 粒度太粗。权重与激活仍然都是 INT8。
    """

    mode: str = MODE_INT
    weight_bits: int = 8
    act_bits: int = 8
    weight_per_channel: bool = True
    act_symmetric: bool = False
    quantize_act: bool = True
    act_per_token: bool = True
    # 下面两项的默认值来自实测：3 个 seed、每个 1000 步，
    # group8 + skip-first 平均回报 11481 vs FP32 11948（-3.9%）。
    skip_first_layer: bool = True
    fp32_modules: tuple[str, ...] = ()
    act_group_size: int = 8

    def __post_init__(self):
        if self.mode not in VALID_MODES:
            raise ValueError(f"Invalid quant mode {self.mode!r}, expected one of {VALID_MODES}")
        if not 2 <= self.weight_bits <= 16:
            raise ValueError(f"weight_bits must be in [2, 16], got {self.weight_bits}")
        if not 2 <= self.act_bits <= 16:
            raise ValueError(f"act_bits must be in [2, 16], got {self.act_bits}")
        if self.act_group_size < 0:
            raise ValueError(f"act_group_size must be >= 0, got {self.act_group_size}")
        # int 模式在 int32 上累加：8bit x 8bit x 256 项 ~ 8.5e6，安全裕度充足；
        # 但 16bit x 16bit x 256 项 ~ 2.1e9 会越过 int32 上界并静默回绕。
        # 所以超过 8bit 只允许走 fake（FP32 matmul）路径做误差归因，不允许走 int。
        if self.mode == MODE_INT and max(self.weight_bits, self.act_bits) > 8:
            raise ValueError(
                "mode='int' 只支持 <=8bit（int32 累加器限制）；"
                f"当前 weight_bits={self.weight_bits} act_bits={self.act_bits}，"
                "请改用 mode='fake' 做位宽归因"
            )

    @property
    def enabled(self) -> bool:
        return self.mode != MODE_OFF

    @property
    def use_grouped_act(self) -> bool:
        return self.act_group_size > 0 and self.quantize_act


_CONFIG: Optional[QuantConfig] = None


def set_quant_config(config: Optional[QuantConfig]) -> None:
    global _CONFIG
    _CONFIG = config


def get_quant_config() -> Optional[QuantConfig]:
    return _CONFIG


def quant_enabled() -> bool:
    return _CONFIG is not None and _CONFIG.enabled


def describe() -> str:
    if not quant_enabled():
        return "off"
    c = _CONFIG
    parts = [
        f"mode={c.mode}",
        f"w{c.weight_bits}" + ("/per-channel" if c.weight_per_channel else "/per-tensor"),
    ]
    if c.quantize_act:
        act = f"a{c.act_bits}" + ("/sym" if c.act_symmetric else "/asym")
        if c.use_grouped_act:
            act += f"/group{c.act_group_size}"
        else:
            act += "/per-token" if c.act_per_token else "/per-tensor"
        parts.append(act)
    else:
        parts.append("a=fp32")
    if c.skip_first_layer:
        parts.append("skip-first")
    if c.fp32_modules:
        parts.append("fp32=" + ",".join(c.fp32_modules))
    return " ".join(parts)


__all__ = [
    "QuantConfig",
    "MODE_OFF",
    "MODE_INT",
    "MODE_FAKE",
    "set_quant_config",
    "get_quant_config",
    "quant_enabled",
    "describe",
    "replace",
]
