"""MLP 层 INT8 量化推理算子。

典型用法（注意 set_quant_config 必须在建网之前调用）：

    from zczhou.quant.int_infer import QuantConfig, set_quant_config
    from zczhou.quant.int_infer.net import build_net

    set_quant_config(QuantConfig(mode="int"))
    net = build_net(log_dir, obs_dim, act_dim, quant_target="both")
"""

from .config import (
    MODE_FAKE,
    MODE_INT,
    MODE_OFF,
    QuantConfig,
    describe,
    get_quant_config,
    quant_enabled,
    set_quant_config,
)
from .layers import QuantLinear
from .quantizer import (
    dequantize,
    fake_grouped_linear,
    fake_quant_act,
    fake_quant_weight,
    grouped_quantized_linear,
    int_matmul,
    quantize_act,
    quantize_weight,
    quantized_linear,
)

__all__ = [
    "QuantConfig",
    "MODE_OFF",
    "MODE_INT",
    "MODE_FAKE",
    "set_quant_config",
    "get_quant_config",
    "quant_enabled",
    "describe",
    "QuantLinear",
    "quantize_weight",
    "quantize_act",
    "int_matmul",
    "dequantize",
    "fake_quant_weight",
    "fake_quant_act",
    "grouped_quantized_linear",
    "fake_grouped_linear",
    "quantized_linear",
]
