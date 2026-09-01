"""QuantLinear：hk.Linear 的量化 drop-in 替换。"""

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from .config import get_quant_config
from .quantizer import quantized_linear


class QuantLinear(hk.Linear):
    """行为等价于 hk.Linear，但矩阵乘走 INT8 量化路径。

    模块名强制回落到 "linear"：haiku 默认取类名的 snake_case，会让参数树 key
    变成 quant_linear，与 checkpoint 里的 dacer_policy_net/linear 对不上。
    """

    # 直接吃原始 obs 的那一层的模块名。obs 未归一化、各维量纲差异大，是量化误差
    # 的主要来源。注意策略网里 linear / linear_1 是时间编码支路（输入是正弦嵌入，
    # 范围本就在 [-1, 1]，量化无害），真正吃 obs 的是主干首层 linear_2。
    _FIRST_LAYER_NAMES = ("q_net/linear", "dacer_policy_net/linear_2")

    def __init__(self, output_size: int, *args, name: str | None = None, **kwargs):
        super().__init__(output_size, *args, name=name or "linear", **kwargs)

    def _is_first_layer(self) -> bool:
        return self.module_name in self._FIRST_LAYER_NAMES

    def _force_fp32(self, config) -> bool:
        if config.skip_first_layer and self._is_first_layer():
            return True
        return self.module_name in config.fp32_modules

    def __call__(self, inputs: jax.Array, *, precision=None) -> jax.Array:
        if not inputs.shape:
            raise ValueError("Input must not be scalar.")

        input_size = self.input_size = inputs.shape[-1]
        dtype = inputs.dtype

        w_init = self.w_init
        if w_init is None:
            stddev = 1.0 / np.sqrt(input_size)
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)
        w = hk.get_parameter("w", [input_size, self.output_size], dtype, init=w_init)

        config = get_quant_config()
        if config is not None and self._force_fp32(config):
            config = None

        if not self.with_bias:
            if config is None or not config.enabled:
                return jnp.dot(inputs, w, precision=precision)
            zero_bias = jnp.zeros((self.output_size,), dtype)
            return quantized_linear(inputs, w, zero_bias, config)

        b = hk.get_parameter("b", [self.output_size], dtype, init=self.b_init)

        if config is None or not config.enabled:
            out = jnp.dot(inputs, w, precision=precision)
            return out + jnp.broadcast_to(b, out.shape)

        return quantized_linear(inputs, w, b, config)


__all__ = ["QuantLinear"]
