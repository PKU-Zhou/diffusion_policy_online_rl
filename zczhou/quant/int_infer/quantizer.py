"""INT8 量化算子。

全部是纯函数，可以在 jax.jit / jax.vmap 下追踪。

约定：
  - 权重对称量化，零点恒为 0，scale 可以是 per-output-channel
  - 激活非对称动态量化，范围强制包含 0 以保证零点可精确表示
  - 累加在 int32 上做，bias 保持 FP32 最后相加
"""

import jax
import jax.numpy as jnp

EPS = 1e-12


def _sym_qmax(bits: int) -> int:
    """对称量化的正向最大码字，int8 对应 127。"""
    return 2 ** (bits - 1) - 1


def _asym_qmax(bits: int) -> int:
    """非对称量化的码字上界，int8 对应 255。"""
    return 2**bits - 1


def quantize_weight(w: jax.Array, per_channel: bool = True, bits: int = 8):
    """权重对称量化。

    返回 (wq, scale, colsum)：
      wq     - 整数码字，保持浮点 dtype 以便走 fake 路径，int 路径再转 int8
      scale  - per_channel 时形状 (1, out)，否则标量
      colsum - wq 沿输入维的列和，用于补偿激活零点，形状 (out,)
    """
    qmax = _sym_qmax(bits)
    if per_channel:
        amax = jnp.max(jnp.abs(w), axis=0, keepdims=True)
    else:
        amax = jnp.max(jnp.abs(w))
    scale = jnp.maximum(amax / qmax, EPS)
    wq = jnp.clip(jnp.round(w / scale), -qmax, qmax)
    colsum = jnp.sum(wq, axis=0)
    return wq, scale, colsum


def quantize_act(x: jax.Array, bits: int = 8, symmetric: bool = False, per_token: bool = True):
    """激活动态量化，范围由当前张量实时统计。

    per_token=True 时只沿最后一维归约并保留维度，每一行拿到自己的 scale；
    否则整个张量共享一个 scale。返回 (xq, scale, zero_point)，
    scale / zero_point 的形状可直接广播回 x，对称模式下 zero_point 为 0。
    """
    axis = -1 if per_token else None

    if symmetric:
        qmax = _sym_qmax(bits)
        amax = jnp.max(jnp.abs(x), axis=axis, keepdims=per_token)
        scale = jnp.maximum(amax / qmax, EPS)
        xq = jnp.clip(jnp.round(x / scale), -qmax, qmax)
        return xq, scale, jnp.zeros_like(scale)

    qmax = _asym_qmax(bits)
    # 把 0 纳入范围，否则零点落在区间外、无法精确表示
    lo = jnp.minimum(jnp.min(x, axis=axis, keepdims=per_token), 0.0)
    hi = jnp.maximum(jnp.max(x, axis=axis, keepdims=per_token), 0.0)
    scale = jnp.maximum((hi - lo) / qmax, EPS)
    zero_point = jnp.round(-lo / scale)
    xq = jnp.clip(jnp.round(x / scale) + zero_point, 0, qmax)
    return xq, scale, zero_point


def _storage_dtype(bits: int, signed: bool):
    """能装下该位宽码字的最窄整数 dtype。8bit 走 int8/uint8，更宽则用 16bit。"""
    if bits <= 8:
        return jnp.int8 if signed else jnp.uint8
    return jnp.int16 if signed else jnp.uint16


def _pad_to_multiple(x: jax.Array, w: jax.Array, group_size: int):
    """把输入维补齐到 group_size 的整数倍。

    激活补 0、权重补 0，则补出来的那些项贡献恒为 0，不影响结果。
    这样 group_size 不必整除各层输入维（17 / 23 / 39 / 256 混在一起）。
    """
    k = x.shape[-1]
    rem = (-k) % group_size
    if rem == 0:
        return x, w
    x = jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, rem)])
    w = jnp.pad(w, [(0, rem), (0, 0)])
    return x, w


def grouped_quantized_linear(x, w, b, config):
    """分组激活量化的整数线性层。

    把输入维切成每组 group_size 个，每组独立统计激活 scale/zero-point，
    组内做 int8 x int8 -> int32 累加，各组反量化后在 FP32 上求和。

    动机（实测）：整个 256 维共享一个 scale 时端到端回报掉 46%，而把激活放宽到
    16bit 只掉 3.2% —— 说明误差几乎全部来自激活 scale 太粗，少数大分量撑大了
    步长。分组把这种浪费限制在组内，权重和激活仍然都是 INT8。
    """
    out_dtype = x.dtype
    gs = config.act_group_size
    x, w = _pad_to_multiple(x, w, gs)
    groups = x.shape[-1] // gs

    xg = x.reshape(*x.shape[:-1], groups, gs)
    wg = w.reshape(groups, gs, w.shape[-1])

    wq, w_scale, _ = quantize_weight(w, config.weight_per_channel, config.weight_bits)
    wqg = wq.reshape(groups, gs, w.shape[-1])
    colsum_g = jnp.sum(wqg, axis=1)

    xq, x_scale, zero_point = quantize_act(
        xg, config.act_bits, config.act_symmetric, per_token=True
    )

    xq_i = xq.astype(_storage_dtype(config.act_bits, config.act_symmetric)).astype(jnp.int32)
    wq_i = wqg.astype(_storage_dtype(config.weight_bits, True)).astype(jnp.int32)
    # (..., G, gs) x (G, gs, out) -> (..., G, out)，G 维成对匹配而非收缩
    acc = jnp.einsum("...gk,gko->...go", xq_i, wq_i, preferred_element_type=jnp.int32)

    compensated = acc.astype(out_dtype) - zero_point.astype(out_dtype) * colsum_g.astype(out_dtype)
    partial = compensated * x_scale.astype(out_dtype)
    out = jnp.sum(partial, axis=-2) * jnp.reshape(w_scale, (-1,)).astype(out_dtype)
    return out + b.astype(out_dtype)


def fake_grouped_linear(x, w, b, config):
    """grouped_quantized_linear 的 fake 对照，用于验证整数路径的反量化公式。"""
    out_dtype = x.dtype
    gs = config.act_group_size
    x, w = _pad_to_multiple(x, w, gs)
    groups = x.shape[-1] // gs
    xg = x.reshape(*x.shape[:-1], groups, gs)
    xd = fake_quant_act(xg, config.act_bits, config.act_symmetric, per_token=True)
    wd = fake_quant_weight(w, config.weight_per_channel, config.weight_bits)
    return jnp.einsum("...gk,gko->...o", xd, wd.reshape(groups, gs, w.shape[-1])) + b.astype(
        out_dtype
    )


def int_matmul(xq: jax.Array, wq: jax.Array, act_signed: bool,
               act_bits: int = 8, weight_bits: int = 8) -> jax.Array:
    """整数矩阵乘，int32 累加。

    激活非对称量化时码字落在 [0, 2^bits-1]，装不进有符号类型，必须用无符号；
    对称量化时范围对称，用有符号。权重恒为有符号。
    两个操作数符号不同，先各自窄化以保住语义，再统一提升到 int32 交给
    dot_general，避免混合 dtype 在 XLA 上被拒。窄化这一步是刻意保留的：
    它让「码字真的能用该位宽表示」成为一个会被 dtype 兜住的事实，而不是约定。
    """
    xq = xq.astype(_storage_dtype(act_bits, act_signed))
    wq = wq.astype(_storage_dtype(weight_bits, True))
    return jax.lax.dot_general(
        xq.astype(jnp.int32),
        wq.astype(jnp.int32),
        (((xq.ndim - 1,), (0,)), ((), ())),
        preferred_element_type=jnp.int32,
    )


def dequantize(acc, act_scale, weight_scale, zero_point, colsum, bias, out_dtype):
    """把 int32 累加结果还原到浮点并加上 FP32 bias。

    展开 sum_k (xq[k] - z) * wq[k] = acc - z * colsum，
    再乘回两个 scale。weight_scale 形状 (1, out) 时会自然广播。
    per-token 时 act_scale / zero_point 形状为 (..., 1)，沿 out 维广播同样成立。
    """
    compensated = acc.astype(out_dtype) - zero_point.astype(out_dtype) * colsum.astype(out_dtype)
    # per-channel 时 weight_scale 形状是 (1, out)，reshape 成 (out,) 才能沿最后一维
    # 广播到任意 batch 形状；per-tensor 时是标量，reshape(-1) 得到 (1,) 同样安全
    w_scale = jnp.reshape(weight_scale, (-1,)).astype(out_dtype)
    out = compensated * act_scale.astype(out_dtype) * w_scale
    return out + bias.astype(out_dtype)


def fake_quant_weight(w: jax.Array, per_channel: bool = True, bits: int = 8) -> jax.Array:
    wq, scale, _ = quantize_weight(w, per_channel, bits)
    return wq * scale


def fake_quant_act(
    x: jax.Array, bits: int = 8, symmetric: bool = False, per_token: bool = True
) -> jax.Array:
    xq, scale, zero_point = quantize_act(x, bits, symmetric, per_token)
    return (xq - zero_point) * scale


def quantized_linear(x, w, b, config):
    """按配置执行一次量化线性层，等价于 x @ w + b。"""
    if config.use_grouped_act:
        if config.mode == "fake":
            return fake_grouped_linear(x, w, b, config)
        if not config.quantize_act:
            raise ValueError("grouped act requires quantize_act=True")
        return grouped_quantized_linear(x, w, b, config)

    out_dtype = x.dtype

    if config.mode == "fake":
        wd = fake_quant_weight(w, config.weight_per_channel, config.weight_bits)
        if config.quantize_act:
            xd = fake_quant_act(x, config.act_bits, config.act_symmetric, config.act_per_token)
        else:
            xd = x
        return xd @ wd + b

    # 只量化权重时没有整数激活可用，退回浮点乘
    if not config.quantize_act:
        wd = fake_quant_weight(w, config.weight_per_channel, config.weight_bits)
        return x @ wd + b

    wq, w_scale, colsum = quantize_weight(w, config.weight_per_channel, config.weight_bits)
    xq, x_scale, zero_point = quantize_act(
        x, config.act_bits, config.act_symmetric, config.act_per_token
    )
    acc = int_matmul(xq, wq, act_signed=config.act_symmetric,
                     act_bits=config.act_bits, weight_bits=config.weight_bits)
    return dequantize(acc, x_scale, w_scale, zero_point, colsum, b, out_dtype)


__all__ = [
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
