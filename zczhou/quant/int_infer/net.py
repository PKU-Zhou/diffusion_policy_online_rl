"""从 checkpoint 目录重建 Diffv2Net。

single_infer.py 默认加载的 deterministic.pkl 是训练时固化的 jaxpr，里面只有
dot_general 这类裸算子，没有可插桩的位置。要做量化就必须从源码重新构图。

重建结果已验证与固化图逐位一致（MAX DIFF = 0.0）。
"""

import dataclasses
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import yaml

from relax.network import blocks
from relax.network.diffv2 import create_diffv2_net

from .config import QuantConfig, set_quant_config
from .layers import QuantLinear

QUANT_TARGETS = ("both", "policy", "q", "none")


def load_train_config(log_dir: Path) -> dict:
    with open(Path(log_dir) / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def _mish(x: jax.Array) -> jax.Array:
    return x * jnp.tanh(jax.nn.softplus(x))


def _create_net(cfg: dict, obs_dim: int, act_dim: int):
    """按训练时的参数构图。

    这里刻意照抄 scripts/train_mujoco.py 的 sdac 分支：只传四个关键字参数，
    beta_schedule_type 走 Diffv2Net 的 dataclass 默认值，不额外传入。
    init 用的 key 只决定初始权重，随后会被 checkpoint 整体覆盖，故种子无关。
    """
    hidden_sizes = [cfg["hidden_dim"]] * cfg["hidden_num"]
    diffusion_hidden_sizes = [cfg["diffusion_hidden_dim"]] * cfg["hidden_num"]
    net, _ = create_diffv2_net(
        jax.random.key(0),
        obs_dim,
        act_dim,
        hidden_sizes,
        diffusion_hidden_sizes,
        _mish,
        num_timesteps=cfg["diffusion_steps"],
        num_particles=cfg["num_particles"],
        noise_scale=cfg["noise_scale"],
        beta_schedule_scale=cfg["beta_schedule_scale"],
    )
    return net


def _quantized(fn):
    """让 fn 在执行期间使用 QuantLinear。

    注意 relax.network.blocks.mlp() 是在 hk.transform 的 **apply 期间**才调用
    get_linear_factory() 的（haiku 每次 apply 都会重新追踪模块），所以只在建网时
    临时设置工厂是无效的 —— 那样只有形状初始化用到了 QuantLinear，真正推理时
    工厂已经被恢复成 hk.Linear。这里把设置动作包到每次调用外面，量化才真正生效，
    同时也让 q / policy 两支可以各自独立选择是否量化。
    """
    def wrapped(*args, **kwargs):
        prev = blocks.get_linear_factory()
        blocks.set_linear_factory(QuantLinear)
        try:
            return fn(*args, **kwargs)
        finally:
            blocks.set_linear_factory(prev)

    return wrapped


def build_net(
    log_dir: Path,
    obs_dim: int,
    act_dim: int,
    quant_config: Optional[QuantConfig] = None,
    quant_target: str = "both",
):
    """重建网络，按 quant_target 决定哪一支走量化。

    quant_target:
        both   - 策略网与 Q 网都量化
        policy - 只量化策略网（扩散去噪，每步动作要跑 num_timesteps 次）
        q      - 只量化 Q 网（候选动作打分）
        none   - 都不量化，用于自证源码路径与固化图等价
    """
    if quant_target not in QUANT_TARGETS:
        raise ValueError(f"Invalid quant_target {quant_target!r}, expected one of {QUANT_TARGETS}")

    cfg = load_train_config(log_dir)

    if quant_config is None or not quant_config.enabled or quant_target == "none":
        set_quant_config(None)
        blocks.set_linear_factory(None)
        return _create_net(cfg, obs_dim, act_dim)

    set_quant_config(quant_config)
    # 用默认 hk.Linear 建网即可：QuantLinear 的参数名与形状完全一致，
    # 量化通过下面包装 apply 的方式注入。
    blocks.set_linear_factory(None)
    net = _create_net(cfg, obs_dim, act_dim)

    if quant_target in ("both", "q"):
        net = dataclasses.replace(net, q=_quantized(net.q))
    if quant_target in ("both", "policy"):
        net = dataclasses.replace(net, policy=_quantized(net.policy))
    return net


def make_policy_fn(net):
    """包成 single_infer.py 期望的 policy_fn(params, obs) 形式。

    clip(-1, 1) 与 relax/trainer/evaluator.py 的评估口径保持一致。
    """
    return jax.jit(lambda params, obs: net.get_deterministic_action(params, obs).clip(-1, 1))


__all__ = ["build_net", "make_policy_fn", "load_train_config", "QUANT_TARGETS"]
