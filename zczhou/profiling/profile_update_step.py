"""
SDAC update 阶段延迟细分微基准

背景
----
`SDAC.stateless_update` 被 `jax.jit` 整体编译成单一 XLA 图（见 relax/algorithm/base.py
的 `_implement_common_behavior`），因此无法在其内部插入 Python 计时器。本脚本把
`stateless_update` 的各个子步骤按代码顺序镜像出来，每个子步骤单独 jit 编译并用
`jax.block_until_ready` 强制同步后计时，从而得到 update 内部的延迟分布。

同时运行完整融合版（算法真实调用的 `_update`）作为对照，两者差值即 XLA 跨阶段
融合带来的收益。

用法
----
    python zczhou/profiling/profile_update_step.py [--num-iters 200] [--gpu 3]

所有超参与 zczhou/configs/profiling_short.json 保持一致。
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import optax

from relax.algorithm.sdac import SDAC
from relax.network.diffv2 import create_diffv2_net
from relax.utils.experience import Experience

sys.path.insert(0, str(SCRIPT_DIR))
from utils import Timer, TimerRegistry  # noqa: E402


# ---------------------------------------------------------------------------
# 与 zczhou/configs/profiling_short.json 对齐的超参
# ---------------------------------------------------------------------------
CONFIG = dict(
    env="HalfCheetah-v4",
    obs_dim=17,
    act_dim=6,
    batch_size=256,        # OffPolicyTrainer 默认 batch_size
    hidden_num=3,
    hidden_dim=256,
    diffusion_steps=20,
    diffusion_hidden_dim=256,
    lr=3e-4,
    lr_schedule_end=3e-5,
    alpha_lr=7e-3,
    delay_alpha_update=250,
    num_particles=32,
    noise_scale=0.1,
    beta_schedule_scale=0.8,
    beta_schedule_type="linear",
    seed=100,
    reverse_mc_num=64,     # sdac.py 中硬编码的蒙特卡洛扩展倍数
)


def mish(x: jax.Array) -> jax.Array:
    """与 scripts/train_mujoco.py 中 sdac 分支使用的激活函数一致。"""
    return x * jnp.tanh(jax.nn.softplus(x))


def build_algorithm(cfg: Dict[str, Any]):
    """构建与真实训练相同结构的 SDAC 算法实例。"""
    key = jax.random.key(cfg["seed"])
    hidden_sizes = [cfg["hidden_dim"]] * cfg["hidden_num"]
    diffusion_hidden_sizes = [cfg["diffusion_hidden_dim"]] * cfg["hidden_num"]

    agent, params = create_diffv2_net(
        key,
        cfg["obs_dim"],
        cfg["act_dim"],
        hidden_sizes,
        diffusion_hidden_sizes,
        mish,
        num_timesteps=cfg["diffusion_steps"],
        num_particles=cfg["num_particles"],
        noise_scale=cfg["noise_scale"],
        beta_schedule_scale=cfg["beta_schedule_scale"],
    )
    algorithm = SDAC(
        agent,
        params,
        lr=cfg["lr"],
        alpha_lr=cfg["alpha_lr"],
        delay_alpha_update=cfg["delay_alpha_update"],
        lr_schedule_end=cfg["lr_schedule_end"],
    )
    return agent, algorithm


def make_synthetic_batch(cfg: Dict[str, Any]) -> Experience:
    """构造合成 batch。维度与 HalfCheetah-v4 + batch_size=256 一致。"""
    key = jax.random.key(cfg["seed"] + 1)
    k_obs, k_next_obs, k_act, k_rew, k_done = jax.random.split(key, 5)
    b, obs_dim, act_dim = cfg["batch_size"], cfg["obs_dim"], cfg["act_dim"]
    return Experience(
        obs=jax.random.normal(k_obs, (b, obs_dim), dtype=jnp.float32),
        action=jax.random.uniform(k_act, (b, act_dim), minval=-1.0, maxval=1.0, dtype=jnp.float32),
        reward=jax.random.normal(k_rew, (b,), dtype=jnp.float32),
        done=jax.random.bernoulli(k_done, 0.01, (b,)),
        next_obs=jax.random.normal(k_next_obs, (b, obs_dim), dtype=jnp.float32),
    )


# ---------------------------------------------------------------------------
# 计时辅助
# ---------------------------------------------------------------------------
def bench(
    name: str,
    fn: Callable[[], Any],
    registry: TimerRegistry,
    num_iters: int,
    warmup: int = 3,
) -> Any:
    """
    计时一个阶段。

    首次调用触发 JIT 编译，其耗时单独记为 <name>__compile，不计入稳态统计。
    随后 warmup 若干次，再循环 num_iters 次逐次计时（每次 block_until_ready）。
    返回最后一次的输出，供下游阶段作为输入使用。
    """
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn())
    compile_time = time.perf_counter() - t0
    registry.record(f"{name}__compile", compile_time)

    for _ in range(warmup):
        out = jax.block_until_ready(fn())

    for _ in range(num_iters):
        with Timer(name, registry=registry):
            out = jax.block_until_ready(fn())

    stats = registry.get_stats()[name]
    print(
        f"  {name:34s} {stats['average_duration'] * 1e3:9.4f} ms/次"
        f"   (编译 {compile_time:6.2f}s, {stats['count']} 次)"
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="SDAC update 阶段延迟细分微基准")
    parser.add_argument("--num-iters", type=int, default=200, help="每个阶段的稳态计时次数")
    parser.add_argument("--output-dir", type=str, default=None, help="结果输出目录")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖 batch_size")
    args = parser.parse_args()

    cfg = dict(CONFIG)
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "results" / "focus_on_update_step"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("SDAC update 阶段延迟细分微基准")
    print("=" * 78)
    print(f"JAX 后端 : {jax.default_backend()}")
    print(f"设备     : {jax.devices()}")
    print(f"batch    : {cfg['batch_size']}, 去噪步数: {cfg['diffusion_steps']}, "
          f"粒子数: {cfg['num_particles']}, MC倍数: {cfg['reverse_mc_num']}")
    print(f"计时次数 : {args.num_iters}")
    print()

    registry = TimerRegistry()
    registry.reset()

    agent, algorithm = build_algorithm(cfg)
    data = make_synthetic_batch(cfg)
    obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done

    state = algorithm.state
    (q1_params, q2_params, target_q1_params, target_q2_params,
     policy_params, target_policy_params, log_alpha) = state.params
    q1_opt_state, q2_opt_state, policy_opt_state, log_alpha_opt_state = state.opt_state

    mc_num = cfg["reverse_mc_num"]
    reward_scaled = reward * algorithm.reward_scale
    key = jax.random.key(cfg["seed"] + 2)
    (next_eval_key, new_eval_key, diffusion_time_key,
     diffusion_noise_key, update_key) = jax.random.split(key, 5)
    diff_key1, diff_key2 = jax.random.split(diffusion_noise_key, 2)
    policy_bundle = (policy_params, log_alpha, q1_params, q2_params)

    # ------------------------------------------------------------------
    # 参考基线：dispatch 开销地板
    # ------------------------------------------------------------------
    print("[0] 参考基线")
    scalar = jnp.float32(1.0)
    noop = jax.jit(lambda x: x + 1.0)
    bench("baseline_dispatch_overhead", lambda: noop(scalar), registry, args.num_iters)
    print()

    # ------------------------------------------------------------------
    # 阶段 1：next_action（sdac.py:113）
    # ------------------------------------------------------------------
    print("[1] 目标动作采样 get_action(next_obs)  —— sdac.py:113")
    f_next_action = jax.jit(lambda k, p, o: agent.get_action(k, p, o))
    next_action = bench(
        "stage1_next_action_get_action",
        lambda: f_next_action(next_eval_key, policy_bundle, next_obs),
        registry, args.num_iters,
    )

    # get_action 内窥：仅 32 粒子 vmap 的 20 步去噪
    def p_sample_particles(k, pp, o, n_particles):
        def sample(kk):
            def model_fn(t, x):
                return agent.policy(pp, o, x, t)
            return agent.diffusion.p_sample(kk, model_fn, (o.shape[0], cfg["act_dim"]))
        keys = jax.random.split(k, n_particles)
        return jax.vmap(sample)(keys)

    f_psample_n = jax.jit(lambda k, pp, o: p_sample_particles(k, pp, o, cfg["num_particles"]))
    acts_particles = bench(
        "stage1a_p_sample_32particles",
        lambda: f_psample_n(next_eval_key, policy_params, next_obs),
        registry, args.num_iters,
    )

    f_psample_1 = jax.jit(lambda k, pp, o: p_sample_particles(k, pp, o, 1))
    bench(
        "stage1b_p_sample_1particle",
        lambda: f_psample_1(next_eval_key, policy_params, next_obs),
        registry, args.num_iters,
    )

    def particle_q_select(q1p, q2p, o, acts):
        def q_of(a):
            return jnp.minimum(agent.q(q1p, o, a), agent.q(q2p, o, a))
        qs = jax.vmap(q_of)(acts)
        best = jnp.argmax(qs, axis=0, keepdims=True)
        return jnp.take_along_axis(acts, best[..., None], axis=0).squeeze(axis=0)

    f_particle_q = jax.jit(particle_q_select)
    bench(
        "stage1c_particle_q_select",
        lambda: f_particle_q(q1_params, q2_params, next_obs, acts_particles),
        registry, args.num_iters,
    )
    print()

    # ------------------------------------------------------------------
    # 阶段 2：目标 Q 计算（sdac.py:114-117）
    # ------------------------------------------------------------------
    print("[2] 目标 Q 计算  —— sdac.py:114-117")

    def target_q_fn(tq1p, tq2p, no, na, r, d):
        q1_target = agent.q(tq1p, no, na)
        q2_target = agent.q(tq2p, no, na)
        q_target = jnp.minimum(q1_target, q2_target)
        return r + (1 - d) * algorithm.gamma * q_target

    f_target_q = jax.jit(target_q_fn)
    q_backup = bench(
        "stage2_target_q_backup",
        lambda: f_target_q(target_q1_params, target_q2_params, next_obs, next_action, reward_scaled, done),
        registry, args.num_iters,
    )
    print()

    # ------------------------------------------------------------------
    # 阶段 3：Critic 梯度（sdac.py:119-125）
    # ------------------------------------------------------------------
    print("[3] Critic 前向+反向 x2  —— sdac.py:119-125")

    def critic_grads_fn(q1p, q2p, o, a, qb):
        def q_loss_fn(q_params):
            q = agent.q(q_params, o, a)
            return jnp.mean((q - qb) ** 2), q
        (q1_loss, _), q1_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q1p)
        (q2_loss, _), q2_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q2p)
        return q1_grads, q2_grads, q1_loss, q2_loss

    f_critic = jax.jit(critic_grads_fn)
    q1_grads, q2_grads, _, _ = bench(
        "stage3_critic_value_and_grad",
        lambda: f_critic(q1_params, q2_params, obs, action, q_backup),
        registry, args.num_iters,
    )
    print()

    # ------------------------------------------------------------------
    # 阶段 4：new_action（sdac.py:130）
    # ------------------------------------------------------------------
    print("[4] 当前动作采样 get_action(obs)  —— sdac.py:130")
    new_action = bench(
        "stage4_new_action_get_action",
        lambda: f_next_action(new_eval_key, policy_bundle, obs),
        registry, args.num_iters,
    )
    print()

    # ------------------------------------------------------------------
    # 阶段 5：单步加噪 + 64 倍 MC 扩展（sdac.py:131-144）
    # ------------------------------------------------------------------
    print("[5] q_sample 单步加噪 + 64倍MC扩展  —— sdac.py:131-144")

    def q_sample_expand_fn(kt, k1, na, o, a):
        t = jax.random.randint(kt, (o.shape[0],), 0, agent.num_timesteps)
        noise1 = jax.random.normal(k1, a.shape)
        tilde_at = jax.vmap(agent.diffusion.q_sample)(t, na, noise1)
        tilde_at = jnp.repeat(tilde_at, mc_num, axis=0)
        t_wide = jnp.repeat(t, mc_num, axis=0)
        wide_obs = jnp.repeat(o, mc_num, axis=0)
        return tilde_at, t_wide, wide_obs

    f_expand = jax.jit(q_sample_expand_fn)
    tilde_at, t_wide, wide_obs = bench(
        "stage5_q_sample_and_mc_expand",
        lambda: f_expand(diffusion_time_key, diff_key1, new_action, obs, action),
        registry, args.num_iters,
    )
    print()

    # ------------------------------------------------------------------
    # 阶段 6：Policy 损失 + 梯度（sdac.py:146-170），有效 batch = 256*64
    # ------------------------------------------------------------------
    print(f"[6] Policy 前向+反向 (有效batch={cfg['batch_size'] * mc_num})  —— sdac.py:146-170")

    def policy_loss_builder(q1p, q2p, la, w_obs, t_w, tat, k2, act_shape):
        def get_min_q(s, a):
            return jnp.minimum(agent.q(q1p, s, a), agent.q(q2p, s, a))

        def policy_loss_fn(pp):
            def denoiser(t, x):
                return agent.policy(pp, w_obs, x, t)
            noise2 = jax.random.normal(k2, (act_shape[0] * mc_num, act_shape[1]))
            recon = agent.diffusion.get_recon(t_w, tat, noise2).clip(-1, 1)
            q_min = get_min_q(w_obs, recon) * 5.0 / jnp.exp(la)
            q_reshape = q_min.reshape((-1, mc_num))
            Z = jax.nn.logsumexp(q_reshape, axis=1, keepdims=True)
            q_weights = jnp.exp(q_reshape - Z).flatten()
            loss = agent.diffusion.reverse_samping_weighted_p_loss(
                noise2, q_weights, denoiser, t_w, tat
            )
            return loss, q_weights
        return policy_loss_fn

    act_shape = tuple(action.shape)

    def policy_grad_fn(pp, q1p, q2p, la, w_obs, t_w, tat):
        loss_fn = policy_loss_builder(q1p, q2p, la, w_obs, t_w, tat, diff_key2, act_shape)
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(pp)
        return grads, loss, aux

    f_policy = jax.jit(policy_grad_fn)
    policy_grads, _, _ = bench(
        "stage6_policy_value_and_grad",
        lambda: f_policy(policy_params, q1_params, q2_params, log_alpha, wide_obs, t_wide, tilde_at),
        registry, args.num_iters,
    )

    # 阶段 6 内窥：仅前向（不含反向）
    def policy_forward_only(pp, q1p, q2p, la, w_obs, t_w, tat):
        loss_fn = policy_loss_builder(q1p, q2p, la, w_obs, t_w, tat, diff_key2, act_shape)
        return loss_fn(pp)

    f_policy_fwd = jax.jit(policy_forward_only)
    bench(
        "stage6a_policy_forward_only",
        lambda: f_policy_fwd(policy_params, q1_params, q2_params, log_alpha, wide_obs, t_wide, tilde_at),
        registry, args.num_iters,
    )

    # 阶段 6 内窥：仅 Q 网络在扩展 batch 上的前向（权重计算的主要成分）
    def wide_q_forward(q1p, q2p, w_obs, tat):
        return jnp.minimum(agent.q(q1p, w_obs, tat), agent.q(q2p, w_obs, tat))

    f_wide_q = jax.jit(wide_q_forward)
    bench(
        "stage6b_wide_q_forward_only",
        lambda: f_wide_q(q1_params, q2_params, wide_obs, tilde_at),
        registry, args.num_iters,
    )

    # 阶段 6 内窥：仅去噪网络在扩展 batch 上的一次前向（梯度路径上唯一的去噪前向）
    def wide_denoiser_forward(pp, w_obs, t_w, tat):
        return agent.policy(pp, w_obs, tat, t_w)

    f_wide_denoise = jax.jit(wide_denoiser_forward)
    bench(
        "stage6c_wide_denoiser_forward_only",
        lambda: f_wide_denoise(policy_params, wide_obs, t_wide, tilde_at),
        registry, args.num_iters,
    )
    print()

    # ------------------------------------------------------------------
    # 阶段 7：优化器更新 + 目标网络软更新（sdac.py:172-216）
    # ------------------------------------------------------------------
    print("[7] 优化器更新 + 目标网络软更新  —— sdac.py:172-216")

    def param_updates_fn(step, q1p, q2p, tq1p, tq2p, pp, tpp, la,
                         q1os, q2os, pos, laos, q1g, q2g, pg):
        def log_alpha_loss_fn(log_alpha_):
            approx_entropy = 0.5 * agent.act_dim * jnp.log(
                2 * jnp.pi * jnp.exp(1) * (0.1 * jnp.exp(log_alpha_)) ** 2
            )
            return -1 * log_alpha_ * (-1 * jax.lax.stop_gradient(approx_entropy) + agent.target_entropy)

        def param_update(optim, params, grads, opt_state):
            update, new_opt_state = optim.update(grads, opt_state)
            return optax.apply_updates(params, update), new_opt_state

        def delay_param_update(optim, params, grads, opt_state):
            return jax.lax.cond(
                step % algorithm.delay_update == 0,
                lambda p, s: param_update(optim, p, grads, s),
                lambda p, s: (p, s),
                params, opt_state,
            )

        def delay_alpha_param_update(optim, params, opt_state):
            return jax.lax.cond(
                step % algorithm.delay_alpha_update == 0,
                lambda p, s: param_update(optim, p, jax.grad(log_alpha_loss_fn)(p), s),
                lambda p, s: (p, s),
                params, opt_state,
            )

        def delay_target_update(params, target_params, tau):
            return jax.lax.cond(
                step % algorithm.delay_update == 0,
                lambda tp: optax.incremental_update(params, tp, tau),
                lambda tp: tp,
                target_params,
            )

        q1p, q1os = param_update(algorithm.optim, q1p, q1g, q1os)
        q2p, q2os = param_update(algorithm.optim, q2p, q2g, q2os)
        pp, pos = delay_param_update(algorithm.policy_optim, pp, pg, pos)
        la, laos = delay_alpha_param_update(algorithm.alpha_optim, la, laos)
        tq1p = delay_target_update(q1p, tq1p, algorithm.tau)
        tq2p = delay_target_update(q2p, tq2p, algorithm.tau)
        tpp = delay_target_update(pp, tpp, algorithm.tau)
        return q1p, q2p, tq1p, tq2p, pp, tpp, la, q1os, q2os, pos, laos

    f_param_updates = jax.jit(param_updates_fn)
    common_args = (q1_params, q2_params, target_q1_params, target_q2_params,
                   policy_params, target_policy_params, log_alpha,
                   q1_opt_state, q2_opt_state, policy_opt_state, log_alpha_opt_state,
                   q1_grads, q2_grads, policy_grads)

    # step=0：policy / target / alpha 全部更新（最重的分支）
    bench(
        "stage7_param_updates_step0_all",
        lambda: f_param_updates(jnp.int32(0), *common_args),
        registry, args.num_iters,
    )
    # step=1：仅 critic 更新（policy / target / alpha 跳过）
    bench(
        "stage7_param_updates_step1_criticonly",
        lambda: f_param_updates(jnp.int32(1), *common_args),
        registry, args.num_iters,
    )
    print()

    # ------------------------------------------------------------------
    # 对照组：完整融合版
    # ------------------------------------------------------------------
    print("[8] 对照：完整融合的 update")
    bench(
        "fused_stateless_update_step0",
        lambda: algorithm._update(update_key, state, data),
        registry, args.num_iters,
    )

    state_step1 = state._replace(step=jnp.int32(1))
    bench(
        "fused_stateless_update_step1",
        lambda: algorithm._update(update_key, state_step1, data),
        registry, args.num_iters,
    )

    # trainer 真实调用路径：algorithm.update 额外做 float(v) 的 device->host 同步
    saved_state = algorithm.state

    def full_update_api():
        algorithm.state = saved_state
        return algorithm.update(update_key, data)

    bench("api_algorithm_update_with_host_sync", full_update_api, registry, args.num_iters)
    algorithm.state = saved_state
    print()

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"profile_update_step_{timestamp}.json"
    registry.export_json(str(out_path))

    with open(out_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["metadata"] = {
        "timestamp": timestamp,
        "config": cfg,
        "num_iters": args.num_iters,
        "jax_backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "python_version": platform.python_version(),
        "note": (
            "各 stage* 为独立 jit 编译的镜像子步骤；fused_* 为真实的整体融合 update；"
            "api_* 为 trainer 实际调用路径（含 float() 主机同步）。"
            "<name>__compile 记录首次调用的编译耗时，不属于稳态延迟。"
        ),
    }
    # records 数量大且对分析无用，导出时裁剪以控制文件体积
    payload.pop("records", None)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"结果已写入: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
