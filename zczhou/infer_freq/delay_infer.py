"""动作延迟对任务回报影响实验：带动作延迟队列的单次推理。

模拟真实控制中的观测-动作延迟：策略在 t 时刻基于观测 obs_t 生成的动作不立即执行，
而是入队，环境每步仍按队列中 N 步之前生成的动作推进，环境持续演化而动作生效滞后
N 步。前 N 步队列未满时用零动作填充（延迟期间无控制输入）。N=0 时退化为无延迟
baseline。

派生自 zczhou/infer_freq/timed_infer.py，仅替换 rollout 核心并新增 --delay_steps。

用法：
    python delay_infer.py --delay_steps 0   # 无延迟 baseline
    python delay_infer.py --delay_steps 1   # 延迟 1 个控制周期（HalfCheetah 为 50ms 仿真时间）
"""

import argparse
import json
import pickle
import platform
import re
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import jax
import yaml
from gymnasium import make

from relax.env import RelaxWrapper, create_env
from relax.utils.fs import PROJECT_ROOT
from relax.utils.persistence import PersistFunction
from zczhou.quant.int_infer import QuantConfig
from zczhou.quant.int_infer.net import QUANT_TARGETS, build_net, make_policy_fn

DEFAULT_LOG_DIR = PROJECT_ROOT / "logs" / "HalfCheetah-v4" / "sdac_2026-09-01_05-18-24_s100_test_use_atp1"
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "videos"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "results"


def latest_policy_path(log_dir: Path) -> Path:
    """挑 sample_step 最大的 policy-{sample_step}-{update_step}.pkl。"""
    pattern = re.compile(r"policy-(\d+)-(\d+)\.pkl$")
    candidates = []
    for path in log_dir.glob("policy-*.pkl"):
        matched = pattern.search(path.name)
        if matched:
            candidates.append((int(matched.group(1)), int(matched.group(2)), path))
    if not candidates:
        raise FileNotFoundError(f"No policy-*.pkl found in {log_dir}")
    return max(candidates)[2]


def resolve_policy_path(log_dir: Path, policy: str | None) -> Path:
    if policy is None:
        return latest_policy_path(log_dir)
    path = Path(policy)
    if not path.is_absolute():
        path = log_dir / path
    if not path.is_file():
        raise FileNotFoundError(f"Policy file not found: {path}")
    return path


def resolve_env_name(log_dir: Path, env: str | None) -> str:
    if env is not None:
        return env
    config_path = log_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"--env not given and {config_path} does not exist")
    with open(config_path) as f:
        return yaml.safe_load(f)["env"]


def make_env(name: str, seed: int, action_seed: int, render_mode: str | None, camera: int | None):
    if render_mode is None:
        return create_env(name, seed, action_seed)
    kwargs = {"render_mode": render_mode}
    if camera is not None:
        kwargs["camera_id"] = camera
    env = make(name, **kwargs)
    env.reset(seed=seed)
    env = RelaxWrapper(env, action_seed)
    return env, env.obs_dim, env.act_dim


def query_ctrl_dt(env) -> float | None:
    """查询环境单步仿真时间 frame_skip * timestep（秒）。查不到返回 None。"""
    try:
        dt = env.unwrapped.frame_skip * env.unwrapped.model.opt.timestep
        return float(dt)
    except Exception:
        return None


def rollout_with_delay(env, policy_fn, policy_params, max_steps: int, delay_steps: int):
    """带动作延迟队列的 rollout。

    策略每步生成 act_new 入队尾，同时从队首取出 delay_steps 步之前的动作送 env.step。
    队列预填 delay_steps 个零动作，保证前 delay_steps 步环境用零动作推进。

    返回:
        ep_len, ep_ret, warmup_act_s, act_lat_s, step_lat_s
        （计时口径与 timed_infer 一致，便于横向对比）
    """
    obs, _ = env.reset()
    act_dim = env.act_dim
    zero_act = np.zeros(act_dim, dtype=np.float32)
    # 队列预填 delay_steps 个零动作。不用 maxlen（append 会挤掉队首使延迟失效），
    # 改为每步 append 新动作后 popleft 取队首，队列长度恒为 delay_steps+1，
    # 取出的恰好是 delay_steps 步之前的动作。
    queue: deque | None = None
    if delay_steps > 0:
        queue = deque(zero_act.copy() for _ in range(delay_steps))

    ep_len = 0
    ep_ret = 0.0
    warmup_act_s: float | None = None
    act_lat_s: list[float] = []
    step_lat_s: list[float] = []

    while True:
        t0 = time.perf_counter()
        act_new = policy_fn(policy_params, obs)
        act_new = jax.block_until_ready(act_new)
        t1 = time.perf_counter()
        act_new = np.asarray(act_new)

        if queue is not None:
            queue.append(act_new)
            act_exec = queue.popleft()    # 队首：delay_steps 步前的动作（前 N 步为零动作）
        else:
            act_exec = act_new            # N=0 退化为 baseline

        obs, reward, terminated, truncated, _ = env.step(act_exec)
        t2 = time.perf_counter()

        if warmup_act_s is None:
            warmup_act_s = t1 - t0
        else:
            act_lat_s.append(t1 - t0)
            step_lat_s.append(t2 - t0)

        ep_len += 1
        ep_ret += float(reward)
        if terminated or truncated:
            break
        if max_steps > 0 and ep_len >= max_steps:
            break

    assert warmup_act_s is not None, "episode 至少需要 1 步"
    return ep_len, ep_ret, warmup_act_s, act_lat_s, step_lat_s


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--policy", type=str, default=None)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--delay_steps", type=int, default=0,
                        help="动作延迟步数（控制周期数），0 表示无延迟 baseline")
    parser.add_argument("--video_dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--video_fps", type=int, default=0, help="0 表示取环境的 render_fps")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None,
                        help="结果 json 输出路径，默认 infer_freq/results/delay_N<delay>_<timestamp>.json")
    # 一个环境只能有一个 render_mode，所以两者互斥
    render_group = parser.add_mutually_exclusive_group()
    render_group.add_argument("--render", action="store_true", default=False)
    render_group.add_argument("--video", action="store_true", default=False)
    parser.add_argument("--quant", action="store_true", default=False,
                        help="MLP 层走 INT8 推理（权重与激活都量化）")
    parser.add_argument("--quant_mode", type=str, default="int", choices=("int", "fake"),
                        help="int: int8xint8->int32 真整数累加；fake: 量化-反量化后走 FP32，用于排查")
    parser.add_argument("--quant_target", type=str, default="both", choices=QUANT_TARGETS,
                        help="量化哪一支网络，用于消融")
    parser.add_argument("--weight_per_tensor", action="store_true", default=False,
                        help="权重退回 per-tensor 量化（精度更差，用于对照）")
    parser.add_argument("--act_symmetric", action="store_true", default=False,
                        help="激活改用对称量化（mish 输出偏斜，默认非对称更好）")
    parser.add_argument("--act_per_tensor", action="store_true", default=False,
                        help="激活 scale 退回整张量共享（默认 per-token，精度显著更好）")
    parser.add_argument("--act_group_size", type=int, default=8,
                        help="激活分组量化的组大小，每组独立 scale；0 关闭")
    parser.add_argument("--skip_first_layer", action="store_true", default=True,
                        help="首层（直接吃未归一化 obs）保持 FP32，默认开启")
    parser.add_argument("--no_skip_first_layer", dest="skip_first_layer",
                        action="store_false",
                        help="首层也量化（回报掉幅会明显变大，用于对照）")
    parser.add_argument("--fp32_modules", type=str, default="",
                        help="逗号分隔的模块名，这些层保持 FP32，如 q_net/linear_3")
    parser.add_argument("--from_source", action="store_true", default=False,
                        help="不量化但走源码重建路径，用于自证与固化图等价")
    return parser


def main():
    args = build_parser().parse_args()
    if args.delay_steps < 0:
        raise ValueError("--delay_steps 必须 >= 0")

    log_dir: Path = args.log_dir
    if not log_dir.is_dir():
        raise FileNotFoundError(f"Log dir not found: {log_dir}")

    policy_path = resolve_policy_path(log_dir, args.policy)
    env_name = resolve_env_name(log_dir, args.env)

    master_rng = np.random.default_rng(args.seed)
    env_seed, env_action_seed = map(int, master_rng.integers(0, 2**32 - 1, 2))
    render_mode = "rgb_array" if args.video else ("human" if args.render else None)
    env, obs_dim, act_dim = make_env(env_name, env_seed, env_action_seed, render_mode, args.camera)
    ctrl_dt = query_ctrl_dt(env)

    quant_config = None
    if args.quant:
        quant_config = QuantConfig(
            mode=args.quant_mode,
            weight_per_channel=not args.weight_per_tensor,
            act_symmetric=args.act_symmetric,
            act_per_token=not args.act_per_tensor,
            skip_first_layer=args.skip_first_layer,
            fp32_modules=tuple(m for m in args.fp32_modules.split(",") if m),
            act_group_size=args.act_group_size,
        )

    if args.quant or args.from_source:
        net = build_net(
            log_dir,
            obs_dim,
            act_dim,
            quant_config=quant_config,
            quant_target=args.quant_target if args.quant else "none",
        )
        policy_fn = make_policy_fn(net)
    else:
        policy = PersistFunction.load(log_dir / "deterministic.pkl")

        @jax.jit
        def policy_fn(policy_params, obs):
            return policy(policy_params, obs).clip(-1, 1)

    with open(policy_path, "rb") as f:
        policy_params = pickle.load(f)

    ep_len, ep_ret, warmup_act_s, act_lat_s, step_lat_s = rollout_with_delay(
        env, policy_fn, policy_params, args.max_steps, args.delay_steps
    )
    env.close()

    # 落盘原始数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out is not None:
        out_path: Path = args.out
    else:
        out_path = DEFAULT_OUT_DIR / f"delay_N{args.delay_steps}_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    delay_sim_ms = args.delay_steps * ctrl_dt * 1e3 if ctrl_dt is not None else None
    payload = {
        "meta": {
            "timestamp": timestamp,
            "checkpoint": str(policy_path),
            "env": env_name,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "delay_steps": args.delay_steps,
            "ctrl_dt_s": ctrl_dt,
            "delay_sim_ms": delay_sim_ms,
            "quant": args.quant,
            "from_source": args.from_source,
            "jax_version": jax.__version__,
            "jax_devices": [str(d) for d in jax.devices()],
            "default_backend": jax.default_backend(),
            "python": platform.python_version(),
            "hostname": platform.node(),
        },
        "episode": {
            "ep_len": ep_len,
            "ep_ret": ep_ret,
        },
        "warmup_act_s": warmup_act_s,
        "act_latencies_s": act_lat_s,
        "step_latencies_s": step_lat_s,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # 控制台摘要
    act_arr = np.asarray(act_lat_s)
    print("=" * 60)
    print(f"checkpoint : {policy_path}")
    print(f"env        : {env_name}  (obs_dim={obs_dim}, act_dim={act_dim})")
    print(f"seed       : {args.seed}")
    print(f"backend    : {jax.default_backend()}  devices={jax.devices()}")
    print(f"delay      : {args.delay_steps} 步"
          + (f"（{delay_sim_ms:.1f} ms 仿真时间 / 滞后 {args.delay_steps} 个控制周期）"
             if delay_sim_ms is not None else ""))
    print(f"ep_len     : {ep_len}")
    print(f"ep_ret     : {ep_ret:.2f}")
    print(f"warmup     : {warmup_act_s * 1e3:.3f} ms  (首步含 XLA 编译)")
    if act_arr.size > 0:
        print(f"act  mean  : {act_arr.mean() * 1e3:.3f} ms  (n={act_arr.size})")
    print(f"saved      : {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
