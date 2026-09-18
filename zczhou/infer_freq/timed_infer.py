"""带逐动作计时的单次推理：加载已训练好的权重，跑完整一个 episode，
并统计每次动作生成耗时与整步耗时，用于计算动作频率。

派生自 zczhou/scripts/single_infer.py，仅替换 rollout 部分并新增结果落盘。
JAX 在 GPU 上异步发射 kernel，必须用 jax.block_until_ready 同步后再停表，
否则测到的只是 host 发射开销。

首步 policy_fn 调用触发 XLA 编译，耗时远超稳态，单独记为 warmup，
不计入稳态频率统计。

用法与 single_infer.py 一致，多一个 --out 指定结果 json 输出路径。
"""

import argparse
import json
import pickle
import platform
import re
import time
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


def rollout_timed(env, policy_fn, policy_params, max_steps: int):
    """逐步计时的 rollout。

    返回:
        ep_len: episode 步数
        ep_ret: episode 回报
        warmup_act_s: 首步 policy_fn 耗时（含 XLA 编译），秒
        act_lat_s: 稳态每步 policy_fn 耗时列表（含 block_until_ready），秒
        step_lat_s: 稳态每步 policy_fn + env.step 耗时列表，秒
    """
    obs, _ = env.reset()
    ep_len = 0
    ep_ret = 0.0
    warmup_act_s: float | None = None
    act_lat_s: list[float] = []
    step_lat_s: list[float] = []

    while True:
        t0 = time.perf_counter()
        act = policy_fn(policy_params, obs)
        # 关键：等 GPU kernel 真正执行完再停表
        act = jax.block_until_ready(act)
        t1 = time.perf_counter()
        act = np.asarray(act)
        obs, reward, terminated, truncated, _ = env.step(act)
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
    parser.add_argument("--video_dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--video_fps", type=int, default=0, help="0 表示取环境的 render_fps")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None,
                        help="结果 json 输出路径，默认 infer_freq/results/latencies_<timestamp>.json")
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

    log_dir: Path = args.log_dir
    if not log_dir.is_dir():
        raise FileNotFoundError(f"Log dir not found: {log_dir}")

    policy_path = resolve_policy_path(log_dir, args.policy)
    env_name = resolve_env_name(log_dir, args.env)

    master_rng = np.random.default_rng(args.seed)
    env_seed, env_action_seed = map(int, master_rng.integers(0, 2**32 - 1, 2))
    render_mode = "rgb_array" if args.video else ("human" if args.render else None)
    env, obs_dim, act_dim = make_env(env_name, env_seed, env_action_seed, render_mode, args.camera)

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

    ep_len, ep_ret, warmup_act_s, act_lat_s, step_lat_s = rollout_timed(
        env, policy_fn, policy_params, args.max_steps
    )
    env.close()

    # 落盘原始数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path: Path = args.out or (DEFAULT_OUT_DIR / f"latencies_{timestamp}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "timestamp": timestamp,
            "checkpoint": str(policy_path),
            "env": env_name,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "seed": args.seed,
            "max_steps": args.max_steps,
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

    # 控制台摘要（统计细节交给 analyze.py）
    act_arr = np.asarray(act_lat_s)
    step_arr = np.asarray(step_lat_s)
    print("=" * 60)
    print(f"checkpoint : {policy_path}")
    print(f"env        : {env_name}  (obs_dim={obs_dim}, act_dim={act_dim})")
    print(f"seed       : {args.seed}")
    print(f"backend    : {jax.default_backend()}  devices={jax.devices()}")
    print(f"ep_len     : {ep_len}")
    print(f"ep_ret     : {ep_ret:.2f}")
    print(f"warmup     : {warmup_act_s * 1e3:.3f} ms  (首步含 XLA 编译)")
    print(f"act  mean  : {act_arr.mean() * 1e3:.3f} ms  "
          f"(n={len(act_arr)}, p50={np.percentile(act_arr, 50) * 1e3:.3f} ms)")
    print(f"step mean  : {step_arr.mean() * 1e3:.3f} ms")
    print(f"act  freq  : {1.0 / act_arr.mean():.2f} Hz (mean)  "
          f"{1.0 / np.percentile(act_arr, 50):.2f} Hz (p50)")
    print(f"step freq  : {1.0 / step_arr.mean():.2f} Hz (mean)")
    print(f"saved      : {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
