"""单次推理：加载已训练好的权重，跑完整一个 episode。

默认复用训练时导出的确定性策略计算图（deterministic.pkl）与某个 checkpoint 的
参数（policy-*.pkl），不需要重建网络，也不依赖训练超参。

加 --quant 时改走源码重建的网络（deterministic.pkl 是固化 jaxpr，无法插桩），
把 MLP 层换成 INT8 实现。两条路径已验证输出逐位一致，所以量化前后的差异
可以干净归因到量化本身。
"""

import argparse
import pickle
import re
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


def rollout(env, policy_fn, policy_params, max_steps: int, collect_frames: bool):
    obs, _ = env.reset()
    ep_len = 0
    ep_ret = 0.0
    frames = []
    if collect_frames:
        frames.append(env.render())
    while True:
        act = np.asarray(policy_fn(policy_params, obs))
        obs, reward, terminated, truncated, _ = env.step(act)
        ep_len += 1
        ep_ret += float(reward)
        if collect_frames:
            frames.append(env.render())
        if terminated or truncated:
            break
        if max_steps > 0 and ep_len >= max_steps:
            break
    return ep_len, ep_ret, frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--policy", type=str, default=None)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--video_dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--video_fps", type=int, default=0, help="0 表示取环境的 render_fps")
    parser.add_argument("--camera", type=int, default=None)
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
                        help="激活分组量化的组大小，每组独立 scale；0 关闭。"
                             "默认 8，实测与 --no_skip_first 组合下回报掉幅约 4%%")
    parser.add_argument("--skip_first_layer", action="store_true", default=True,
                        help="首层（直接吃未归一化 obs）保持 FP32，默认开启")
    parser.add_argument("--no_skip_first_layer", dest="skip_first_layer",
                        action="store_false",
                        help="首层也量化（回报掉幅会明显变大，用于对照）")
    parser.add_argument("--fp32_modules", type=str, default="",
                        help="逗号分隔的模块名，这些层保持 FP32，如 q_net/linear_3")
    parser.add_argument("--from_source", action="store_true", default=False,
                        help="不量化但走源码重建路径，用于自证与固化图等价")
    args = parser.parse_args()

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

    ep_len, ep_ret, frames = rollout(env, policy_fn, policy_params, args.max_steps, args.video)
    fps = args.video_fps or env.metadata.get("render_fps", 30)
    env.close()

    video_path = None
    if args.video:
        video_dir: Path = args.video_dir
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"{env_name}_{policy_path.stem}_s{args.seed}.mp4"
        imageio.mimsave(video_path, frames, fps=fps, macro_block_size=1)

    print("=" * 60)
    print(f"checkpoint : {policy_path}")
    print(f"env        : {env_name}  (obs_dim={obs_dim}, act_dim={act_dim})")
    print(f"seed       : {args.seed}")
    if quant_config is not None:
        weight_gran = "per-channel" if quant_config.weight_per_channel else "per-tensor"
        act_gran = "symmetric" if quant_config.act_symmetric else "asymmetric"
        if quant_config.use_grouped_act:
            act_scope = f"group{quant_config.act_group_size}"
        elif quant_config.act_per_token:
            act_scope = "per-token"
        else:
            act_scope = "per-tensor"
        print(f"quant      : mode={quant_config.mode} target={args.quant_target} "
              f"w={quant_config.weight_bits}bit/{weight_gran} "
              f"a={quant_config.act_bits}bit/{act_gran}/{act_scope}")
        skipped = ["first-layer"] if quant_config.skip_first_layer else []
        skipped += list(quant_config.fp32_modules)
        print(f"fp32 keep  : {', '.join(skipped) if skipped else '(none)'}")
    elif args.from_source:
        print("quant      : off (from_source, FP32)")
    print(f"ep_len     : {ep_len}")
    print(f"ep_ret     : {ep_ret:.2f}")
    if video_path is not None:
        print(f"video      : {video_path}")
        print(f"frames     : {len(frames)} @ {fps} fps")
    print("=" * 60)
