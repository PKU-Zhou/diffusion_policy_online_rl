"""单次推理：加载已训练好的权重，跑完整一个 episode。

复用训练时导出的确定性策略计算图（deterministic.pkl）与某个 checkpoint 的
参数（policy-*.pkl），不需要重建网络，也不依赖训练超参。
"""

import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import jax
import yaml
from gymnasium import make

from relax.env import RelaxWrapper, create_env
from relax.utils.fs import PROJECT_ROOT
from relax.utils.persistence import PersistFunction

DEFAULT_LOG_DIR = PROJECT_ROOT / "logs" / "HalfCheetah-v4" / "sdac_2026-09-01_05-18-24_s100_test_use_atp1"


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


def make_env(name: str, seed: int, action_seed: int, render: bool):
    if not render:
        return create_env(name, seed, action_seed)
    env = make(name, render_mode="human")
    env.reset(seed=seed)
    env = RelaxWrapper(env, action_seed)
    return env, env.obs_dim, env.act_dim


def rollout(env, policy_fn, policy_params, max_steps: int):
    obs, _ = env.reset()
    ep_len = 0
    ep_ret = 0.0
    while True:
        act = np.asarray(policy_fn(policy_params, obs))
        obs, reward, terminated, truncated, _ = env.step(act)
        ep_len += 1
        ep_ret += float(reward)
        if terminated or truncated:
            break
        if max_steps > 0 and ep_len >= max_steps:
            break
    return ep_len, ep_ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--policy", type=str, default=None)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--render", action="store_true", default=False)
    args = parser.parse_args()

    log_dir: Path = args.log_dir
    if not log_dir.is_dir():
        raise FileNotFoundError(f"Log dir not found: {log_dir}")

    policy_path = resolve_policy_path(log_dir, args.policy)
    env_name = resolve_env_name(log_dir, args.env)

    master_rng = np.random.default_rng(args.seed)
    env_seed, env_action_seed = map(int, master_rng.integers(0, 2**32 - 1, 2))
    env, obs_dim, act_dim = make_env(env_name, env_seed, env_action_seed, args.render)

    policy = PersistFunction.load(log_dir / "deterministic.pkl")

    @jax.jit
    def policy_fn(policy_params, obs):
        return policy(policy_params, obs).clip(-1, 1)

    with open(policy_path, "rb") as f:
        policy_params = pickle.load(f)

    ep_len, ep_ret = rollout(env, policy_fn, policy_params, args.max_steps)
    env.close()

    print("=" * 60)
    print(f"checkpoint : {policy_path}")
    print(f"env        : {env_name}  (obs_dim={obs_dim}, act_dim={act_dim})")
    print(f"seed       : {args.seed}")
    print(f"ep_len     : {ep_len}")
    print(f"ep_ret     : {ep_ret:.2f}")
    print("=" * 60)
