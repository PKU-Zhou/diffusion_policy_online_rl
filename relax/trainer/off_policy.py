from pathlib import Path
import subprocess
import sys
from typing import Callable, Optional, Tuple
import time

import jax
import numpy as np
from gymnasium import Env
from tqdm import tqdm
from tensorboardX import SummaryWriter
from tensorboardX.summary import hparams
import wandb

from relax.algorithm import Algorithm
from relax.buffer import ExperienceBuffer
from relax.env.vector import VectorEnv
from relax.trainer.accumulator import SampleLog, VectorSampleLog, UpdateLog, Interval
from relax.utils.experience import Experience

# Import profiling utilities
try:
    import os
    import inspect
    current_file = inspect.getfile(inspect.currentframe())
    project_root = Path(current_file).parent.parent.parent
    profiling_utils_path = project_root / 'zczhou' / 'profiling' / 'utils'
    if profiling_utils_path.exists():
        sys.path.insert(0, str(profiling_utils_path.parent.parent))
        from zczhou.profiling.utils.timer import Timer, TimerRegistry
        PROFILING_AVAILABLE = True
    else:
        PROFILING_AVAILABLE = False
except Exception:
    PROFILING_AVAILABLE = False


class OffPolicyTrainer:
    def __init__(
        self,
        env: Env,
        algorithm: Algorithm,
        buffer: ExperienceBuffer,
        log_path: Path,
        batch_size: int = 256,
        start_step: int = 1000,
        total_step: int = int(1e6),
        sample_per_iteration: int = 1,
        update_per_iteration: int = 1,
        evaluate_env: Optional[Env] = None,
        evaluate_every: int = 10000,
        evaluate_n_episode: int = 20,
        sample_log_n_episode: int = 10,
        update_log_n_step: int = 1000,
        done_info_keys: Tuple[str, ...] = (),
        save_policy_every: int = 10000,
        save_value: bool = True,
        hparams: Optional[dict] = None,
        policy_pkl_template: str = "policy-{sample_step}-{update_step}.pkl",
        warmup_with: str = "random",  # "policy" or "random"
        enable_profiling: bool = False,
        profiling_output: Optional[Path] = None,
    ):
        self.env = env
        self.algorithm = algorithm
        self.buffer = buffer
        self.batch_size = batch_size
        self.start_step = start_step
        self.total_step = total_step
        self.sample_per_iteration = sample_per_iteration
        self.update_per_iteration = update_per_iteration
        self.log_path = log_path
        self.policy_pkl_template = policy_pkl_template
        self.evaluate_env = evaluate_env
        self.evaluate_every = evaluate_every
        self.evaluate_n_episode = evaluate_n_episode
        self.sample_log_n_episode = sample_log_n_episode
        self.update_log_n_step = update_log_n_step
        self.done_info_keys = done_info_keys
        self.save_policy_every = save_policy_every
        self.hparams = hparams
        self.warmup_with = warmup_with
        self.save_value = save_value
        
        # Initialize profiling
        self.enable_profiling = enable_profiling and PROFILING_AVAILABLE
        self.profiling_output = profiling_output
        if self.enable_profiling:
            self.timer_registry = TimerRegistry()
        else:
            self.timer_registry = None
        # TODO: make EpisodeLog and Experience configurable
        # TODO: re-add done_info_keys support
        # TODO: re-add evaluation support

        if isinstance(self.env.unwrapped, VectorEnv):
            self.is_vec = True
            self.sample_log = VectorSampleLog(self.env.unwrapped.num_envs)
        else:
            self.is_vec = False
            self.sample_log = SampleLog()
        self.update_log = UpdateLog()
        self.last_metrics = {}
        # The following two depends on sample_step, which may not update by one only
        self.sample_log_interval = Interval(self.sample_log_n_episode)
        self.save_policy_interval = Interval(self.save_policy_every)
        # self.eval_interval = Interval()
        wandb.init(project="diffusion_online_rl",
                   name=log_path.name,
                   dir=log_path,
                   group=env.spec.id)

    def setup(self, dummy_data: Experience):
        if self.enable_profiling:
            with Timer('setup_total', self.timer_registry):
                self._setup_impl(dummy_data)
        else:
            self._setup_impl(dummy_data)
    
    def _setup_impl(self, dummy_data: Experience):
        if self.enable_profiling:
            with Timer('algorithm_warmup', self.timer_registry):
                self.algorithm.warmup(dummy_data)
        else:
            self.algorithm.warmup(dummy_data)

        # Setup logger
        if self.enable_profiling:
            with Timer('logger_init', self.timer_registry):
                self.logger = SummaryWriter(str(self.log_path))
                self.progress = tqdm(total=self.total_step, desc="Sample Step", disable=None, dynamic_ncols=True)
        else:
            self.logger = SummaryWriter(str(self.log_path))
            self.progress = tqdm(total=self.total_step, desc="Sample Step", disable=None, dynamic_ncols=True)

        if self.enable_profiling:
            with Timer('save_network_structure', self.timer_registry):
                self.algorithm.save_policy_structure(self.log_path, dummy_data.obs[0])
                if self.save_value:
                    self.algorithm.save_q_structure(self.log_path, dummy_obs=dummy_data.obs[0], dummy_action=dummy_data.action[0])
        else:
            self.algorithm.save_policy_structure(self.log_path, dummy_data.obs[0])
            if self.save_value:
                self.algorithm.save_q_structure(self.log_path, dummy_obs=dummy_data.obs[0], dummy_action=dummy_data.action[0])
        
        if self.enable_profiling:
            with Timer('evaluator_init', self.timer_registry):
                self.evaluator = subprocess.Popen(
                    [
                        sys.executable,
                        "-m", "relax.trainer.evaluator",
                        str(self.log_path),
                        "--env", self.env.spec.id,
                        "--num_episodes", str(self.evaluate_n_episode),
                        "--seed", str(0),
                    ],
                    stdin=subprocess.PIPE,
                    bufsize=0,
                )
        else:
            self.evaluator = subprocess.Popen(
                [
                    sys.executable,
                    "-m", "relax.trainer.evaluator",
                    str(self.log_path),
                    "--env", self.env.spec.id,
                    "--num_episodes", str(self.evaluate_n_episode),
                    "--seed", str(0),
                ],
                stdin=subprocess.PIPE,
                bufsize=0,
            )

    def warmup(self, key: jax.Array, obs: np.ndarray):
        if self.enable_profiling:
            with Timer('warmup_total', self.timer_registry):
                return self._warmup_impl(key, obs)
        else:
            return self._warmup_impl(key, obs)
    
    def _warmup_impl(self, key: jax.Array, obs: np.ndarray):
        step = 0
        key_fn = jax.jit(lambda step: jax.random.fold_in(key, step))
        while len(self.buffer) < self.start_step:
            step += 1
            if self.warmup_with == "random":
                action = self.env.action_space.sample()
            elif self.warmup_with == "policy":
                action = self.algorithm.get_action(key_fn(step), obs)
            else:
                raise ValueError(f"Invalid warmup_with {self.warmup_with}!")
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            experience = Experience.create(obs, action, reward, terminated, truncated, next_obs, info)
            if self.is_vec:
                self.buffer.add_batch(experience)
            else:
                self.buffer.add(experience)

            if np.any(terminated) or np.any(truncated):
                obs, _ = self.env.reset()
            else:
                obs = next_obs
        return obs

    def sample(self, sample_key: jax.Array, obs: np.ndarray):
        if self.enable_profiling:
            start_time = time.perf_counter()
        
        sl = self.sample_log

        action = self.algorithm.get_action(sample_key, obs)
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        experience = Experience.create(obs, action, reward, terminated, truncated, next_obs, info)
        if self.is_vec:
            self.buffer.add_batch(experience)
        else:
            self.buffer.add(experience)

        any_done = sl.add(reward, terminated, truncated, info)

        if any_done:
            if self.sample_log_interval.check(sl.sample_episode):
                sl.log(self.add_scalar)
            self.progress.update(sl.sample_step - self.progress.n)

            obs, _ = self.env.reset()
        else:
            obs = next_obs

        if self.enable_profiling:
            duration = time.perf_counter() - start_time
            self.timer_registry.record('sample_step', duration, parent='training_loop')
        
        return obs

    def update(self, update_key: jax.Array):
        if self.enable_profiling:
            start_time = time.perf_counter()
        
        ul = self.update_log
        data = self.buffer.sample(self.batch_size)
        info, dist_info = self.algorithm.update(update_key, data)

        ul.add(info)

        if ul.update_step % self.update_log_n_step == 0:
            self.add_hist(dist_info, ul.update_step * 5)
            ul.log(self.add_scalar)
        
        if self.enable_profiling:
            duration = time.perf_counter() - start_time
            self.timer_registry.record('update_step', duration, parent='training_loop')

    def train(self, key: jax.Array):
        if self.enable_profiling:
            with Timer('training_loop', self.timer_registry):
                self._train_impl(key)
        else:
            self._train_impl(key)
    
    def _train_impl(self, key: jax.Array):
        key, warmup_key = jax.random.split(key)

        obs, _ = self.env.reset()
        obs = self.warmup(warmup_key, obs)

        iter_key_fn = create_iter_key_fn(key, self.sample_per_iteration, self.update_per_iteration)
        sl, ul = self.sample_log, self.update_log

        self.progress.unpause()
        while sl.sample_step <= self.total_step:
            sample_keys, update_keys = iter_key_fn(sl.sample_step)

            for i in range(self.sample_per_iteration):
                obs = self.sample(sample_keys[i], obs)

            for i in range(self.update_per_iteration):
                self.update(update_keys[i])

            if self.save_policy_interval.check(sl.sample_step):
                if self.enable_profiling:
                    save_start = time.perf_counter()
                
                policy_pkl_name = self.policy_pkl_template.format(
                    sample_step=sl.sample_step,
                    update_step=ul.update_step,
                )
                self.algorithm.save_policy(self.log_path / policy_pkl_name)
                
                if self.save_value:
                    self.algorithm.save_q(self.log_path / policy_pkl_name.replace('policy', 'value'))
                
                if self.enable_profiling:
                    save_duration = time.perf_counter() - save_start
                    self.timer_registry.record('save_policy', save_duration, parent='training_loop')
                    eval_start = time.perf_counter()

                command = f"{sl.sample_step},{self.log_path / policy_pkl_name}\n"
                self.evaluator.stdin.write(command.encode())
                
                if self.enable_profiling:
                    eval_duration = time.perf_counter() - eval_start
                    self.timer_registry.record('evaluator_communication', eval_duration, parent='training_loop')

    def add_scalar(self, tag: str, value: float, step: int):
        self.last_metrics[tag] = value
        wandb.log({tag: value}, step=step)
        self.logger.add_scalar(tag, value, step)
        self.logger.flush()
        
    def add_hist(self, info_hist, step):
        for tag, value in info_hist.items():
            self.logger.add_histogram(tag, np.array(value), step)
            wandb.log({tag: wandb.Histogram(np.array(value))}, step=step)
        self.logger.flush()

    def run(self, key: jax.Array):
        try:
            self.train(key)
        except KeyboardInterrupt:
            pass
        finally:
            self.finish()

    def finish(self):
        if self.enable_profiling:
            with Timer('finish_total', self.timer_registry):
                self._finish_impl()
        else:
            self._finish_impl()
    
    def _finish_impl(self):
        self.env.close()
        self.algorithm.save(self.log_path / "state.pkl")
        if self.hparams is not None and len(self.last_metrics) > 0:
            exp, ssi, sei = hparams(self.hparams, self.last_metrics)
            self.logger.file_writer.add_summary(exp)
            self.logger.file_writer.add_summary(ssi)
            self.logger.file_writer.add_summary(sei)
        self.logger.close()
        self.progress.close()
        self.evaluator.stdin.close()
        self.evaluator.wait()
    
    def export_profiling_results(self, filepath: Optional[str] = None):
        """导出性能分析结果"""
        if not self.enable_profiling or self.timer_registry is None:
            return
        
        if filepath is None:
            if self.profiling_output is not None:
                filepath = str(self.profiling_output)
            else:
                filepath = str(self.log_path / "profiling_results.json")
        
        self.timer_registry.export_json(filepath)
        print(f"Profiling results exported to: {filepath}")

def create_iter_key_fn(key: jax.Array, sample_per_iteration: int, update_per_iteration: int) -> Callable[[int], Tuple[jax.Array, jax.Array]]:
    def iter_key_fn(step: int):
        iter_key = jax.random.fold_in(key, step)
        sample_key, update_key = jax.random.split(iter_key)
        if sample_per_iteration > 1:
            sample_key = jax.random.split(sample_key, sample_per_iteration)
        else:
            sample_key = (sample_key,)
        if update_per_iteration > 1:
            update_key = jax.random.split(update_key, update_per_iteration)
        else:
            update_key = (update_key,)
        return sample_key, update_key

    iter_key_fn = jax.jit(iter_key_fn)
    iter_key_fn(0)  # Warm up
    return iter_key_fn
