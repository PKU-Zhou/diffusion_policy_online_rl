import argparse
import os.path
from pathlib import Path
import time
from functools import partial
import yaml
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import jax, jax.numpy as jnp

from relax.algorithm.sac import SAC
from relax.algorithm.dsact import DSACT
from relax.algorithm.dacer import DACER
from relax.algorithm.dacer_doubleq import DACERDoubleQ
from relax.algorithm.qsm import QSM
from relax.algorithm.dipo import DIPO
from relax.algorithm.qvpo import QVPO
from relax.algorithm.sdac import SDAC
from relax.algorithm.dpmd import DPMD
from relax.algorithm.idem import IDEM
from relax.buffer import TreeBuffer
from relax.network.sac import create_sac_net
from relax.network.dsact import create_dsact_net
from relax.network.dacer import create_dacer_net
from relax.network.dacer_doubleq import create_dacer_doubleq_net
from relax.network.qsm import create_qsm_net
from relax.network.dipo import create_dipo_net
from relax.network.diffv2 import create_diffv2_net
from relax.network.qvpo import create_qvpo_net
from relax.trainer.off_policy import OffPolicyTrainer
from relax.env import create_env, create_vector_env
from relax.utils.experience import Experience, ObsActionPair
from relax.utils.fs import PROJECT_ROOT
from relax.utils.random_utils import seeding
from relax.utils.log_diff import log_git_details

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str, default="sdac")
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--suffix", type=str, default="test_use_atp1")
    parser.add_argument("--num_vec_envs", type=int, default=5)
    parser.add_argument("--hidden_num", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--diffusion_hidden_dim", type=int, default=256)
    parser.add_argument("--start_step", type=int, default=int(3e4)) # other envs 3e4
    parser.add_argument("--total_step", type=int, default=int(1e6))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_schedule_end", type=float, default=3e-5)
    parser.add_argument("--alpha_lr", type=float, default=7e-3)
    parser.add_argument("--delay_alpha_update", type=float, default=250)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--num_particles", type=int, default=32)
    parser.add_argument("--noise_scale", type=float, default=0.1)
    parser.add_argument("--cluster", default=False, action="store_true")
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--beta_schedule_scale", type=float, default=0.8)
    parser.add_argument("--beta_schedule_type", type=str, default='linear')
    parser.add_argument("--enable_profiling", action='store_true', default=False, help='Enable performance profiling')
    parser.add_argument("--profiling_output", type=str, default=None, help='Path to save profiling results (default: log_path/profiling_results.json)')
    parser.add_argument("--enable_data_collection", action='store_true', default=False, help='Enable data distribution collection during training')
    parser.add_argument("--data_collection_interval", type=int, default=1000, help='Interval (in steps) for data collection')
    parser.add_argument("--data_collection_layers", type=str, default="linear_0,linear_1,linear_2,linear_3,linear_4,linear_5", help='Comma-separated layer names to collect data from')
    parser.add_argument("--data_collection_types", type=str, default="weight,gradient,activation", help='Comma-separated data types to collect (weight,gradient,activation)')
    parser.add_argument("--disable_evaluator", action='store_true', default=False, help='Disable the evaluator subprocess during training')
    args = parser.parse_args()

    if args.debug:
        from jax import config
        config.update("jax_disable_jit", True)

    master_seed = args.seed
    master_rng, _ = seeding(master_seed)
    env_seed, env_action_seed, eval_env_seed, buffer_seed, init_network_seed, train_seed = map(
        int, master_rng.integers(0, 2**32 - 1, 6)
    )
    init_network_key = jax.random.key(init_network_seed)
    train_key = jax.random.key(train_seed)
    del init_network_seed, train_seed

    if args.num_vec_envs > 0:
        env, obs_dim, act_dim = create_vector_env(args.env, args.num_vec_envs, env_seed, env_action_seed, mode="futex")
    else:
        env, obs_dim, act_dim = create_env(args.env, env_seed, env_action_seed)
    eval_env = None

    hidden_sizes = [args.hidden_dim] * args.hidden_num
    diffusion_hidden_sizes = [args.diffusion_hidden_dim] * args.hidden_num

    buffer = TreeBuffer.from_experience(obs_dim, act_dim, size=int(1e6), seed=buffer_seed)

    gelu = partial(jax.nn.gelu, approximate=False)
    
    print(f"Algorithm: {args.alg}")
    
    # 创建数据收集器（如果启用）
    data_collector = None
    layer_patterns = []
    if args.enable_data_collection:
        from zczhou.data_distribution.utils.data_collector import DataCollector
        
        # 日志路径会在后面创建，这里先预留
        # 数据收集器会在算法创建后、trainer创建时被使用
        layer_patterns = args.data_collection_layers.split(',')
        collect_types = [t.strip() for t in args.data_collection_types.split(',') if t.strip()]
        print(f"数据收集已启用，采样间隔：{args.data_collection_interval} 步")
        print(f"收集层：{layer_patterns}")
        print(f"收集类型：{collect_types}")

    if args.alg == 'sdac':
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_diffv2_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                          num_timesteps=args.diffusion_steps, 
                                          num_particles=args.num_particles, 
                                          noise_scale=args.noise_scale,
                                          beta_schedule_scale=args.beta_schedule_scale)
        
        # 为SDAC创建数据收集器
        if args.enable_data_collection:
            # 临时目录，实际目录会在exp_dir创建后更新
            temp_output_dir = Path("temp_data_collection")
            data_collector = DataCollector(
                output_dir=temp_output_dir,
                sample_interval=args.data_collection_interval,
                layer_patterns=layer_patterns,
                save_batch_size=1,
                collect_types=collect_types
            )
        
        algorithm = SDAC(agent, params, lr=args.lr, alpha_lr=args.alpha_lr, 
                        delay_alpha_update=args.delay_alpha_update, 
                        lr_schedule_end=args.lr_schedule_end,
                        data_collector=data_collector)
    
    elif args.alg == 'dpmd':
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_diffv2_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                          num_timesteps=args.diffusion_steps, 
                                          num_particles=args.num_particles, 
                                          noise_scale=args.noise_scale,
                                          beta_schedule_scale=args.beta_schedule_scale)
        algorithm = DPMD(agent, params, lr=args.lr, alpha_lr=args.alpha_lr, delay_alpha_update=args.delay_alpha_update, lr_schedule_end=args.lr_schedule_end)

    elif args.alg == 'idem':
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_diffv2_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                          num_timesteps=args.diffusion_steps, 
                                          num_particles=args.num_particles, 
                                          noise_scale=args.noise_scale,
                                          beta_schedule_scale=args.beta_schedule_scale)
        algorithm = IDEM(agent, params, lr=args.lr, alpha_lr=args.alpha_lr, delay_alpha_update=args.delay_alpha_update, lr_schedule_end=args.lr_schedule_end)
    elif args.alg == "qsm":
        agent, params = create_qsm_net(init_network_key, obs_dim, act_dim, hidden_sizes, num_timesteps=20, num_particles=args.num_particles)
        algorithm = QSM(agent, params, lr=args.lr, lr_schedule_end=args.lr_schedule_end)
    elif args.alg == "sac":
        agent, params = create_sac_net(init_network_key, obs_dim, act_dim, hidden_sizes, gelu)
        algorithm = SAC(agent, params, lr=args.lr)
    elif args.alg == "dsact":
        agent, params = create_dsact_net(init_network_key, obs_dim, act_dim, hidden_sizes, gelu)
        algorithm = DSACT(agent, params, lr=args.lr)
    elif args.alg == "dacer":
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_dacer_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish, 
                                         num_timesteps=args.diffusion_steps)
        algorithm = DACER(agent, params, lr=args.lr, lr_schedule_end=args.lr_schedule_end)
    elif args.alg == "dacer_doubleq":
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_dacer_doubleq_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish, num_timesteps=args.diffusion_steps)
        algorithm = DACERDoubleQ(agent, params, lr=args.lr)
    elif args.alg == "dipo":
        diffusion_buffer = TreeBuffer.from_example(
            ObsActionPair.create_example(obs_dim, act_dim),
            args.total_step,
            int(master_rng.integers(0, 2**32 - 1)),
            remove_batch_dim=False
        )
        TreeBuffer.connect(buffer, diffusion_buffer, lambda exp: ObsActionPair(exp.obs, exp.action))

        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))

        agent, params = create_dipo_net(init_network_key, obs_dim, act_dim, hidden_sizes, num_timesteps=100)
        algorithm = DIPO(agent, params, diffusion_buffer, lr=args.lr, action_gradient_steps=30, policy_target_delay=2, action_grad_norm=0.16)
    elif args.alg == "qvpo":
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_qvpo_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                          num_timesteps=args.diffusion_steps,
                                          num_particles=args.num_particles,
                                          noise_scale=args.noise_scale)
        algorithm = QVPO(agent, params, lr=args.lr, alpha_lr=args.alpha_lr, delay_alpha_update=args.delay_alpha_update)
    else:
        raise ValueError(f"Invalid algorithm {args.alg}!")

    if args.cluster:
        PROJECT_ROOT = Path('/n/netscratch/nali_lab_seas/Lab/haitongma/sdac_logs')
    
    exp_dir = PROJECT_ROOT / "logs" / args.env / (args.alg + '_' + time.strftime("%Y-%m-%d_%H-%M-%S") + f'_s{args.seed}_{args.suffix}')
    
    # Update data collector output path if enabled
    if args.enable_data_collection and data_collector is not None:
        data_output_dir = exp_dir / "data_distribution"
        data_output_dir.mkdir(parents=True, exist_ok=True)
        data_collector.output_dir = data_output_dir
        print(f"数据收集输出目录：{data_output_dir}")
    
    # Set profiling output path if profiling is enabled
    profiling_output = None
    if args.enable_profiling and args.profiling_output:
        profiling_output = Path(args.profiling_output)
    
    trainer = OffPolicyTrainer(
        env=env,
        algorithm=algorithm,
        buffer=buffer,
        start_step=args.start_step,
        total_step=args.total_step,
        sample_per_iteration=1,
        evaluate_env=eval_env,
        save_policy_every=int(args.total_step / 20),
        warmup_with="random",
        log_path=exp_dir,
        update_log_n_step=1 if args.debug else 1000,
        enable_profiling=args.enable_profiling,
        profiling_output=profiling_output,
        disable_evaluator=args.disable_evaluator,
    )

    trainer.setup(Experience.create_example(obs_dim, act_dim, trainer.batch_size))
    log_git_details(log_file=os.path.join(exp_dir, 'dacer.diff'))
    
    # Save the arguments to a YAML file
    args_dict = vars(args)
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as yaml_file:
        yaml.dump(args_dict, yaml_file)
    
    # Run training
    trainer.run(train_key)
    
    # Finalize data collection if enabled
    if args.enable_data_collection and data_collector is not None:
        data_collector.finalize()
    
    # Export profiling results if enabled
    if args.enable_profiling:
        trainer.export_profiling_results()
