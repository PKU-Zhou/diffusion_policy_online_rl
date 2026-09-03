SDAC算法的网络架构详解
SDAC (Soft Actor-Critic with Diffusion) 确实使用了6个神经网络,这个设计继承自SAC算法并做了扩展。让我从代码中为你详细解释:

六个网络的组成
从代码第26-28行可以看到参数结构:

class Diffv2Params(NamedTuple):
    q1: 主Q网络1的参数
    q2: 主Q网络2的参数
    target_q1: 目标Q网络1的参数
    target_q2: 目标Q网络2的参数
    policy: 主策略网络的参数
    target_policy: 目标策略网络的参数
    log_alpha: 温度参数(不是网络)
1. 主Q网络 (q1, q2) - 价值评估器
作用: 评估在某个状态下采取某个动作的价值(Q值)

从代码104-107行的Q网络损失函数可以看到:


sdac.py
Ln 122–126
def q_loss_fn(q_params: hk.Params) -> jax.Array:
    q = self.agent.q(q_params, obs, action)
    q_loss = jnp.mean((q - q_backup) ** 2)
    return q_loss, q
为什么需要两个Q网络? 这是Double Q-learning技术,用于解决Q值过高估计问题
在更新时,两个Q网络独立训练,但在选择动作时取最小值(代码110-113行):

sdac.py
Ln 108–113
def get_min_q(s, a):
    q1 = self.agent.q(q1_params, s, a)
    q2 = self.agent.q(q2_params, s, a)
    q = jnp.minimum(q1, q2)
    return q
2. 目标Q网络 (target_q1, target_q2) - 稳定的价值目标
作用: 提供稳定的Q值目标,用于计算TD误差

从代码116-120行可以看到目标Q网络的使用:


sdac.py
Ln 116–120
next_action = self.agent.get_action(next_eval_key, (policy_params, log_alpha, q1_params, q2_params), next_obs)
q1_target = self.agent.q(target_q1_params, next_obs, next_action)
q2_target = self.agent.q(target_q2_params, next_obs, next_action)
q_target = jnp.minimum(q1_target, q2_target)  # - jnp.exp(log_alpha) * next_logp
q_backup = reward + (1 - done) * self.gamma * q_target
为什么需要目标网络?

在强化学习中,如果用同一个网络既生成目标又更新自己,会导致训练不稳定(追逐移动目标)
目标网络通过软更新(slow tracking)方式缓慢跟踪主网络,提供稳定的学习目标
从代码234-239行可以看到目标网络的更新方式:


sdac.py
Ln 234–239
def delay_target_update(params, target_params, tau):
    return jax.lax.cond(
        step % self.delay_update == 0,
        lambda target_params: optax.incremental_update(params, target_params, tau),
        lambda target_params: target_params,
        target_params
    )
使用参数 tau=0.005(第58行),意味着每次更新时:

target_params = 0.005 * main_params + 0.995 * target_params
3. 主策略网络 (policy) - 动作生成器
作用: 这是一个基于扩散模型的策略网络,用于生成动作

这是SDAC的核心创新!不同于传统SAC使用高斯策略,这里使用扩散模型来表示策略分布。

从代码136-167行可以看到策略网络的训练:


sdac.py
Ln 136–167
def policy_loss_fn(policy_params) -> jax.Array:
                
    # q_weights = q_weights
    def denoiser(t, x):
        return self.agent.policy(policy_params, wide_obs, x, t)
                
    # loss = self.agent.diffusion.weighted_p_loss(diffusion_noise_key, q_weights, denoiser, t,
    #                                             jax.lax.stop_gradient(next_action))
    noise2 = jax.random.normal(diff_key2, (action.shape[0] * reverse_mc_num, action.shape[1]))
    recon = self.agent.diffusion.get_recon(t, tilde_at, noise2).clip(-1, 1)
    q_min = get_min_q(wide_obs, recon) * 5. / jnp.exp(log_alpha) # 5 is the initial alpha value
    q_mean, q_std = q_min.mean(), q_min.std()
    q_reshape = q_min.reshape((-1, reverse_mc_num)) # [batch_size, mc_num]
    Z = jax.nn.logsumexp(q_reshape, axis=1, keepdims=True) # [batch_size, 1]
    q_weights = jnp.exp(q_reshape - Z).flatten() # [batch_size, mc_num]
                
    loss = self.agent.diffusion.reverse_samping_weighted_p_loss(noise2,
                                                                q_weights, #  * t_weights
                                                                denoiser,
                                                                t,
                                                                tilde_at,)
    return loss, (q_weights, q_min, q_mean, q_std, recon)
策略网络通过Q值加权的扩散损失来训练,目标是生成高Q值的动作。

4. 目标策略网络 (target_policy) - 稳定的策略目标
作用: 提供稳定的策略,用于生成next_action

从代码242-244行可以看到目标策略网络的更新:


sdac.py
Ln 242–244
target_q1_params = delay_target_update(q1_params, target_q1_params, self.tau)
target_q2_params = delay_target_update(q2_params, target_q2_params, self.tau)
target_policy_params = delay_target_update(policy_params, target_policy_params, self.tau)
网络之间的关系图
训练循环中的信息流:
1. 采样经验: (s, a, r, s')
   
2. Q网络更新:
   s' -> target_policy -> a' (下一个状态的动作)
   (s', a') -> target_q1, target_q2 -> min(Q_target) 
   Q_backup = r + γ * min(Q_target)
   (s, a) -> q1, q2 -> 计算与Q_backup的MSE损失
   
3. 策略网络更新:
   s -> policy -> 采样多个动作
   (s, 采样的动作) -> q1, q2 -> min(Q)
   根据Q值加权的扩散损失更新policy
   
4. 目标网络软更新:
   target_q1 ← τ*q1 + (1-τ)*target_q1
   target_q2 ← τ*q2 + (1-τ)*target_q2
   target_policy ← τ*policy + (1-τ)*target_policy
为什么需要这么多网络?总结
两个Q网络 (q1, q2): 防止Q值过高估计(Double Q-learning)
目标Q网络 (target_q1, target_q2): 提供稳定的TD目标,避免训练发散
主策略网络 (policy): 生成动作并不断改进
目标策略网络 (target_policy): 提供稳定的next_action,用于计算Q值目标
这种设计是现代深度强化学习的标准做法,结合了:

Double Q-learning (Hasselt et al., 2010)
Target Networks (Mnih et al., 2015, DQN)
Soft Actor-Critic框架 (Haarnoja et al., 2018)
扩散策略 (本算法的创新点)
这样的多网络架构虽然复杂,但能够提供稳定、高效的训练过程!