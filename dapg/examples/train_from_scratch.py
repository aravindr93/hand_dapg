"""
This script illustrates training from scratch
using NPG on the relocate-v0 task.
"""

from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.utils.train_agent import train_agent
import mj_envs
import time as timer

SEED = 100

e = GymEnv('relocate-v0')
policy = MLP(e.spec, hidden_sizes=(64, 64), seed=SEED, init_log_std=-0.5)
baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=64, epochs=2, learn_rate=1e-3)
agent = NPG(e, policy, baseline, normalized_step_size=0.1, seed=SEED, save_logs=True)

print("========================================")
print("Training with RL")
ts = timer.time()
train_agent(job_name='relocate_scratch',
            agent=agent,
            seed=SEED,
            niter=100,
            gamma=0.995,
            gae_lambda=0.97,
            num_cpu=5,
            sample_mode='trajectories',
            num_traj=200,
            save_freq=25,
            evaluation_rollouts=20)
print("========================================")
print("Training complete !!!")
print("time taken = %f" % (timer.time()-ts))
