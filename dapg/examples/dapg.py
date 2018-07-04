"""
This script illustrates initializing with behavior cloning
and finetuning policy with DAPG.
"""

from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.dapg import DAPG
from mjrl.algos.behavior_cloning_2 import BC
from mjrl.utils.train_agent import train_agent
import mj_envs
import time as timer
import pickle

SEED = 100

# ------------------------------
# Get demonstrations
print("========================================")
print("Collecting expert demonstrations")
print("========================================")
demo_paths = pickle.load(open('../demonstrations/relocate-v0_demos.pickle', 'rb'))

# ------------------------------
# Train BC
e = GymEnv('relocate-v0')
policy = MLP(e.spec, hidden_sizes=(32,32), seed=SEED)
bc_agent = BC(demo_paths, policy=policy, epochs=5, batch_size=32, lr=1e-3)

ts = timer.time()
print("========================================")
print("Running BC with expert demonstrations")
print("========================================")
bc_agent.train()
print("========================================")
print("BC training complete !!!")
print("time taken = %f" % (timer.time()-ts))
print("========================================")

score = e.evaluate_policy(policy, num_episodes=10, mean_action=True)
print("Score with behavior cloning = %f" % score[0][0])

# ------------------------------
# Finetune with DAPG
print("========================================")
print("Finetuning with DAPG")
baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=64, epochs=2, learn_rate=1e-3)
agent = DAPG(e, policy, baseline, demo_paths=demo_paths, normalized_step_size=0.1,
             seed=SEED, lam_0=1e-2, lam_1=0.99, save_logs=True)

ts = timer.time()
train_agent(job_name='relocate_demo_init_dapg',
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
print("time taken = %f" % (timer.time()-ts))
