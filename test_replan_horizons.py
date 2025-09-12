# -*- coding: utf-8 -*-
"""
Test different replanning horizons to find optimal setting
"""

import torch
import numpy as np

from hopper import HopperEnv
from dit_utils import State_Normalizer, set_seed, load_dataset
from DiT_SA import SA_ODE, SA_Planner

#%% Setup
set_seed(0)
device = "cpu"
H = 32
model_size = {"d_model": 128, "n_heads": 4, "depth": 4}
mode = "multi_policy_1500trajs_GPU"

# Load dataset
env = HopperEnv()
obs, act = load_dataset("datasets/multi_policy_1500trajs_300steps_noise0.1")
obs = torch.FloatTensor(obs[:, :299]).to(device)
act = torch.FloatTensor(act[:, :299]).to(device)
normalizer = State_Normalizer(obs)
nor_obs = normalizer.normalize(obs)
x = torch.cat([nor_obs, act], dim=-1)
sigma_data = x.std().item()

# Test parameters
traj_id = 10
true_traj = obs[traj_id]
s0 = true_traj[0].reshape(1, env.state_size)

# Test different replanning horizons
replan_horizons = [2, 4, 6, 8, 12, 16, 24, 32]

print("🧪 Testing Different Replanning Horizons (CLOSED-LOOP PERFORMANCE)")
print("=" * 60)

results = []

for replan_h in replan_horizons:
    print(f"\n📊 Testing Replan Horizon: {replan_h}")
    
    # Initialize model
    ode = SA_ODE(env, sigma_data, attr_dim=env.state_size, device=device, N=5, **model_size, horizon=H)
    if not ode.load(extra=f"_{mode}"):
        print(f"❌ Model not found for {mode}")
        continue
    
    planner = SA_Planner(env, ode, normalizer)
    
    # Test closed-loop performance
    try:
        pred, actual, reward = planner.closed_loop_traj(s0, traj_len=300, replan_horizon=replan_h)
        survival_steps = len(actual) if len(actual) < 300 else 300
        
        results.append({
            "replan_h": replan_h,
            "reward": reward,
            "survival_steps": survival_steps
        })
        
        print(f"✅ Reward: {reward:.1f}, Steps: {survival_steps}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        results.append({
            "replan_h": replan_h,
            "reward": 0,
            "survival_steps": 0
        })

#%% Results
print(f"\n📈 RESULTS SUMMARY:")
print(f"{'Replan H':<10} {'Reward':<8} {'Steps':<6}")
print("-" * 30)
for result in results:
    print(f"{result['replan_h']:<10} {result['reward']:<8.1f} {result['survival_steps']:<6}")

# Find best
if results:
    best_result = max(results, key=lambda x: x['reward'])
    print(f"\n🏆 Best Replan Horizon: {best_result['replan_h']}")
    print(f"   Reward: {best_result['reward']:.1f}")
    print(f"   Survival Steps: {best_result['survival_steps']}")
