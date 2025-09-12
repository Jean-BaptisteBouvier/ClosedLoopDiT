# -*- coding: utf-8 -*-
"""
Test model performance on different initial states
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

# Initialize model
ode = SA_ODE(env, sigma_data, attr_dim=env.state_size, device=device, N=5, **model_size, horizon=H)
if not ode.load(extra=f"_{mode}"):
    print(f"❌ Model not found for {mode}")
    exit()

planner = SA_Planner(env, ode, normalizer)

# Test parameters
replan_horizon = 4
test_traj_ids = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]  # Different initial states

print("🧪 Testing Multiple Initial States (CLOSED-LOOP PERFORMANCE)")
print("=" * 60)

results = []

for traj_id in test_traj_ids:
    print(f"\n📊 Testing Trajectory {traj_id}:")
    
    true_traj = obs[traj_id]
    s0 = true_traj[0].reshape(1, env.state_size)
    
    # Test closed-loop performance
    try:
        pred, actual, reward = planner.closed_loop_traj(s0, traj_len=300, 
                                                      replan_horizon=replan_horizon)
        survival_steps = len(actual) if len(actual) < 300 else 300
        
        results.append({
            "traj_id": traj_id,
            "reward": reward,
            "survival_steps": survival_steps
        })
        
        print(f"  Closed-loop: {reward:.1f} reward, {survival_steps} steps")
        
    except Exception as e:
        print(f"  Closed-loop: Failed - {e}")
        results.append({
            "traj_id": traj_id,
            "reward": 0,
            "survival_steps": 0
        })
    
    # Test best_traj for comparison
    try:
        pred, actions, actual, reward = planner.best_traj(s0, traj_len=300, 
                                                        replan_horizon=replan_horizon, 
                                                        n_samples_per_s0=8)
        if type(actual) == list:
            actual = actual[0]
        survival_steps = len(actual) if len(actual) < 300 else 300
        print(f"  Best-traj:   {reward:.1f} reward, {survival_steps} steps")
        
    except Exception as e:
        print(f"  Best-traj:   Failed - {e}")

#%% Results Summary
print(f"\n📈 RESULTS SUMMARY:")
print(f"{'Traj ID':<8} {'CL Reward':<10} {'CL Steps':<9}")
print("-" * 35)
for result in results:
    print(f"{result['traj_id']:<8} {result['reward']:<10.1f} {result['survival_steps']:<9}")

if results:
    avg_reward = np.mean([r['reward'] for r in results])
    avg_steps = np.mean([r['survival_steps'] for r in results])
    print(f"\n📊 Averages:")
    print(f"   Closed-loop Reward: {avg_reward:.1f}")
    print(f"   Closed-loop Steps:  {avg_steps:.1f}")
    
    # Find best and worst performing states
    best_result = max(results, key=lambda x: x['reward'])
    worst_result = min(results, key=lambda x: x['reward'])
    
    print(f"\n🏆 Best State: Trajectory {best_result['traj_id']} ({best_result['reward']:.1f} reward)")
    print(f"❌ Worst State: Trajectory {worst_result['traj_id']} ({worst_result['reward']:.1f} reward)")
    
    print(f"\n💡 Insights:")
    print(f"   - Model performance varies significantly across different initial states")
    print(f"   - Some states are much easier/harder for the model to handle")
    print(f"   - Consider training on more diverse initial states if performance is inconsistent")
