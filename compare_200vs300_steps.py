# -*- coding: utf-8 -*-
"""
Compare models trained on 200-step vs 300-step trajectories
Testing advisor's hypothesis: shorter training trajectories might improve performance
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

# Load dataset
env = HopperEnv()
obs, act = load_dataset("datasets/multi_policy_1500trajs_300steps_noise0.1")
obs = torch.FloatTensor(obs[:, :299]).to(device)  # Full 300-step data for testing
act = torch.FloatTensor(act[:, :299]).to(device)  # Full 300-step data for testing
normalizer = State_Normalizer(obs)
nor_obs = normalizer.normalize(obs)
x = torch.cat([nor_obs, act], dim=-1)
sigma_data = x.std().item()

# Test parameters
traj_id = 10
true_traj = obs[traj_id]
s0 = true_traj[0].reshape(1, env.state_size)
replan_horizon = 2
N = 10  # Best denoising steps from optimization

print("🧪 Comparing 200-step vs 300-step Training")
print("=" * 60)
print("Testing advisor's hypothesis: shorter training trajectories might improve performance")

#%% Test 300-step trained model
print(f"\n📊 Testing 300-step trained model:")
print("-" * 40)

ode_300 = SA_ODE(env, sigma_data, attr_dim=env.state_size, device=device, N=5, **model_size, horizon=H)
if ode_300.load(extra="_multi_policy_1500trajs_GPU"):
    planner_300 = SA_Planner(env, ode_300, normalizer)
    
    # Test closed-loop performance
    try:
        pred, actual, reward = planner_300.closed_loop_traj(s0, traj_len=300, 
                                                          replan_horizon=replan_horizon, N=N)
        survival_steps = len(actual) if len(actual) < 300 else 300
        print(f"✅ Closed-loop: {reward:.1f} reward, {survival_steps} steps")
        reward_300_cl = reward
    except Exception as e:
        print(f"❌ Closed-loop: Failed - {e}")
        reward_300_cl = 0
    
    # Test best_traj for comparison
    try:
        pred, actions, actual, reward = planner_300.best_traj(s0, traj_len=300, 
                                                            replan_horizon=replan_horizon, 
                                                            n_samples_per_s0=8)
        if type(actual) == list:
            actual = actual[0]
        survival_steps = len(actual) if len(actual) < 300 else 300
        print(f"✅ Best-traj:   {reward:.1f} reward, {survival_steps} steps")
        reward_300_bt = reward
    except Exception as e:
        print(f"❌ Best-traj:   Failed - {e}")
        reward_300_bt = 0
else:
    print("❌ 300-step model not found")
    reward_300_cl = 0
    reward_300_bt = 0

#%% Test 200-step trained model
print(f"\n📊 Testing 200-step trained model:")
print("-" * 40)

ode_200 = SA_ODE(env, sigma_data, attr_dim=env.state_size, device=device, N=5, **model_size, horizon=H)
if ode_200.load(extra="_multi_policy_1500trajs_200steps_GPU"):
    planner_200 = SA_Planner(env, ode_200, normalizer)
    
    # Test closed-loop performance
    try:
        pred, actual, reward = planner_200.closed_loop_traj(s0, traj_len=300, 
                                                          replan_horizon=replan_horizon, N=N)
        survival_steps = len(actual) if len(actual) < 300 else 300
        print(f"✅ Closed-loop: {reward:.1f} reward, {survival_steps} steps")
        reward_200_cl = reward
    except Exception as e:
        print(f"❌ Closed-loop: Failed - {e}")
        reward_200_cl = 0
    
    # Test best_traj for comparison
    try:
        pred, actions, actual, reward = planner_200.best_traj(s0, traj_len=300, 
                                                            replan_horizon=replan_horizon, 
                                                            n_samples_per_s0=8)
        if type(actual) == list:
            actual = actual[0]
        survival_steps = len(actual) if len(actual) < 300 else 300
        print(f"✅ Best-traj:   {reward:.1f} reward, {survival_steps} steps")
        reward_200_bt = reward
    except Exception as e:
        print(f"❌ Best-traj:   Failed - {e}")
        reward_200_bt = 0
else:
    print("❌ 200-step model not found - need to train it first")
    reward_200_cl = 0
    reward_200_bt = 0

#%% Results Comparison
print(f"\n📈 RESULTS COMPARISON:")
print("=" * 60)

print(f"{'Method':<15} {'300-step':<12} {'200-step':<12} {'Difference':<12}")
print("-" * 60)
print(f"{'Closed-loop':<15} {reward_300_cl:<12.1f} {reward_200_cl:<12.1f} {reward_200_cl - reward_300_cl:<+12.1f}")
print(f"{'Best-traj':<15} {reward_300_bt:<12.1f} {reward_200_bt:<12.1f} {reward_200_bt - reward_300_bt:<+12.1f}")

#%% Analysis
print(f"\n🔍 ANALYSIS:")
print("=" * 40)

if reward_200_cl > 0 and reward_300_cl > 0:
    cl_improvement = ((reward_200_cl - reward_300_cl) / reward_300_cl) * 100
    print(f"Closed-loop improvement: {cl_improvement:+.1f}%")
    
    if cl_improvement > 0:
        print("✅ Advisor's hypothesis CONFIRMED: 200-step training is better!")
    else:
        print("❌ Advisor's hypothesis REJECTED: 300-step training is better")
        
    print(f"\n💡 Key insights:")
    print(f"   - 200-step model: {reward_200_cl:.1f} reward")
    print(f"   - 300-step model: {reward_300_cl:.1f} reward")
    print(f"   - Difference: {reward_200_cl - reward_300_cl:+.1f} points")
    
    if abs(reward_200_cl - reward_300_cl) < 10:
        print(f"   - Results are very close - both approaches work well")
    elif reward_200_cl > reward_300_cl:
        print(f"   - Shorter training trajectories lead to better generalization")
    else:
        print(f"   - Longer training trajectories provide more learning data")

else:
    print("❌ Cannot compare - one or both models failed to load/run")
    print("💡 Make sure to train the 200-step model first:")

print(f"\n🚀 Next steps:")
print(f"   1. Train 200-step model: python train_DiT_200steps.py")
print(f"   2. Run this comparison: python compare_200vs300_steps.py")
print(f"   3. Use the better performing model for your final results")


