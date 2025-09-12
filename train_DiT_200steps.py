# -*- coding: utf-8 -*-
"""
Training the State Action DiT for the Hopper with 200-step trajectories
Testing advisor's suggestion: shorter training trajectories might improve performance
"""

import torch
import numpy as np
import time

from hopper import HopperEnv
from dit_utils import State_Normalizer, set_seed, load_dataset
from DiT_SA import SA_ODE

#%% GPU Setup and Optimization
if torch.cuda.is_available():
    device = torch.device("cuda")
    # GPU memory optimization
    torch.cuda.set_per_process_memory_fraction(0.9)
    # Enable memory efficient attention if available
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA: {torch.version.cuda}")
else:
    device = torch.device("cpu")
    print("⚠️  No GPU available, using CPU")

print("Device:", device)

#%% Hyperparameters
n_gradient_steps = 30_000  # Same as before
batch_size = 64
print(f"\n Batch size {batch_size}\n")
model_size = {"d_model": 128, "n_heads": 4, "depth": 4}
H = 32 # horizon, length of each trajectory

#%% Dataset of good trajectories
set_seed(0)
env = HopperEnv() 

# Use the 1500-trajectory dataset but truncate to 200 steps
obs, act = load_dataset("datasets/multi_policy_1500trajs_300steps_noise0.1")
obs = torch.FloatTensor(obs[:, :199]).to(device)  # Use only 200 steps (0-199)
act = torch.FloatTensor(act[:, :199]).to(device)  # Use only 200 steps (0-199)
normalizer = State_Normalizer(obs)
nor_obs = normalizer.normalize(obs)
x = torch.cat([nor_obs, act], dim=-1) # dataset of state-action trajectories
sigma_data = x.std().item()

print(f"📊 Dataset loaded successfully on {device}")
print(f"   Observations shape: {obs.shape}")
print(f"   Actions shape: {act.shape}")
print(f"   Combined data shape: {x.shape}")
print(f"   Using 200-step trajectories instead of 300-step")

#%% Training Diffusion model
print("State Action Diffusion Transformer - 200-step training")
ode = SA_ODE(env, sigma_data, attr_dim=env.state_size, device=device,
             N=5, **model_size, horizon=H)

# Start training timer
print(f"\n🚀 Starting training with {n_gradient_steps * 5} gradient steps (150k total)...")
print(f"📝 Training on 200-step trajectories (vs 300-step in original)")
training_start_time = time.time()

# GPU-optimized training with 200-step data
ode.train(x, int(5*n_gradient_steps), batch_size, extra = "_multi_policy_1500trajs_200steps_GPU")

# End training timer
training_time = time.time() - training_start_time
print(f"\n⏱️  Training completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
print(f"📊 Average time per gradient step: {training_time/(n_gradient_steps * 5):.3f} seconds")

# GPU memory cleanup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("🧹 GPU memory cleared")

print(f"\n✅ Model saved as: SA_ODE_Cond_Hopper__specs_128_4_4_h_32_multi_policy_1500trajs_200steps_GPU.pt")
print(f"💡 Next step: Test this model against the 300-step trained model")
