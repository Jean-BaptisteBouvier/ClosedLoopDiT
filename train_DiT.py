# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 20:16:54 2024

@author: Jean-Baptiste Bouvier

Training the State Action DiT for the Hopper in closed-loop
"""

import torch
import numpy as np

from hopper import HopperEnv
from dit_utils import State_Normalizer, set_seed, load_dataset
from DiT_SA import SA_ODE
from projectors_SA import Reference_Projector


#%% Hyperparameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
n_gradient_steps = 10_000
batch_size = 64
print(f"\n Batch size {batch_size}\n")
model_size = {"d_model": 128, "n_heads": 4, "depth": 4}
H = 32 # horizon, length of each trajectory

mode = "full"



#%% Dataset of good trajectories

set_seed(0)
env = HopperEnv() 
N_trajs = 1000 # number of trajectories for training
obs, act = load_dataset("datasets/" + mode + f"_{N_trajs}trajs_300steps")
obs = torch.FloatTensor(obs[:, :299]).to(device)
act = torch.FloatTensor(act[:, :299]).to(device)
normalizer = State_Normalizer(obs)
nor_obs = normalizer.normalize(obs)
x = torch.cat([nor_obs, act], dim=-1) # dataset of state-action trajectories
sigma_data = x.std().item()


#%% Training Diffusion model on the dataset of clean trajectories
print("State Action Diffusion Transformer without projections")
ode = SA_ODE(env, sigma_data, attr_dim=env.state_size, device=device,
             N=5, **model_size, horizon=H)
# assert ode.load(extra = "_" + mode)
ode.train(x, int(5*n_gradient_steps), batch_size, extra = "_" + mode)



#%% Recovery dataset
# obs, act = load_dataset("datasets/" + mode + f"_500trajs_40steps")
# obs = torch.FloatTensor(obs).to(device)
# act = torch.FloatTensor(act).to(device)

# nor_obs = normalizer.normalize(obs)
# x = torch.cat([nor_obs, act], dim=-1) # dataset of state-action trajectories
# sigma_data = x.std().item()


# #%% Training Diffusion models
# print("State Action Diffusion Transformer without projections")
# ode = SA_ODE(env, sigma_data, device=device, N=5, **model_size, horizon=H)
# assert ode.load(extra = "_" + mode)
# ode.train(x, int(5*n_gradient_steps), batch_size, extra = "_" + mode + "_finetuned")




