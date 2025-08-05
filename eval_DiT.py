# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 13:29:38 2025

@author: jeanb
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from hopper import HopperEnv
from plots import traj_comparison
from dit_utils import State_Normalizer, set_seed, load_dataset
# from utils import closed_loop_stats, barplot_comparison
from DiT_SA import SA_ODE, SA_Planner
# from projectors_SA import Reference_Projector, Admissible_Projector
 
 
 
#%% Hyperparameters

plot_height = False
device = "cpu"
H = 32 # prediction horizon
N_s0 = 16 # number of different initial states sampled to evaluate each model
N_samples = 8 # number of trajectories sampled pre initial state to evaluate each model
model_size = {"d_model": 128, "n_heads": 4, "depth": 4}
mode = "full"
  
 
  
#%% Normalizes states and concatenates them to actions

set_seed(0) 
env = HopperEnv()
N_trajs = 1000
obs, act = load_dataset(f"datasets/{mode}_{N_trajs}trajs_300steps")
obs = torch.FloatTensor(obs[:, :299]).to(device)
act = torch.FloatTensor(act[:, :299]).to(device)
normalizer = State_Normalizer(obs)
nor_obs = normalizer.normalize(obs)
x = torch.cat([nor_obs, act], dim=-1) # dataset of state-action trajectories
sigma_data = x.std().item()
  
 
  
#%% Diffusion Transformer

ode = SA_ODE(env, sigma_data, device=device, N=5, **model_size, horizon=H)
assert ode.load(extra=f"_{mode}")
# assert ode.load(extra=f"_{mode}_finetuned")
planner = SA_Planner(env, ode, normalizer)
 

  
#%% Conditional Diffusion Transformer on initial state

cond_ode = SA_ODE(env, sigma_data, attr_dim=env.state_size, device=device,
                  N=5, **model_size, horizon=H)
assert cond_ode.load(extra=f"_{mode}")
# assert cond_ode.load(extra=f"_{mode}_finetuned")
cond_planner = SA_Planner(env, cond_ode, normalizer)
 
 
 
#%% Initial trajectory

traj_id = 0
true_traj = obs[traj_id]
s0 = true_traj[0].reshape(1, env.state_size)
 
replan_horizon = H#//2
 
  
#%% DiT

# pred, actual, reward = planner.closed_loop_traj(s0, traj_len=300, replan_horizon=replan_horizon)
pred, actions, actual, reward = planner.best_traj(s0, traj_len=300, replan_horizon=replan_horizon, n_samples_per_s0=8)
if type(pred) == list:
    pred = pred[0]
    actual = actual[0]

traj_comparison(env, pred, "sampled", actual, "actual", horizon=replan_horizon,
                title=f"Prediction_h={H} replan_h={replan_horizon}", 
                plot_height=plot_height)



#%% Conditional DiT on initial state

pred, actual, reward = cond_planner.closed_loop_traj(s0, traj_len=300, replan_horizon=replan_horizon)
# pred, actions, actual, reward = cond_planner.best_traj(s0, traj_len=300, replan_horizon=replan_horizon, n_samples_per_s0=8)
if type(pred) == list:
    pred = pred[0]
    actual = actual[0]

traj_comparison(env, pred, "sampled", actual, "actual", horizon=replan_horizon,
                title=f"Conditioned s_0 prediction_h={H} replan_h={replan_horizon}", 
                plot_height=plot_height)

  


#%%

# traj_pred, action_pred, actual_traj, Rewards = planner.best_traj(s0, traj_len=300, replan_horizon=replan_horizon,
#                                                                   n_samples_per_s0=N_samples, projector=None)

# traj_comparison(env, true_traj, "true", traj[0], "sampled",
#                 traj_3=ID_traj, label_3="ID",
#                 traj_4=ol_traj, label_4="open-loop",
#                 title="Trained without projections", 
#                 plot_height=plot_height)
  
 
   
