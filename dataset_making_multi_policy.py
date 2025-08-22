# -*- coding: utf-8 -*-
"""
Multi-Policy Dataset Generation

Creates datasets using multiple trained PPO policies for better diversity
and more robust DiT training.
"""

import pickle
import numpy as np
import random
from ppo_utils import load
from hopper import rollout, HopperEnv
from dataset_making import create_dataset, noised_dataset, noised_rollout


def create_multi_policy_dataset(env, filename, list_of_models, N_per_model, H=300, noise_scale=1e-2):
    """
    Create a dataset using multiple policies for better diversity
    
    Args:
        - env: the gym environment
        - filename: the name where the dataset should be stored
        - list_of_models: list of policy names to use
        - N_per_model: number of trajectories per model
        - H: horizon length of each trajectory
        - noise_scale: noise for initial state variation
    """
    
    print(f"Creating multi-policy dataset with {len(list_of_models)} policies:")
    for model in list_of_models:
        print(f"  - {model}")
    
    # Create dataset using all policies
    create_dataset(env, filename, list_of_models, N_per_model, H, noise_scale)
    
    # Load and analyze the created dataset
    from dit_utils import load_dataset
    try:
        obs, act = load_dataset(filename)
        print(f"\nDataset created successfully!")
        print(f"Shape: {obs.shape} observations, {act.shape} actions")
        print(f"Total trajectories: {obs.shape[0]}")
        print(f"Trajectory length: {obs.shape[1]}")
        
        # Analyze diversity
        print(f"\nState diversity analysis:")
        for i in range(min(6, obs.shape[2])):  # First 6 states
            state_range = obs[:,:,i].max() - obs[:,:,i].min()
            print(f"  State {i}: range = {state_range:.3f}")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")


def create_challenging_states_dataset(env, filename, list_of_models, N_per_model, H=300):
    """
    Create dataset with challenging initial states for better closed-loop performance
    """
    
    print(f"Creating challenging states dataset...")
    
    # Define challenging initial states
    challenging_states = [
        # Near height limit
        np.array([0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        # Near angle limit (positive)
        np.array([0.0, 1.25, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        # Near angle limit (negative)
        np.array([0.0, 1.25, -0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        # High velocity
        np.array([0.0, 1.25, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        # High angular velocity
        np.array([0.0, 1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]),
    ]
    
    total_trajs = len(challenging_states) * len(list_of_models) * N_per_model
    Traj_dataset = np.zeros((total_trajs, H, env.state_size))
    Action_dataset = np.zeros((total_trajs, H, env.action_size))
    
    traj_idx = 0
    
    for model_name in list_of_models:
        print(f"Processing policy: {model_name}")
        _, state_norm, agent = load("policies/" + model_name)
        
        for state_idx, s0 in enumerate(challenging_states):
            for traj_id in range(N_per_model):
                # Add small noise to initial state
                s = s0 + 1e-2 * (np.random.rand(12) - 0.5)
                
                try:
                    Traj, Actions = rollout(env, agent, state_norm, s0=s, display=False)
                    
                    # Pad or truncate to H
                    if Traj.shape[0] < H:
                        # Pad with last state
                        pad_length = H - Traj.shape[0]
                        Traj = np.vstack([Traj, np.tile(Traj[-1], (pad_length, 1))])
                        Actions = np.vstack([Actions, np.tile(Actions[-1], (pad_length, 1))])
                    else:
                        Traj = Traj[:H]
                        Actions = Actions[:H-1]
                    
                    Traj_dataset[traj_idx] = Traj
                    Action_dataset[traj_idx] = Actions
                    traj_idx += 1
                    
                except Exception as e:
                    print(f"Failed to generate trajectory for {model_name}, state {state_idx}: {e}")
                    continue
    
    # Save dataset
    dataset = {"States": Traj_dataset[:traj_idx], "Actions": Action_dataset[:traj_idx]}
    with open(filename + ".pkl", 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"Challenging states dataset created: {traj_idx} trajectories")


def create_closed_loop_dataset(env, filename, list_of_models, N_per_model, H=300):
    """
    Create dataset specifically designed for closed-loop training
    Includes trajectories with deviations and recoveries
    """
    
    print(f"Creating closed-loop specific dataset...")
    
    # This would require more sophisticated trajectory generation
    # that includes deliberate deviations and recovery examples
    # For now, we'll create a dataset with more diverse initial conditions
    
    total_trajs = len(list_of_models) * N_per_model
    Traj_dataset = np.zeros((total_trajs, H, env.state_size))
    Action_dataset = np.zeros((total_trajs, H, env.action_size))
    
    traj_idx = 0
    
    for model_name in list_of_models:
        print(f"Processing policy: {model_name}")
        _, state_norm, agent = load("policies/" + model_name)
        
        for traj_id in range(N_per_model):
            # Create diverse initial states
            s0 = np.array([0.0, 1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
            # Add more variation to initial state
            s = s0 + 5e-2 * (np.random.rand(12) - 0.5)  # Larger noise for diversity
            
            try:
                Traj, Actions = rollout(env, agent, state_norm, s0=s, display=False)
                
                # Pad or truncate to H
                if Traj.shape[0] < H:
                    pad_length = H - Traj.shape[0]
                    Traj = np.vstack([Traj, np.tile(Traj[-1], (pad_length, 1))])
                    Actions = np.vstack([Actions, np.tile(Actions[-1], (pad_length, 1))])
                else:
                    Traj = Traj[:H]
                    Actions = Actions[:H-1]
                
                Traj_dataset[traj_idx] = Traj
                Action_dataset[traj_idx] = Actions
                traj_idx += 1
                
            except Exception as e:
                print(f"Failed to generate trajectory for {model_name}: {e}")
                continue
    
    # Save dataset
    dataset = {"States": Traj_dataset[:traj_idx], "Actions": Action_dataset[:traj_idx]}
    with open(filename + ".pkl", 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"Closed-loop dataset created: {traj_idx} trajectories")


if __name__ == "__main__":
    
    env = HopperEnv()
    
    # Define different dataset configurations
    
    # 1. Multi-policy dataset (recommended)
    print("=== Creating Multi-Policy Dataset ===")
    multi_policy_models = ["base", "hopping", "balance", "neg", "mid", "full"]
    create_multi_policy_dataset(
        env, 
        "datasets/multi_policy_300trajs_300steps", 
        multi_policy_models, 
        N_per_model=50,  # 50 trajectories per policy = 300 total
        H=300, 
        noise_scale=1e-2
    )
    
    # 2. Challenging states dataset
    print("\n=== Creating Challenging States Dataset ===")
    challenging_models = ["full", "mid", "neg"]  # Use policies that can handle challenges
    create_challenging_states_dataset(
        env,
        "datasets/challenging_states_150trajs_300steps",
        challenging_models,
        N_per_model=10,  # 10 trajectories per challenging state per policy
        H=300
    )
    
    # 3. Closed-loop specific dataset
    print("\n=== Creating Closed-Loop Dataset ===")
    closed_loop_models = ["full", "mid"]  # Use stable policies
    create_closed_loop_dataset(
        env,
        "datasets/closed_loop_100trajs_300steps",
        closed_loop_models,
        N_per_model=50,
        H=300
    )
    
    print("\n=== Dataset Creation Complete ===")
    print("Available datasets:")
    print("1. multi_policy_300trajs_300steps.pkl - Diverse multi-policy dataset")
    print("2. challenging_states_150trajs_300steps.pkl - Challenging initial conditions")
    print("3. closed_loop_100trajs_300steps.pkl - Closed-loop specific dataset")
