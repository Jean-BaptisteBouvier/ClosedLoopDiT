# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 16:18:34 2024

@author: Jean-Baptiste

Generating a dataset of trajectories from different trained PPO models on the Hopper
"""

import pickle
import numpy as np
from ppo_utils import load
from hopper import rollout, noised_rollout
from tqdm import tqdm
import time


def print_progress(current, total, model_name, noise_level):
    """Print progress information"""
    percentage = (current / total) * 100
    print(f"\r{noise_level.upper()} Dataset - {model_name}: {current}/{total} ({percentage:.1f}%)", end="", flush=True)


def create_dataset(env, filename, list_of_models, N, H=300, noise_scale=1e-2, noise_level="clean"):
    """Create a dataset of trajectories using pretrained models.
    Args:
        - env: the gym environment
        - filename: the name where the dataset should be stored
        - list_of_models: a list containing the name of each pretrained model
        - N: the number of trajectories per model
        - H: the horizon, i.e., length of each trajectory
        - noise_scale: minmax bound for the uniform noise on the state reset
        - noise_level: string identifier for noise level ("clean", "low", "medium", "high", "extreme")"""
    
    s0 = np.array([0.0, 1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    nb_models = len(list_of_models)
    Traj_dataset = np.zeros((int(N*nb_models), H, env.state_size))
    Action_dataset = np.zeros((int(N*nb_models), H-1, env.action_size))
    attr = np.zeros((int(N*nb_models), nb_models)) # attribute = model used for the traj
    
    # Progress tracking
    total_trajectories = nb_models * N
    pbar = tqdm(total=total_trajectories, desc=f"Generating {noise_level} dataset")
    
    for model_id in range(nb_models):
        _, state_norm, agent = load("policies/" + list_of_models[model_id])
        pbar.set_description(f"Generating {noise_level} dataset - Model: {list_of_models[model_id]}")
        
        for traj_id in range(N):
            s = s0 + 2*noise_scale*(np.random.rand(12)-0.5)
            Traj, Actions = rollout(env, agent, state_norm, s0=s, display=False)
            repeat = 0
            while Traj.shape[0] < H: # too short traj
                repeat += 1
                assert repeat < 20, "Model " + list_of_models[model_id] + " failed 20 times"
                s = s0 + 2*noise_scale*(np.random.rand(12)-0.5)#/repeat
                Traj, Actions = rollout(env, agent, state_norm, s0=s, display=False)
        
            Traj_dataset[model_id*N + traj_id] = Traj[:H]
            Action_dataset[model_id*N + traj_id] = Actions[:H-1]
            attr[model_id*N + traj_id, model_id] = 1
            
            # Update progress
            pbar.update(1)
    
    pbar.close()
    
    # Add noise level to filename for better identification
    filename = filename + f"_{noise_level}"
    
    if nb_models == 1:
        dataset = {"States": Traj_dataset, "Actions": Action_dataset, 
                  "noise_level": noise_level, "noise_scale": noise_scale}
    else:
        filename = filename + "_attr"
        dataset = {"States": Traj_dataset, "Actions": Action_dataset,
                   "Attributes": attr, "noise_level": noise_level, "noise_scale": noise_scale,
                   "info": f"Dataset with {noise_level} noise level. Attributes are 1-hot encoding for the mode " + str(list_of_models)}

    with open(filename + ".pkl", 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"Generated {noise_level} dataset: {filename}.pkl")
    print(f"Total trajectories: {len(Traj_dataset)}")
    print(f"Models: {list_of_models}")
    print(f"Noise level: {noise_level}, Noise scale: {noise_scale}")
        


def generate_progressive_datasets(env, dataset_name="multi", list_of_models=None, N=500, H=40, 
                                 noise_levels=["clean", "low", "medium", "high", "extreme"]):
    """Generate progressive noise level datasets for transfer learning.
    
    Args:
        - env: Hopper environment
        - dataset_name: custom name for dataset (default: "multi" for multi-policy)
        - list_of_models: list of policy models to use (default: all available)
        - N: number of trajectories per model
        - H: horizon length
        - noise_levels: list of noise levels to generate
    """
    
    # Default models if not specified
    if list_of_models is None:
        list_of_models = ["base", "hopping", "neg", "mid", "full", "balance"]
    
    # Generate filename
    if dataset_name == "multi" or len(list_of_models) > 1:
        nb_trajs = int(len(list_of_models) * N)
        base_filename = f"datasets/{dataset_name}_{nb_trajs}trajs_{H}steps"
    else:
        base_filename = f"datasets/{dataset_name}_{N}trajs_{H}steps"
    
    print(f"Dataset name: {dataset_name}")
    print(f"Models: {list_of_models}")
    print(f"Trajectories per model: {N}")
    print(f"Total trajectories: {nb_trajs if dataset_name == 'multi' or len(list_of_models) > 1 else N}")
    print(f"Horizon: {H} steps")
    
    noise_configs = {
        "clean": {"scale": 1e-3, "ratio": 0.0, "function": "create_dataset"},
        "low": {"scale": 1e-3, "ratio": 0.05, "function": "mixed_dataset"},
        "medium": {"scale": 2e-3, "ratio": 0.1, "function": "mixed_dataset"},
        "high": {"scale": 3e-3, "ratio": 0.2, "function": "mixed_dataset"},
        "extreme": {"scale": 5e-3, "ratio": 0.3, "function": "mixed_dataset"}
    }
    
    for level in noise_levels:
        if level not in noise_configs:
            print(f"⚠️ Unknown noise level: {level}")
            continue
            
        config = noise_configs[level]
        print(f"\n=== Generating {level.upper()} Noise Dataset ===")
        
        if config["function"] == "create_dataset":
            create_dataset(env, base_filename, list_of_models, N, H, 
                          noise_scale=config["scale"], noise_level=level)
        else:
            mixed_dataset(env, base_filename, list_of_models, N, H,
                         noise_scale=config["scale"], noise_ratio=config["ratio"], 
                         noise_level=level)
    
    print(f"\n✅ Generated {len(noise_levels)} datasets successfully!")
    print(f"Base filename: {base_filename}")


def load_dataset(filename):
    """Load a dataset of trajectories generated by pretrained models.
    Args:
        - filename: the name of the dataset to load
    Returns:
        - Traj_dataset: an array of state trajectories (N, H, state_size)
        - Action_dataset: the array of corresponding actions (N, H-1, action_size)
        - Attributes: (optional) attribute information for multimodal datasets
        - Metadata: (optional) dictionary containing noise_level, noise_scale, noise_ratio
    with N: the total number of trajectories
    and  H: the horizon, i.e., length of each trajectory """
        
    print("Loading dataset " + filename)
    with open(filename + '.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    
    # Extract metadata
    metadata = {}
    if "noise_level" in loaded_dict:
        metadata["noise_level"] = loaded_dict["noise_level"]
    if "noise_scale" in loaded_dict:
        metadata["noise_scale"] = loaded_dict["noise_scale"]
    if "noise_ratio" in loaded_dict:
        metadata["noise_ratio"] = loaded_dict["noise_ratio"]
    
    if "Attributes" in loaded_dict:
        print(loaded_dict["info"])
        if metadata:
            print(f"Dataset metadata: {metadata}")
        return loaded_dict['States'], loaded_dict['Actions'], loaded_dict["Attributes"], metadata
    else:
        if metadata:
            print(f"Dataset metadata: {metadata}")
        return loaded_dict['States'], loaded_dict['Actions'], metadata


# States, Actions = load_dataset("policies_data/neg_10trajs_300steps")
# np.savez("policies_data/Hopper_10trajs_300steps.npz", Trajs=States, Actions=Actions)

#%%
import random
# from hopper import noised_rollout



def noised_dataset(env, filename, list_of_models, N, H=40, noise_scale=1e-2, noise_ratio=0.3, noise_level="noised"):
    """Create a dataset of trajectories using pretrained models with some noised trajectories.
    Args:
        - env: the gym environment
        - filename: the name where the dataset should be stored
        - list_of_models: a list containing the name of each pretrained model
        - N: the number of trajectories per model
        - H: the horizon, i.e., length of each trajectory
        - noise_scale: minmax bound for the uniform noise on the state reset
        - noise_ratio: fraction of trajectories that should be noised (0.0 to 1.0)
        - noise_level: string identifier for noise level ("clean", "low", "medium", "high", "extreme")"""
    
    s0 = np.array([0.0, 1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    nb_models = len(list_of_models)
    Traj_dataset = np.zeros((int(N*nb_models), H, env.state_size))
    Action_dataset = np.zeros((int(N*nb_models), H-1, env.action_size))
    attr = np.zeros((int(N*nb_models), nb_models)) # attribute = model used for the traj
    N_step = max(500, H + 50) # maximal number of steps, ensure it's larger than H
    N_noised_steps = 10 # number of noised actions steps

    # Progress tracking
    total_trajectories = nb_models * N
    pbar = tqdm(total=total_trajectories, desc=f"Generating {noise_level} noised dataset")
    
    for model_id in range(nb_models):
        _, state_norm, agent = load("policies/" + list_of_models[model_id])
        pbar.set_description(f"Generating {noise_level} noised dataset - Model: {list_of_models[model_id]}")
        
        for traj_id in range(N):
            s = s0 + 2*noise_scale*(np.random.rand(12)-0.5)
            
            # Decide if this trajectory should be noised
            use_noise = np.random.rand() < noise_ratio
            
            if use_noise:
                # Generate noised trajectory
                max_t0 = max(0, N_step-N_noised_steps-H-1)
                if max_t0 <= 0:
                    # If not enough room for noise injection, skip noise for this trajectory
                    use_noise = False
                else:
                    t0 = random.randint(0, max_t0)
                Traj, Actions = noised_rollout(env, agent, state_norm, s0=s, display=False,
                                             t0=t0, t1=t0+N_noised_steps)
                
                repeat = 0
                while Traj.shape[0] < t0 + N_noised_steps + H: # too short traj
                    repeat += 1
                    assert repeat < 20, "Model " + list_of_models[model_id] + " failed 20 times"
                    s = s0 + 2*noise_scale*(np.random.rand(12)-0.5)#/repeat
                    Traj, Actions = noised_rollout(env, agent, state_norm, s0=s, display=False,
                                                 t0=t0, t1=t0+N_noised_steps)
                
                # Extract the segment after noise injection
                start_idx = min(t0+N_noised_steps, max(0, len(Traj)-H))
                end_idx = start_idx + H
                
                traj_slice = Traj[start_idx:end_idx]
                action_end_idx = min(start_idx+H, len(Actions))
                action_slice = Actions[start_idx:action_end_idx]
                
                # Pad if necessary
                if len(traj_slice) < H:
                    last_state = traj_slice[-1]
                    while len(traj_slice) < H:
                        traj_slice = np.vstack([traj_slice, last_state])
                
                if len(action_slice) < H-1:
                    last_action = action_slice[-1] if len(action_slice) > 0 else np.zeros(env.action_size)
                    while len(action_slice) < H-1:
                        action_slice = np.vstack([action_slice, last_action])
                
                Traj_dataset[model_id*N + traj_id] = traj_slice[:H]
                Action_dataset[model_id*N + traj_id] = action_slice[:H-1]
            else:
                # Generate normal trajectory
                Traj, Actions = rollout(env, agent, state_norm, s0=s, display=False)
                repeat = 0
                while Traj.shape[0] < H: # too short traj
                    repeat += 1
                    assert repeat < 20, "Model " + list_of_models[model_id] + " failed 20 times"
                    s = s0 + 2*noise_scale*(np.random.rand(12)-0.5)#/repeat
                    Traj, Actions = rollout(env, agent, state_norm, s0=s, display=False)
                
                # Use available trajectory, adjust if too short
                start_idx = min(0, max(0, len(Traj)-H))
                end_idx = start_idx + H
                
                traj_slice = Traj[start_idx:end_idx]
                action_end_idx = min(start_idx+H, len(Actions))
                action_slice = Actions[start_idx:action_end_idx]
                
                # Pad if necessary
                if len(traj_slice) < H:
                    last_state = traj_slice[-1]
                    while len(traj_slice) < H:
                        traj_slice = np.vstack([traj_slice, last_state])
                
                if len(action_slice) < H-1:
                    last_action = action_slice[-1] if len(action_slice) > 0 else np.zeros(env.action_size)
                    while len(action_slice) < H-1:
                        action_slice = np.vstack([action_slice, last_action])
                
                Traj_dataset[model_id*N + traj_id] = traj_slice[:H]
                Action_dataset[model_id*N + traj_id] = action_slice[:H-1]
            
            attr[model_id*N + traj_id, model_id] = 1
            
            # Update progress
            pbar.update(1)
    
    pbar.close()
    
    # Add noise level to filename for better identification
    filename = filename + f"_{noise_level}"
    
    if nb_models == 1:
        dataset = {"States": Traj_dataset, "Actions": Action_dataset, 
                  "noise_level": noise_level, "noise_scale": noise_scale, "noise_ratio": noise_ratio}
    else:
        filename = filename + "_attr"
        dataset = {"States": Traj_dataset, "Actions": Action_dataset,
                   "Attributes": attr, "noise_level": noise_level, "noise_scale": noise_scale, "noise_ratio": noise_ratio,
                   "info": f"Noised dataset with {noise_level} noise level, {noise_ratio*100:.0f}% noised trajectories. Attributes are 1-hot encoding for the mode " + str(list_of_models)}

    with open(filename + ".pkl", 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"Generated {noise_level} noised dataset: {filename}.pkl")
    print(f"Total trajectories: {len(Traj_dataset)}")
    print(f"Models: {list_of_models}")
    print(f"Noise level: {noise_level}, Noise scale: {noise_scale}, Noise ratio: {noise_ratio*100:.0f}%")


def mixed_dataset(env, filename, list_of_models, N, H=40, noise_scale=1e-2, noise_ratio=0.3, noise_level="mixed"):
    """Create a mixed dataset of trajectories using pretrained models with both regular and noised trajectories.
    Args:
        - env: the gym environment
        - filename: the name where the dataset should be stored
        - list_of_models: a list containing the name of each pretrained model
        - N: the number of trajectories per model
        - H: the horizon, i.e., length of each trajectory
        - noise_scale: minmax bound for the uniform noise on the state reset
        - noise_ratio: fraction of trajectories that should be noised (0.0 to 1.0)
        - noise_level: string identifier for noise level ("clean", "low", "medium", "high", "extreme")"""
    
    s0 = np.array([0.0, 1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    nb_models = len(list_of_models)
    Traj_dataset = np.zeros((int(N*nb_models), H, env.state_size))
    Action_dataset = np.zeros((int(N*nb_models), H-1, env.action_size))
    attr = np.zeros((int(N*nb_models), nb_models)) # attribute = model used for the traj
    N_step = max(500, H + 50) # maximal number of steps, ensure it's larger than H
    N_noised_steps = 10 # number of noised actions steps

    # Progress tracking
    total_trajectories = nb_models * N
    pbar = tqdm(total=total_trajectories, desc=f"Generating {noise_level} noised dataset")
    
    for model_id in range(nb_models):
        _, state_norm, agent = load("policies/" + list_of_models[model_id])
        pbar.set_description(f"Generating {noise_level} noised dataset - Model: {list_of_models[model_id]}")
        
        for traj_id in range(N):
            s = s0 + 2*noise_scale*(np.random.rand(12)-0.5)
            
            # Decide if this trajectory should be noised
            use_noise = np.random.rand() < noise_ratio
            
            if use_noise:
                # Generate noised trajectory
                max_t0 = max(0, N_step-N_noised_steps-H-1)
                if max_t0 <= 0:
                    # If not enough room for noise injection, skip noise for this trajectory
                    use_noise = False
                else:
                    t0 = random.randint(0, max_t0)
                Traj, Actions = noised_rollout(env, agent, state_norm, s0=s, display=False,
                                             t0=t0, t1=t0+N_noised_steps)
                
                repeat = 0
                while Traj.shape[0] < t0 + N_noised_steps + H: # too short traj
                    repeat += 1
                    assert repeat < 20, "Model " + list_of_models[model_id] + " failed 20 times"
                    s = s0 + 2*noise_scale*(np.random.rand(12)-0.5)#/repeat
                    Traj, Actions = noised_rollout(env, agent, state_norm, s0=s, display=False,
                                                 t0=t0, t1=t0+N_noised_steps)
                
                # Extract the segment after noise injection
                start_idx = min(t0+N_noised_steps, max(0, len(Traj)-H))
                end_idx = start_idx + H
                
                traj_slice = Traj[start_idx:end_idx]
                action_end_idx = min(start_idx+H, len(Actions))
                action_slice = Actions[start_idx:action_end_idx]
                
                # Pad if necessary
                if len(traj_slice) < H:
                    last_state = traj_slice[-1]
                    while len(traj_slice) < H:
                        traj_slice = np.vstack([traj_slice, last_state])
                
                if len(action_slice) < H-1:
                    last_action = action_slice[-1] if len(action_slice) > 0 else np.zeros(env.action_size)
                    while len(action_slice) < H-1:
                        action_slice = np.vstack([action_slice, last_action])
                
                Traj_dataset[model_id*N + traj_id] = traj_slice[:H]
                Action_dataset[model_id*N + traj_id] = action_slice[:H-1]
            else:
                # Generate normal trajectory
                Traj, Actions = rollout(env, agent, state_norm, s0=s, display=False)
                repeat = 0
                while Traj.shape[0] < H: # too short traj
                    repeat += 1
                    assert repeat < 20, "Model " + list_of_models[0] + " failed 20 times"
                    s = s0 + 2*noise_scale*(np.random.rand(12)-0.5)#/repeat
                    Traj, Actions = rollout(env, agent, state_norm, s0=s, display=False)
                
                # Use available trajectory, adjust if too short
                start_idx = min(0, max(0, len(Traj)-H))
                end_idx = start_idx + H
                
                traj_slice = Traj[start_idx:end_idx]
                action_end_idx = min(start_idx+H, len(Actions))
                action_slice = Actions[start_idx:action_end_idx]
                
                # Pad if necessary
                if len(traj_slice) < H:
                    last_state = traj_slice[-1]
                    while len(traj_slice) < H:
                        traj_slice = np.vstack([traj_slice, last_state])
                
                if len(action_slice) < H-1:
                    last_action = action_slice[-1] if len(action_slice) > 0 else np.zeros(env.action_size)
                    while len(action_slice) < H-1:
                        action_slice = np.vstack([action_slice, last_action])
                
                Traj_dataset[model_id*N + traj_id] = traj_slice[:H]
                Action_dataset[model_id*N + traj_id] = action_slice[:H-1]
            
            attr[model_id*N + traj_id, model_id] = 1
            
            # Update progress bar
            pbar.update(1)
    
    # Close progress bar
    pbar.close()
    
    # Always add attributes for mixed dataset
    # Add noise level and ratio to filename for better identification
    noise_percentage = int(noise_ratio * 100)
    filename = filename + f"_{noise_level}_noise{noise_percentage}pct_mixed_attr"
    dataset = {"States": Traj_dataset, "Actions": Action_dataset,
               "Attributes": attr, "noise_level": noise_level, "noise_scale": noise_scale, "noise_ratio": noise_ratio,
               "info": f"Mixed dataset with {noise_level} noise level, {noise_ratio*100:.0f}% noised trajectories. Attributes are 1-hot encoding for the mode " + str(list_of_models)}

    with open(filename + ".pkl", 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"Generated {noise_level} mixed dataset: {filename}.pkl")
    print(f"Total trajectories: {len(Traj_dataset)}")
    print(f"Models: {list_of_models}")
    print(f"Noise level: {noise_level}, Noise scale: {noise_scale}, Noise ratio: {noise_ratio*100:.0f}%")










# assert 0 > 1

#%% Testing and generating the dataset

if __name__ == "__main__":
    
    from hopper import HopperEnv
    from plots import traj_comparison
    
    env = HopperEnv()

    # Comment out testing section to avoid plotting issues
    # list_of_models = ["full"] # "mid"
    # for model_name in list_of_models:
    #     _, state_norm, agent = load("policies/" + model_name)
    #     Traj, Actions = rollout(env, agent, state_norm, display=True, title=model_name)
    #     t0 = random.randint(0, 200)
    #     print("t0:", t0)
    #     Traj_noised, Actions_noised = noised_rollout(env, agent, state_norm, display=True, title=model_name + "_noised", t0=t0, t1=t0+10)
    #     print(Traj.shape, Actions.shape)
    #     print(Traj_noised.shape, Actions_noised.shape)



    #%%
    # Dataset generation parameters
    N = 250 # number of trajectories per model
    H = 300 # horizon of the trajectories (recommended for hopper: covers full hop cycle)
    
    # Choose which datasets to generate - multiple options:
    
    # OPTION 1: Generate just the CLEAN dataset (baseline) with custom name
    print("=== Generating LOW noise Dataset ===")
    generate_progressive_datasets(env, dataset_name="multi", noise_levels=["low"], N=N, H=H)
    
    # OPTION 2: Generate all progressive datasets at once
    # generate_progressive_datasets(env, dataset_name="multi", N=N, H=H)
    
    # OPTION 3: Generate specific noise levels only
    # generate_progressive_datasets(env, dataset_name="multi", 
    #                              noise_levels=["low", "medium", "high", "extreme"], N=N, H=H)
    
    # OPTION 4: Generate datasets with custom names
    # generate_progressive_datasets(env, dataset_name="hopper_robust", 
    #                              list_of_models=["base", "full"], N=N, H=H)
    
    # OPTION 5: Generate single policy dataset
    # generate_progressive_datasets(env, dataset_name="base_only", 
    #                              list_of_models=["base"], N=N, H=H)
    
    print("\n=== Dataset Generation Complete ===")
    print("Available options:")
    print("1. Use generate_progressive_datasets() with custom dataset_name")
    print("2. Specify list_of_models for specific policies")
    print("3. Adjust N and H parameters as needed")
    
#%% Multimodal dataset with attributes

# Traj_dataset, Action_dataset, N, H = load_dataset("datasets/base_full_neg_1500trajs_300steps")
# attr = np.zeros((1500, 3))
# attr[:500, 0] += 1
# attr[500:1000, 1] += 1
# attr[1000:, 2] += 1

# dataset = {"States": Traj_dataset, "Actions": Action_dataset,
#            "Attributes": attr, "info": "Attributes are 1-hot encoding for the mode " + str(list_of_models)}

# with open("datasets/base_full_neg_1500trajs_300steps_attr.pkl", 'wb') as f:
#     pickle.dump(dataset, f)
    
#%% Making datasets truly unimodal
import copy
import matplotlib.pyplot as plt

# N = 1000
# H = 300
# name = "base"
# Trajs, Actions = load_dataset("datasets/" + name + f"_{N}trajs_{H}steps_5e-3")
# Trajs, Actions = load_dataset(filename)

# max_top_angle = copy.deepcopy(Trajs[0, :, 2])
# min_top_angle = copy.deepcopy(Trajs[0, :, 2])

# for i in range(1, N):
#     top_angle = copy.deepcopy(Trajs[i, :, 2])
    
#     idx_above = np.arange(H)[top_angle > max_top_angle]
#     max_top_angle[idx_above] = copy.deepcopy(top_angle[idx_above])
    
#     idx_below = np.arange(H)[top_angle < min_top_angle]
#     min_top_angle[idx_below] = copy.deepcopy(top_angle[idx_below])


# time = np.arange(H)*env.dt
# ax = plt.gcf().gca()
# ax.set_axis_off()
# plt.plot(time, min_top_angle, label="min")
# plt.plot(time, max_top_angle, label="max")
# plt.legend(frameon=False)
# plt.title(f"Top angle in dataset {name}")
# plt.show()


# max_top_angle[0], min_top_angle[0]

# traj_comparison(env, Trajs[0], "0", Trajs[1], "1", traj_3=Trajs[2], label_3="2",
#                 traj_4=Trajs[3], label_4="3", plot_height=False)
