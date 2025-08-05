# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:15:08 2024

@author: Jean-Baptiste Bouvier

Main file to train the Hopper to move forward
Environment has 12 states and 3 actions
Once trained, the policy can generate base trajectories for the diffusion     

PPO implementation from
https://github.com/Lizhi-sjtu/DRL-code-pytorch/tree/main/5.PPO-continuous

Several policies of the Hopper have been trained, saved, and can be loaded:
    base: 12 states with x-position (top angle stays positive)
    hopping: hops faster
    balance: stays in place balancing
    neg: top angle stays negative
    mid: top angle oscillate around 0
    full: top angle oscillate in its full range -10 to +10 deg

"""

import torch
import argparse
import numpy as np

from PPO import PPO
from hopper import HopperEnv, plot_traj, rollout

from ppo_utils import ReplayBuffer, training, load, save
from normalization import Normalization, RewardScaling


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Hyperparameters

loading = True
# loading = False

parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
parser.add_argument("--max_train_steps", type=int, default=int(5e6), help="Maximum number of training steps")
parser.add_argument("--evaluate_freq", type=float, default=5e4, help="Evaluate the policy every 'evaluate_freq' steps")
parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Gaussian") # or Beta")
parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
parser.add_argument("--mini_batch_size", type=int, default=32, help="Minibatch size")
parser.add_argument("--hidden_width", type=int, default=128, help="The number of neurons in hidden layers of the neural network")
parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2: state normalization")
parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4: reward scaling")
parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6: learning rate Decay")
parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
parser.add_argument("--set_adam_eps", type=bool, default=True, help="Trick 9: set Adam epsilon=1e-5")
parser.add_argument("--seed", type=int, default=10, help="Common seed for all environments")
args = parser.parse_args()

#### Hopper
# filename = "saved/base" # 12 states with x-position (top angle stays positive)
# filename = "saved/hopping" # hops faster
# filename = "saved/balance" # stays in place balancing
filename = "saved/neg" # top angle stays negative
# filename = "saved/mid" # top angle oscillate around 0
# filename = "saved/full" # top angle oscillate in its full range : -10 to +10 deg

### Set random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

### Environment
env = HopperEnv()
env_eval = HopperEnv()


### Add new constants to the arguments
args.state_dim = env.state_size
args.action_dim = env.action_size
args.max_action = env.action_max

assert args.use_reward_scaling
reward_scaling = RewardScaling(shape=1, gamma=args.gamma)



#%% Training the hopper

if loading:
    args, state_norm, agent = load(filename)
    
else:
    agent = PPO(args) # new agent
    state_norm = Normalization(shape=args.state_dim)
    
    replay_buffer = ReplayBuffer(args)
    trained = training(args, env, agent, replay_buffer, env_eval, state_norm, reward_scaling,
                       saveas=filename)

    #%% Saving trained agent
    
    save(filename, args, state_norm, agent)
    # save("saved/baseline", args, state_norm, agent)


#%% Testing

Traj, Actions = rollout(env, agent, state_norm)

#%% Visualisation

env = HopperEnv(render_mode="human")
state = env.reset()

episode_reward = 0.
Trajectory = np.zeros((env.max_episode_steps, env.state_size))
Trajectory[0] = state
Actions = np.zeros((env.max_episode_steps, env.action_size))

for t in range(env.max_episode_steps):
    with torch.no_grad():
        action = agent.evaluate(state, state_norm)
    state, reward, done = env.step(action)
    episode_reward += reward
    
    Trajectory[t] = state
    Actions[t-1] = action
    
    env.render()
    if done: break

print(f"Average reward: {episode_reward/env.max_episode_steps:.3f}")
plot_traj(env, Trajectory[:t+1], Actions[:t], "Visualisation", plot_all=True)

env.close()

#%%
# state, reward, done = env.step(action)





