# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 07:52:25 2023

@author: Jean-Baptiste Bouvier

Function utils for the PPO applied to the Hopper environmnent.
"""

import copy
import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




#%% Training utils

class ReplayBuffer:
    """Store experiences. The sampling from this buffer is handled by the update function
    of PPO, where each experience is sampled as part of mini-batches."""
    def __init__(self, args):
        self.s = np.zeros((args.batch_size, args.state_dim))
        self.a = np.zeros((args.batch_size, args.action_dim))
        self.a_logprob = np.zeros((args.batch_size, args.action_dim))
        self.r = np.zeros((args.batch_size, 1))
        self.s_ = np.zeros((args.batch_size, args.state_dim))
        self.dw = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.count = 0

    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s[:self.count], dtype=torch.float)
        a = torch.tensor(self.a[:self.count], dtype=torch.float)
        a_logprob = torch.tensor(self.a_logprob[:self.count], dtype=torch.float)
        r = torch.tensor(self.r[:self.count], dtype=torch.float)
        s_ = torch.tensor(self.s_[:self.count], dtype=torch.float)
        dw = torch.tensor(self.dw[:self.count], dtype=torch.float)
        done = torch.tensor(self.done[:self.count], dtype=torch.float)

        return s, a, a_logprob, r, s_, dw, done


def evaluate_policy(args, env, agent, state_norm):
    """Evaluates the policy, returns the average reward and whether the
    reward threshold has been met"""
    
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.evaluate(s, state_norm)  # deterministic policy for evaluating
            s, r, done = env.step(action)
            episode_reward += r
        evaluate_reward += episode_reward

    average_reward = evaluate_reward / times
    return average_reward, average_reward >= env.reward_threshold







def training(args, env, agent, replay_buffer, env_eval, state_norm=None, reward_scaling=None,
             saveas=None):
    """Training of the PPO agent to achieve high reward"""
    
    evaluate_num = 0  # Record the number of evaluations
    if hasattr(args, 'total_steps'):
        total_steps = args.total_steps
    else:
        total_steps = 0  # Record the total steps during the training
    trained = False 
    episode = 0
    assert args.use_reward_scaling
    Ep_rewards = []
    max_reward = -np.inf
    
    while (not trained):
        episode += 1
        s = env.reset()
        
        reward_scaling.reset()
        episode_reward = 0
        done = False
        while not done:
            a, a_logprob = agent.choose_action(s, state_norm)  # Action and the corresponding log probability
            s_, r, done = env.step(a)
            episode_reward += r

            r = reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, done will be True, we need to distinguish them;
            # dw means dead or win, there is no next state s'
            # but when reaching the max_episode_steps, there is a next state s' actually.
            if done and env.episode_step != env.max_episode_steps:
                dw = True
            else:
                dw = False

            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = s_
            total_steps += 1

            # When the number of transitions in buffer reaches batch_size, then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps, state_norm)
                replay_buffer.count = 0

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward, trained = evaluate_policy(args, env_eval, agent, state_norm)
                print(f"evaluation {evaluate_num} \t reward: {evaluate_reward:.1f}")
                plot_rewards(Ep_rewards)
                if evaluate_reward > max_reward:
                    max_reward = evaluate_reward
                    if saveas is not None:
                        save(saveas+f"_eval_{evaluate_num}", args, state_norm, agent)
            
            trained = total_steps > args.max_train_steps or trained
            
        Ep_rewards.append(episode_reward)

    args.total_steps = total_steps
    return trained


def nice_plot():
    """Makes the plot nice"""
    fig = plt.gcf()
    ax = fig.gca()
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['font.sans-serif'] = ['Palatino Linotype']
    ax.spines['bottom'].set_color('w')
    ax.spines['top'].set_color('w') 
    ax.spines['right'].set_color('w')
    ax.spines['left'].set_color('w')
    
    return fig, ax
      
def plot_rewards(reward_list):
    """Plots the rewards during training given a list of episode rewards"""
    fig, ax = nice_plot()
    plt.title("Rewards")
    plt.plot(np.arange(len(reward_list)), reward_list)
    plt.xlabel("episodes")
    plt.show()





    
    






#%% Utils to save and load the whole framework


import json
import argparse
from PPO import PPO
from normalization import Normalization


 

def save(filename, args, state_norm, agent):
    """Saving the arguments, state_norm and controller"""
    agent.save(filename)
    args_2 = copy.deepcopy(args)
    
    ### Convert arrays into lists  as arrays cannot be stored in json
    args_2.state_norm_n = state_norm.running_ms.n
    args_2.state_norm_mean = state_norm.running_ms.mean.tolist()
    args_2.state_norm_S = state_norm.running_ms.S.tolist()
    args_2.state_norm_std = state_norm.running_ms.std.tolist()
    args_2.max_action = args_2.max_action.tolist()
    
    with open(filename + '_args.txt', 'w') as f:
        json.dump(args_2.__dict__, f, indent=2)



def load(filename):
    """Loads the arguments, state_norm and controller"""
    
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    args = parser.parse_args()
    with open(filename + '_args.txt', 'r') as f:
        args.__dict__ = json.load(f)
    
    ### Convert lists back to arrays  as arrays cannot be stored in json
    args.max_action = np.array(args.max_action)
    state_norm = Normalization(shape=args.state_dim)  
    state_norm.running_ms.n = args.state_norm_n
    state_norm.running_ms.mean = np.array(args.state_norm_mean)
    state_norm.running_ms.S = np.array(args.state_norm_S)
    state_norm.running_ms.std = np.array(args.state_norm_std)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    agent = PPO(args)
    agent.load(filename)
        
    return args, state_norm, agent





#%% Extra 

def bump(x, low, high, max_y=1.):
    """bump function(x) that is 0 for x < low and x > high
    reaches max_y for x = (high+low)/2"""
    if low > high: # exchange
        low, high = high, low
    if x < low or x > high:
        return 0.
    mid = (low + high)/2
    if x < mid:
        return (x - low)*max_y/(mid - low)
    return (high - x)*max_y/(high - mid)
    
    



