# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:27:02 2024

@author: Jean-Baptiste

Taken from https://github.com/ZibinDong/AlignDiff-ICLR2024/blob/main/utils/dit_utils.py
Simplified, removing masks and attributes since those are only used to condition
the output based on human feedback
Prediction of the states and actions together

Closed-loop model
"""

import os
import time
import math
import torch
import einops
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
from typing import Optional
import matplotlib.pyplot as plt

# hidden_size = 384 | 768 | 1024 | 1152
# depth =       12  | 24  | 28
# patch_size =  2   | 4   | 8
# n_heads =     6   | 12  | 16  (hidden_size can be divided by n_heads)



#%% Diffusion Transformer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



class ContinuousCondEmbedder(nn.Module):
    """Modified from DiscreteCondEmbedder to embed the initial state,
    a continuous variable instead of a 1-hot vector
    The embedding transforms the discrete 1-hot into a continuous vector, don't need that here.
    Just a regular affine layer to make the initial state of the right dimension."""
    
    def __init__(self, attr_dim: int, hidden_size: int):
        super().__init__()
        self.attr_dim = attr_dim
        self.embedding = nn.Linear(attr_dim, int(attr_dim*128)) # 1 layer affine to transform initial state into embedding vector
        # self.embedding = nn.Sequential(nn.Linear(attr_dim, 128), nn.ReLU(),
        #                                nn.Linear(128, 128)) # 2 layers embedding
        self.attn = nn.MultiheadAttention(128, num_heads=2, batch_first=True)
        self.linear = nn.Linear(128 * attr_dim, hidden_size)
    
    def forward(self, attr: torch.Tensor, mask: torch.Tensor = None):
        '''
        attr: (batch_size, attr_dim)
        mask: (batch_size, attr_dim) 0 or 1, 0 means ignoring
        '''
        emb = self.embedding(attr).reshape((-1, self.attr_dim, 128)) # (b, attr_dim, 128)
        if mask is not None: emb *= mask.unsqueeze(-1) # (b, attr_dim, 128)
        emb, _ = self.attn(emb, emb, emb) # (b, attr_dim, 128)
        return self.linear(einops.rearrange(emb, 'b c d -> b (c d)')) # (b, hidden_size)








def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, dim), nn.Mish(), nn.Linear(dim, dim))
    def forward(self, x: torch.Tensor):
        return self.mlp(x)


class DiTBlock(nn.Module):
    """ A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning. """
    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4), approx_gelu(), nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6))
        
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=1)
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x,x,x)[0]
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Finallayer1d(nn.Module):
    def __init__(self, hidden_size: int, out_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)

    
   
class DiT1d(nn.Module):
    def __init__(self, x_dim: int, attr_dim: int, d_model: int = 384, 
                 n_heads: int = 6, depth: int = 12, dropout: float = 0.1):
        super().__init__()
        self.attr_dim = attr_dim # dimension of the attributes
        self.x_dim, self.d_model, self.n_heads, self.depth = x_dim, d_model, n_heads, depth
        self.x_proj = nn.Linear(x_dim, d_model)
        self.t_emb = TimeEmbedding(d_model)
        if attr_dim > 0:
            self.attr_proj = ContinuousCondEmbedder(attr_dim, d_model)
        self.pos_emb = SinusoidalPosEmb(d_model)
        self.pos_emb_cache = None
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, n_heads, dropout) for _ in range(depth)])
        self.final_layer = Finallayer1d(d_model, x_dim)
        self.initialize_weights()
        
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_emb.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor,
                attr: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        '''
        Input:  x: (batch, horizon, x_dim)     t:  (batch, 1)
             attr: (batch, attr_dim)         mask: (batch, attr_dim)
        
        Output: y: (batch, horizon, x_dim)
        '''
        if self.pos_emb_cache is None or self.pos_emb_cache.shape[0] != x.shape[1]:
            self.pos_emb_cache = self.pos_emb(torch.arange(x.shape[1], device=x.device))
        x = self.x_proj(x) + self.pos_emb_cache[None,]
        t = self.t_emb(t)
        if attr is not None:
            t += self.attr_proj(attr, mask)
        for block in self.blocks:
            x = block(x, t)
        x = self.final_layer(x, t)
        return x
    
    
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
    
    
    
    
#%% ODE predicting both states and actions given the initial state


class SA_ODE():
    def __init__(self, env, sigma_data: float, attr_dim:int = None,
                 horizon:int = 16,
        sigma_min: float = 0.002, sigma_max: float = 80,
        rho: float = 7, p_mean: float = -1.2, p_std: float = 1.2, 
        d_model: int = 384, n_heads: int = 6, depth: int = 12,
        device: str = "cpu", N: int = 5, projector = None):
        """Takes states to predict states and actions
        Diffusion trained according to EDM: "Elucidating the Design Space of Diffusion-Based Generative Models"
        """
        
        
        self.is_conditional = (attr_dim is not None) and type(attr_dim) == int
        if self.is_conditional:
            self.attr_dim = attr_dim
            assert attr_dim == env.state_size
        else:
            self.attr_dim = 0
        assert type(horizon) == int and horizon < 299 and horizon > 2
        self.horizon = horizon
        self.projector = projector
        if projector is None:
            self.projector_name = ""
        else:
            self.projector_name = projector.name
        self.task = env.name
        self.specs = f"{d_model}_{n_heads}_{depth}"
        self.filename = "SA_ODE_" + self.is_conditional*"Cond_" + self.task + "_" + self.projector_name + "_specs_" + self.specs + f"_h_{horizon}"
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.x_dim = self.state_size + self.action_size # predict state and action together
        self.sigma_data, self.sigma_min, self.sigma_max = sigma_data, sigma_min, sigma_max
        self.rho, self.p_mean, self.p_std = rho, p_mean, p_std
        
        self.device = device
        self.F = DiT1d(self.x_dim, d_model=d_model, attr_dim=self.attr_dim,
                       n_heads=n_heads, depth=depth, dropout=0.1).to(device)
        self.F.train()
        # Exponential Moving Average (ema)
        self.F_ema = deepcopy(self.F).requires_grad_(False)
        self.F_ema.eval()
        self.optim = torch.optim.AdamW(self.F.parameters(), lr=2e-4, weight_decay=1e-4)
        self.set_N(N) # number of noise scales
        # Bins of sigma to split the training loss
        self.sigma_bins = self.p_mean + self.p_std * torch.tensor([-1.2, -0.6, -0.15, 0.15, 0.6, 1.2])
        self.sigma_bins = self.sigma_bins.exp() # exp in sample_noise_distribution
        print(f'Initialized State Action ODE model with {count_parameters(self.F)} parameters.')
        
    def ema_update(self, decay=0.999):
        for p, p_ema in zip(self.F.parameters(), self.F_ema.parameters()):
            p_ema.data = decay*p_ema.data + (1-decay)*p.data

    def set_N(self, N):
        self.N = N
        self.sigma_s = (self.sigma_max**(1/self.rho)+torch.arange(N, device=self.device)/(N-1)*\
            (self.sigma_min**(1/self.rho)-self.sigma_max**(1/self.rho)))**self.rho
        self.t_s = self.sigma_s
        self.scale_s = torch.ones_like(self.sigma_s) * 1.0
        self.dot_sigma_s = torch.ones_like(self.sigma_s) * 1.0
        self.dot_scale_s = torch.zeros_like(self.sigma_s)
        if self.t_s is not None:
            self.coeff1 = (self.dot_sigma_s/self.sigma_s + self.dot_scale_s/self.scale_s)
            self.coeff2 = self.dot_sigma_s/self.sigma_s*self.scale_s
            
    def c_skip(self, sigma): return self.sigma_data**2/(self.sigma_data**2+sigma**2)
    def c_out(self, sigma): return sigma*self.sigma_data/(self.sigma_data**2+sigma**2).sqrt()
    def c_in(self, sigma): return 1/(self.sigma_data**2+sigma**2).sqrt()
    def c_noise(self, sigma): return 0.25*(sigma).log()
    def loss_weighting(self, sigma): return (self.sigma_data**2+sigma**2)/((sigma*self.sigma_data)**2)
    def sample_noise_distribution(self, N):
        log_sigma = torch.randn((N,1,1),device=self.device)*self.p_std + self.p_mean
        return log_sigma.exp()
    
    def D(self, x, sigma, condition = None, mask = None, use_ema = False):
        c_skip, c_out, c_in, c_noise = self.c_skip(sigma), self.c_out(sigma), self.c_in(sigma), self.c_noise(sigma)
        F = self.F_ema if use_ema else self.F
        return c_skip*x + c_out*F(c_in*x, c_noise.squeeze(-1), condition, mask)
     
    
    def update(self, x, condition = None):
        """Updates the DiT module given a trajectory batch x: (batch, horizon, state_size + action_size) """
        
        sigma = self.sample_noise_distribution(x.shape[0])
        eps = torch.randn_like(x) * sigma
        eps[:, 0, :self.state_size] = 0. # preserve the first state observation since given
        loss_mask = torch.ones_like(x)
        loss_mask[:, 0, :self.state_size] = 0. # no loss on the first state since constant
        loss_mask[:, 0, self.state_size:] = 10. # higher coefficient for the first action
        
        if condition is None:
            mask = None
        else:
            mask = (torch.rand(*condition.shape, device=self.device) > 0.2).int()
        
        pred = self.D(x + eps, sigma, condition, mask)
        
        if self.projector is not None:
            self.nb_exact_proj += (sigma < self.projector.sigma_min).sum().item()
            
            pred_s = pred[:, :, :self.state_size] # predicted states
            pred_a = pred[:, :, self.state_size:] # predicted actions
            ref_s = x[:, :, :self.state_size] # true states
            ref_a = x[:, :, self.state_size:] # true actions
            
            if "reference" in self.projector.name:
                if self.train_with_ref:
                    s, a = self.projector.project_traj(Trajs=pred_s, Ref_Trajs=ref_s, sigma=sigma, Actions=ref_a) 
                else: # train using only the prediction data, like at inference
                    s, a = self.projector.project_traj(Trajs=pred_s, Ref_Trajs=pred_s, sigma=sigma, Actions=pred_a) 
            else:
                s, a = self.projector.project_traj(Trajs=pred_s, sigma=sigma, Actions=pred_a)
            pred = torch.cat((s, a), dim=2)
            
        loss = (loss_mask * self.loss_weighting(sigma) * (pred - x)**2).mean(dim=(1,2))
        
        ### Adding the loss of each prediction to the corresponding sigma_bin
        sigma = sigma.squeeze()
        L = loss[sigma < self.sigma_bins[0]].detach().mean().item()
        self.training_loss[0].append(L)
        for i in range(1, self.sigma_bins.shape[0]):
            L = loss[(self.sigma_bins[i-1] <= sigma)*(sigma < self.sigma_bins[i])].detach().mean().item()
            self.training_loss[i].append(L)
        L = loss[sigma >= self.sigma_bins[-1]].detach().mean().item()
        self.training_loss[-1].append(L)
        
        loss = loss.mean()
        self.optim.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.F.parameters(), 10.)
        self.optim.step()
        self.ema_update()
        return loss.item(), grad_norm.item()
    
    
    def train(self, x_normalized:torch.Tensor, n_gradient_steps:int, batch_size:int = 32,
              extra:str="", time_limit=None):
        """Trains the DiT module from NORMALIZED data x_normalized: (nb trajs, horizon, state_size + action_size)
        time_limit in seconds
        Can be conditioned on initial state
        """
        assert len(x_normalized.shape) == 3 and x_normalized.shape[-1] == self.state_size + self.action_size
        print('Begins training of the Diffusion Transformer ' + self.filename + extra)
        if time_limit is not None:
            t0 = time.time()
            print(f"Training limited to {time_limit:.0f}s")
        
        N_trajs = x_normalized.shape[0]    
        loss_avg = 0.
        self.nb_exact_proj = 0
        self.training_loss = [[]]
        for _ in range(self.sigma_bins.shape[0]):
            self.training_loss.append([])
        
        pbar = tqdm(range(n_gradient_steps))
        for step in range(n_gradient_steps):
            if self.projector is not None:
                sigma_low = max(0.1*self.projector.sigma_max*(1 - 2*step/n_gradient_steps), self.sigma_min)
                self.p_mean = (np.log(self.sigma_max) + np.log(sigma_low))/2
                self.p_std = (np.log(self.sigma_max) - np.log(sigma_low))/10 # 5 sigmas on each sides of p_mean
            
            idx = np.random.randint(0, N_trajs, batch_size) # sample a random batch of trajectories
            start_idx = np.random.randint(0, x_normalized.shape[1]-self.horizon, batch_size) # sample starting timesteps
            x = torch.zeros((batch_size, self.horizon, self.x_dim)).to(self.device)
            for traj_id in range(batch_size):
                x[traj_id] = x_normalized[ idx[traj_id], start_idx[traj_id]:start_idx[traj_id] + self.horizon].clone()
                
            if self.is_conditional: # condition on initial state
                attr = x[:, 0, :self.state_size] # batch, state_size
                loss, grad_norm = self.update(x, attr)
            else:
                loss, grad_norm = self.update(x)
                
            loss_avg += loss
            if (step+1) % 10 == 0:
                pbar.set_description(f'step: {step+1} loss: {loss_avg / 10.:.4f} grad_norm: {grad_norm:.4f}     {100*self.nb_exact_proj/(step*batch_size):.2f}% of exact projections   ')
                pbar.update(10)
                loss_avg = 0.
                self.save(extra)
                if time_limit is not None and time.time() - t0 > time_limit:
                    print(f"Time limit reached at {time.time() - t0:.0f}s")
                    break
                
        print('\nTraining completed!')
        print(f"{100*self.nb_exact_proj/(n_gradient_steps*batch_size):.2f}% = {self.nb_exact_proj} of exact projections")
        
        for i in range(len(self.training_loss)):
            fig = plt.gcf()
            ax = fig.gca()
            plt.rcParams.update({'font.size': 16})
            ax.spines['bottom'].set_color('w')
            ax.spines['top'].set_color('w') 
            ax.spines['right'].set_color('w')
            ax.spines['left'].set_color('w')
            if i == 0:
                plt.plot(self.training_loss[i], label=f"sigma < {self.sigma_bins[0]:.3f}")
            elif i == len(self.training_loss)-1:
                plt.plot(self.training_loss[i], label=f"{self.sigma_bins[-1]:.3f} < sigma")
            else:
                plt.plot(self.training_loss[i], label=f"{self.sigma_bins[i-1]:.3f} < sigma < {self.sigma_bins[i]:.3f}")
            plt.title(self.filename)
            plt.ylabel("Training loss")
            plt.legend(frameon=False)
            plt.savefig("fig/" + self.filename + f"_{i}.png")
            plt.show()
            plt.clf()
        
    
    @torch.no_grad()
    def sample(self, s0:torch.Tensor, n_samples: int, projector = None, 
               N:int = None):
        """Samples 'n_samples' trajectories of starting from
        initial state s0.  Can project or not on the reachable set
        1st order sampling from the EDM paper"""
        if N is not None and N != self.N: self.set_N(N)
        x = torch.randn((n_samples, self.horizon, self.x_dim), device=self.device) * self.sigma_s[0] * self.scale_s[0]
        x[:, 0, :self.state_size] = s0
        
        if self.is_conditional:
            w = 1.5
            attr = s0.clone()
            attr_mask = torch.ones_like(attr) # Is it the right mask?
            attr = attr.repeat(2, 1)
            attr_mask = attr_mask.repeat(2, 1)
            attr_mask[n_samples:] = 0
        
        for i in range(self.N):
            with torch.no_grad():
                if self.is_conditional:
                    D = self.D(x.repeat(2,1,1)/self.scale_s[i], torch.ones((2*n_samples,1,1),device=self.device)*self.sigma_s[i], attr, attr_mask, use_ema=True)
                    D = w*D[:n_samples] + (1-w)*D[n_samples:]
                else:
                    D = self.D(x/self.scale_s[i], torch.ones((n_samples,1,1),device=self.device)*self.sigma_s[i], use_ema=True)  
            d = self.coeff1[i] * x - self.coeff2[i] * D
            
            if i == self.N-1:
                dt = 0 - self.t_s[i]
            else:
                dt = self.t_s[i+1] - self.t_s[i]
                
            x = x + dt * d
            x[:, 0, :self.state_size] = s0 # at each denoising step, reset the initial state to s0
           
            if projector is not None: # project the rest of the trajectory onto its admissible space
                sigma = self.sigma_s[i]*torch.ones((x.shape[0],1))
                s = x[:, :, :self.state_size] # sampled states
                a = x[:, :, self.state_size:] # sampled actions
                if "reference" in projector.name:
                    # use the sampled traj as reference after training to minimize deviations from projections
                    s, a = projector.project_traj(Trajs=s, Ref_Trajs=s, sigma=sigma, Actions=a)
                else:
                    s, a = projector.project_traj(Trajs=s, sigma=sigma, Actions=a)
                x = torch.cat((s, a), dim=2)
        
        return x
    
    
    def save(self, extra:str = ""):
        torch.save({'model': self.F.state_dict(),
                    'model_ema': self.F_ema.state_dict()},
                   "trained/"+ self.filename+extra+".pt")
        
        
    def load(self, extra:str = ""):
        name = "trained/" + self.filename + extra + ".pt"
        if os.path.isfile(name):
            print("Loading " + name)
            checkpoint = torch.load(name, map_location=self.device, weights_only=True)
            self.F.load_state_dict(checkpoint['model'])
            self.F_ema.load_state_dict(checkpoint['model_ema'])
            return True # loaded
        else:
            print("File " + name + " doesn't exist. Not loading anything.")
            return False # not loaded


    def update_projector(self, projector):
        """Upade the projector, changes the filename for saving/loading""" 
        self.projector = projector
        self.projector_name = projector.name
        self.filename = "SA_ODE_" + self.task + "_" + self.projector_name + "_specs_" + self.specs + f"_h_{self.horizon}"



#%%

def relu(x):
    """Numpy ReLU"""
    return np.maximum(x, np.zeros_like(x))

class SA_Planner():
    def __init__(self, env, ode: SA_ODE, normalizer):
        """State Action Planner enables next action prediction
        Normalizer only for the states: actions in [-1, 1]"""
        self.env = env
        self.ode = ode
        self.normalizer = normalizer
        self.device = ode.device
        self.low_bound = env.low_bound.clone().numpy()
        self.high_bound = env.high_bound.clone().numpy()
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.horizon = ode.horizon
    
    def distance_to_bounds(self, s:np.ndarray):
        """Calculate the distance to the low and high bounds of a given state s
        The larger the better. A negative value, means violation of the bounds.
        SPECIFIC TO HOPPER"""
        assert s.shape == (self.state_size,)
        # Only 3 bounds to enforce for the Hopper
        z_min = self.low_bound[1]
        angle_min = self.low_bound[2]
        angle_max = self.high_bound[2]
        return min(s[1] - z_min, s[2] - angle_min, angle_max - s[2])



    def closed_loop_traj(self, s0, traj_len, replan_horizon:int = None, projector=None, N:int=None):
        """Returns a trajectory of length traj_len starting from
        UNnormalized state s0 along with their corresponding action prediction.
        Closed-loop prediction based on the ODE's horizon with given replanning horizon"""
        
        if type(s0) == np.ndarray:
            s0 = torch.tensor(s0, dtype=torch.float32, device=self.device)[None,]
        if s0.device != self.device:
            s0 = s0.to(self.device)
        assert s0.shape == (1, self.state_size)
        if replan_horizon is None:
            replan_horizon = self.horizon
        else:
            assert type(replan_horizon) == int and replan_horizon > 1 and replan_horizon <= self.horizon
        
        num_horizons = traj_len//replan_horizon
        traj_pred = []
        actual_traj = np.zeros((traj_len+1, self.state_size))
        actual_traj[0] = self.env.reset_to(s0[0].numpy())
        reward = 0.

        for h_id in range(num_horizons):
            nor_s0 = self.normalizer.normalize(s0)
            pred = self.ode.sample(nor_s0, n_samples=1, projector=projector, N=N)
            traj_pred.append(self.normalizer.unnormalize(pred[0, :replan_horizon, :self.state_size]))
            action_pred = pred[0, :replan_horizon, self.state_size:].numpy()
            
            for t in range(replan_horizon):
                actual_traj[h_id*replan_horizon + t + 1], r, done = self.env.step(action_pred[t])[:3]
                reward += r
                if done:
                    print(f"Failure at timestep {h_id*replan_horizon + t + 1}")
                    break
            s0 = torch.FloatTensor(self.env.state.copy())
            if done: break

        traj_pred = torch.vstack(traj_pred)
        print(f"Reward {reward:.1f}")
        return traj_pred, actual_traj[:h_id*replan_horizon + t + 2], reward


    @torch.no_grad()
    def best_traj(self, s0, traj_len:int, n_samples_per_s0:int=1,
                  replan_horizon:int = None, projector=None, N:int=None):
        """Returns 1 trajectory of length 'traj_len' starting from the UNnormalized state 's0'.
        For each s0 'n_samples_per_s0' are generated of horizon determined by the ode
        These trajectories are interrupted at 'replan_horizon'. 
        Actions are evaluated with the learned world-model from the initial state, the best trajectory is selected. 
        This sequence of action is then applied open-loop on the initial state with the real robot dynamics (env).
        The resulting state is used as a new initial state for the planning."""
        
        print("No world model in Planner.best_traj.\n Using env instead to select best candidate (cheating)")
        if type(s0) == np.ndarray:
            s0 = torch.tensor(s0, dtype=torch.float32, device=self.device)
        if s0.device != self.device:
            s0 = s0.to(self.device)
        if len(s0.shape) == 1:
            s0 = s0.unsqueeze(0)
        # assert s0.shape == (1, self.state_size), f"Only one initial state at a time, not {s0.shape}"
        assert len(s0.shape) == 2 and s0.shape[1] == self.state_size
        if replan_horizon is None:
            replan_horizon = self.horizon
        else:
            assert type(replan_horizon) == int and replan_horizon > 1 and replan_horizon <= self.horizon
        
        N_s0 = s0.shape[0]
        # actual_traj = np.zeros((N_s0, traj_len+1, self.state_size))
        # actual_traj[:, 0] = s0.clone().numpy()
        actual_traj = [ [state_0.numpy()] for state_0 in s0 ]
        assert len(actual_traj) == N_s0 and len(actual_traj[0]) == 1 and actual_traj[0][0].shape == (self.state_size,), f"shape is {actual_traj[0][0].shape}"
        
        s0 = s0.repeat_interleave(n_samples_per_s0, dim=0)
        n_samples = s0.shape[0] # total number of samples = N_s0 * n_samples_per_s0

        num_horizons = traj_len//replan_horizon
        traj_pred = [[] for _ in range(N_s0)]
        action_pred = [[] for _ in range(N_s0)]
        Rewards = np.zeros(N_s0)
        Done = [False] * N_s0

        for h_id in range(num_horizons):
            nor_s0 = self.normalizer.normalize(s0)
            # Sample all the trajectories at once: faster
            pred = self.ode.sample(nor_s0, n_samples=n_samples, projector=projector, N=N)
            new_traj_pred = self.normalizer.unnormalize(pred[:, :, :self.state_size])
            new_action_pred = pred[:, :, self.state_size:].numpy()
            
            # Pick the best sample based on how well the world-model trajectory does
            for s_id in range(N_s0): # index of the s0
                if Done[s_id]: # trajectory has already failed
                    continue
                T = h_id*replan_horizon
                largest_t = 0
                # largest_distance_to_bounds = -10**9
                largest_distance_to_pred = -10**9
                idx = s_id*n_samples_per_s0
                for sample_id in range(n_samples_per_s0):
                    self.env.reset_to(actual_traj[s_id][T])
                    for t in range(replan_horizon):
                        s, _, done = self.env.step(new_action_pred[idx + sample_id, t])[:3] # TODO: replace with learned World-model
                        if done:
                            break

                    ## Looking at distance to bounds ends up keeping trajectories near 0 which eventually fails
                    # # Pick in priority the longest surviving and if equal, pick the largest distance to state-space bounds as tie-breaker
                    # d_to_bounds = self.distance_to_bounds(s)
                    # if t == largest_t and d_to_bounds > largest_distance_to_bounds:
                    #     largest_distance_to_bounds = d_to_bounds
                    #     best_sample_id = sample_id # keep the sample that stays in bounds the longest

                    # Pick in priority the longest surviving and if equal, pick the closest to prediction as tie-breaker
                    d_to_pred = np.linalg.norm(new_traj_pred[idx + sample_id, t].numpy() - s)
                    if t == largest_t and d_to_pred > largest_distance_to_pred:
                        largest_distance_to_pred = d_to_pred
                        best_sample_id = sample_id # keep the sample that stays in bounds the longest
                    elif t > largest_t:
                        largest_t = t
                        best_sample_id = sample_id # keep the sample that stays in bounds the longest
                    
                print(f"{s_id} longest survival: {T} + {largest_t}")
                traj_pred[s_id].append( new_traj_pred[idx + best_sample_id, :replan_horizon].clone() )
                action_pred[s_id].append( new_action_pred[idx + best_sample_id, :replan_horizon].copy() )

            # # Pick the best sample based on bound respect
            # for s_id in range(N_s0): # index of the s0
            #     if Done[s_id]: # trajectory has already failed
            #         continue
            #     largest_t = 0
            #     idx = s_id*n_samples_per_s0
            #     for sample_id in range(n_samples_per_s0):
            #         traj = new_traj_pred[idx + sample_id]
            #         for t in range(traj.shape[0]): # search for first time index t where traj goes out of bounds
            #             if (traj[t] < self.low_bound).any() or (traj[t] > self.high_bound).any():
            #                 break
            #         if t > largest_t:
            #             largest_t = t
            #             best_sample_id = sample_id # keep the sample that stays in bounds the longest
            #         if t == traj_len-1: break # traj reached the end of the horizon, no need to test the others
                    
            #     traj_pred[s_id].append( new_traj_pred[idx + best_sample_id, :replan_horizon].clone() )
            #     action_pred[s_id].append( new_action_pred[idx + best_sample_id, :replan_horizon].copy() )

            # Apply the actions (up to replan_horizon) corresponding to the best sample
            for s_id in range(N_s0): # index of the s0
                if Done[s_id]: # trajectory has already failed
                    continue
                T = h_id*replan_horizon
                self.env.reset_to(actual_traj[s_id][T])
                for t in range(replan_horizon):
                    next_state, r, done = self.env.step(action_pred[s_id][-1][t])[:3]
                    actual_traj[s_id].append(next_state)
                    Rewards[s_id] += r
                    if done:
                        print(f"Failure of trajectory {s_id} at timestep {T+t+1}")
                        Done[s_id] = True
                        break
            
            # Update initial state as final state of actual trajectories
            s0 = torch.FloatTensor(np.vstack([actual_traj[s_id][-1] for s_id in range(N_s0)])).to(self.device)
            s0 = s0.repeat_interleave(n_samples_per_s0, dim=0) # multiply per samples

        for s_id in range(N_s0): # concatenate over the time dimension
            actual_traj[s_id] = np.vstack(actual_traj[s_id])
            traj_pred[s_id] = torch.vstack(traj_pred[s_id])#.unsqueeze(0)
            action_pred[s_id] = np.vstack(action_pred[s_id])
            # action_pred[s_id] = np.expand_dims(np.vstack(action_pred[s_id]), axis=0)

        ### Cannot concatenate over s0 since trajectories might have different lengths
        # traj_pred = torch.vstack(traj_pred) # N_s0, num_horizon*replan_horizon, state_size
        # action_pred = np.vstack(action_pred) # N_s0, num_horizon*replan_horizon, action_size
        # print(traj_pred.shape, action_pred.shape)

        print("Reward", [f"{r:.1f}" for r in Rewards])
        return traj_pred, action_pred, actual_traj, Rewards
# %%
