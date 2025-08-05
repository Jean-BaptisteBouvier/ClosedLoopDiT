# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 08:24:10 2024

@author: Jean-Baptiste Bouvier

Projectors parent class
Admissible_Projector


Reference Progressive Projector
Like the progressive projector in terms of scheduling, except that it is also
given the reference trajectory to project onto and tries to maintain the trajectory close
"""

import torch
import cvxpy as cp
import numpy as np
from utils import norm, vertices
from cvxpylayers.torch import CvxpyLayer




#%% Parent Projector

class Projector():
    def __init__(self, env, normalizer, sigma_min=0.0021, sigma_max=0.2, states_projected="vel",
                 correction=False, reference=False, target=False):
        """Parent class for all projectors.
        sigma > sigma_max: no projections
        sigma in [sigma_min, sigma_max]: projection probability proportional to sigma
        sigma < sigma_min: 100% projection
        states_projected: describes whether the position ("pos"), velocities ("vel"), or the full state ("full")
            is projected in the reachable set
        correction: whether a post-projection correction is necessary
        reference: whether the projector uses a reference trajectory when projecting a trajectory
        target: (requires reference) whether the projector tries to get as close as possible to the reference
        
        Optimized projection by defining the cvxpy problem once at initialization and solving it with different parameters.
        """

        self.env = env
        self.normalizer = normalizer
        assert sigma_min >= 0., "Projections happen for sigma <= sigma_min which must be non-negative"
        self.sigma_min = sigma_min
        assert sigma_max >= sigma_min
        self.sigma_max = sigma_max
        assert states_projected in ["pos", "vel", "full"], "states_projected must be in 'pos', 'vel', or 'full'."
        self.states_projected = states_projected
        self.correction = correction
        if reference: assert states_projected == "vel", "We use the reference velocity"
        self.reference = reference
        if target: assert reference, "To optimize the projection towards a target, the projector needs a reference"
        self.target = target
        
        self.device = normalizer.mean.device
        self.state_size = self.env.state_size
        self.action_size = self.env.action_size
        self.a_min = torch.FloatTensor(self.env.action_min)
        self.a_max = torch.FloatTensor(self.env.action_max)
        self.extremal_actions = vertices(self.a_min, self.a_max).reshape((2**self.action_size, self.action_size))
        # self.extremal_actions = torch.FloatTensor(self.extremal_actions)
        self.nb_actions = self.extremal_actions.shape[0]
        
        self.pos = self.env.position_states
        self.vel = self.env.velocity_states
        assert self.vel == env.actuated_states, "Each velocity and only the velocities need to be actuated"
 
        self.dt = env.dt # time step of the environment
        self.eps = 1e-3 # precision of the projector

        
        # Convex optimization problem to project predicted next state on reachable set
        self.nb_vertices = self.extremal_actions.shape[0]
        x = cp.Variable(self.nb_vertices) # 1 coefficient for each vertex to describe point in convexhull
        
        if self.states_projected == "full": # projecting the full state
             P = cp.Parameter((self.state_size, self.nb_vertices)) # vertices
             z = cp.Parameter(self.state_size)     # point to project into convexhull of vertices
        elif self.states_projected == "vel": # projecting only the velocity states
            P = cp.Parameter((len(self.vel), self.nb_vertices)) # vertices
            z = cp.Parameter(len(self.vel))     # point to project into convexhull of vertices
        else: # projecting only the position states
            P = cp.Parameter((len(self.pos), self.nb_vertices)) # vertices
            z = cp.Parameter(len(self.pos))     # point to project into convexhull of vertices
        
        base_constraints = [sum(x) == 1, x >= 0] # ensure that we get a convex combination of vertices
        base_objective = cp.Minimize(cp.pnorm(P @ x - z, p=2)) # minimize 2-norm distance of convex combination to z
        base_problem = cp.Problem(base_objective, base_constraints)
        assert base_problem.is_dpp()
        self.base_cvxpylayer = CvxpyLayer(base_problem, parameters=[P, z], variables=[x])
    
    
        if self.reference:
            assert self.states_projected == "vel", "Reference projectors need to project the velocity"
            # First optimize with hard constraints on the deviation from the reference
            lb = cp.Parameter(1) # lower bound on the torso angular velocity
            ub = cp.Parameter(1) # upper bound on the torso angular velocity
            params = [P, z, lb, ub]
            ref_constraints = [sum(x) == 1, x >= 0, P[2] @ x <= ub, P[2] @ x >= lb]
            
            if self.target: # add the optimization target of the reference angle
                ref = cp.Parameter(1) # value of the reference angle
                params.append(ref)
                target_objective = cp.Minimize(cp.pnorm(P @ x - z, p=2) + cp.abs(P[2] @ x - ref))
                constrained_problem = cp.Problem(target_objective, ref_constraints)
            else:
                constrained_problem = cp.Problem(base_objective, ref_constraints)
                
            assert constrained_problem.is_dpp() 
            self.constrained_cvxpylayer = CvxpyLayer(constrained_problem, parameters=params, variables=[x])
        
            # If there is no solution verifying the ub and lb, try a penalized approach
            if self.target:
                penalized_objective = cp.Minimize(cp.pnorm(P @ x - z, p=2) + cp.maximum(0, P[2]@x - ub) + cp.maximum(0, lb - P[2]@x) + cp.abs(P[2] @ x - ref))
            else:
                penalized_objective = cp.Minimize(cp.pnorm(P @ x - z, p=2) + cp.maximum(0, P[2]@x - ub) + cp.maximum(0, lb - P[2]@x) )
            
            penalized_problem = cp.Problem(penalized_objective, base_constraints)
            assert penalized_problem.is_dpp()
            self.penalized_cvxpylayer = CvxpyLayer(penalized_problem, parameters=params, variables=[x])
            
        
        
    @torch.no_grad()
    def reachable_vertices(self, States: torch.Tensor, actions: torch.Tensor):
        """Generates the reachable set from 'States'.
        Returns a list of extremal vertices corresponding to bang inputs: nb_actions = 2**action_size
        In:  current state (nb_traj, state_size)
             actions (nb_trajs, nb_actions, action_size)
        Out: reachable vertices (nb_traj, nb_actions, state_size)"""
        
        N_trajs = States.shape[0]
        S0 = States.cpu().numpy()
        actions = actions.cpu().numpy()
        assert len(actions.shape) == 3 # whether the actions are different for each traj
        
        Reachable_vertices = np.zeros((N_trajs, self.nb_actions, self.state_size))
        
        for traj_id in range(N_trajs):
            for i in range(self.nb_actions):
                self.env.reset_to(S0[traj_id])
                Reachable_vertices[traj_id, i] = self.env.step(actions[traj_id, i])[0]
        
        return torch.FloatTensor(Reachable_vertices).cpu()


    def projection_probabilities(self, sigma: torch.Tensor):
        """Calculates the probability of projection given sigma"""
        prob = torch.zeros_like(sigma)
        prob += sigma < self.sigma_min # probability = 1 for sigma < sigma_min
        idx = (sigma >= self.sigma_min) * (sigma < self.sigma_max)
        prob[idx] += (self.sigma_max - sigma[idx])/(self.sigma_max - self.sigma_min)
        return prob


    def reshape_sigma(self, sigma, size):
        """Reshape the noise level sigma into a tensor of desired size"""
        if type(sigma) == float:
            sigma = torch.ones((size))*sigma
        elif type(sigma) == torch.Tensor:
            sigma = sigma.reshape((-1))
        elif type(sigma) == np.ndarray:
            sigma = torch.FloatTensor(sigma)
        else:
            raise Exception(f"sigma is neither a float, array, or a tensor but a {type(sigma)}")
        return sigma
    
    
    def project_traj(self, Trajs: torch.Tensor, Ref_Trajs: torch.Tensor = None,
                     sigma: float = 0., normalized=True, Actions=None):
        """Projects trajectories onto an admissible set at noise scale sigma, 
        trying to keep them close to their reference trajectories
        Trajs:     Tensor (N_trajs, horizon, state_size) assumed to be normalized
        Ref_Trajs: Tensor (N_trajs, horizon, state_size) assumed to be normalized
        Actions:   Tensor or Array (N_trajs, horizon, action_size)  or None
        Return projected_trajs:   Tensor (N_trajs, horizon, state_size) 
               projected_actions: Tensor (N_trajs, horizon, action_size) # last action unchanged
        """
        
        if self.reference:
            assert Trajs.shape == Ref_Trajs.shape, f"Trajs {Trajs.shape} does not match Ref_Trajs {Ref_Trajs.shape}"
        
        N_trajs = Trajs.shape[0]
        N_steps = Trajs.shape[1]-1
        Proj_Trajs = Trajs.clone()
        if Actions is None:
            Proj_Actions = torch.zeros((N_trajs, N_steps+1, self.action_size))
        else:
            assert Actions.shape == (N_trajs, N_steps+1, self.action_size), f"Actions should be of shape ({N_trajs}, {N_steps+1}, {self.action_size})"
            if type(Actions) == torch.Tensor:
                Proj_Actions = Actions.clone() # return a tensor of actions
                # Actions = np.array(Actions)    # need an array of actions to propage in gym
            else:
                Actions = torch.FloatTensor(Actions)
                Proj_Actions = Actions.clone()
                
        sigma = self.reshape_sigma(sigma, Trajs.shape[0])
        
        if min(sigma) > self.sigma_max: # no projections to be done
            return Proj_Trajs, Proj_Actions
        rand = torch.rand(N_steps) # random steps where projections are needed
        pp = self.projection_probabilities(sigma) # vector (N_trajs, 1) of probabilities
        idx = torch.arange(sigma.shape[0], device=self.device)
        
        for t_id in range(N_steps):
            # indices of the trajectories that need projections
            idx_proj = idx[rand[t_id] < pp]    
            if len(idx_proj) > 0: # needs exact projection
                S_t = Proj_Trajs[idx_proj, t_id, :] # projected current state
                S_t_dt = Trajs[idx_proj, t_id+1, :] # predicted next state to be made admissible
                
                if self.reference:
                    Ref_t_dt = Ref_Trajs[idx_proj, t_id+1, :] # reference next state
                else:
                    Ref_t_dt = None
                   
                if Actions is None:
                    A_t = None
                else:
                    A_t = Actions[idx_proj, t_id]
                    
                Proj_Trajs[idx_proj, t_id+1, :], Proj_Actions[idx_proj, t_id, :] = self.make_next_state_admissible(S_t, S_t_dt, Ref_t_dt, sigma[idx_proj], normalized, A_t)
                
        return Proj_Trajs, Proj_Actions
    


    def make_next_state_admissible(self, S_t: torch.Tensor, S_t_dt: torch.Tensor,
                                   Ref_t_dt:torch.Tensor = None, sigma: float = 0.,
                                   normalized=True, A_t: torch.Tensor = None):
        """Makes S_t_dt admissible from S_t using a reachable set approximation
        for the actuated states. 
        Uses semi-implicit Euler integrator to assign the value of the unactuated states. 
        Ref_t_dt is the reference next state from which S_t_dt should not be too far away
        States are assumed to be normalized
        S_t, S_t_dt, Ref_t_dt  Tensor(N_trajs, state_size)
        A_t: candidate action  Tensor(N_trajs, action_size)"""
        
        N_trajs = S_t.shape[0]
        sigma = self.reshape_sigma(sigma, N_trajs)
        
        if normalized: # Unnormalize states to use the dynamics
            S_t = self.normalizer.unnormalize(S_t)
            S_t_dt = self.normalizer.unnormalize(S_t_dt)
            if self.reference:
                Ref_t_dt = self.normalizer.unnormalize(Ref_t_dt)
        
        adm_S_t_dt = torch.zeros_like(S_t_dt) # admissible next state (to be computed)
        adm_A_t = torch.zeros((N_trajs, self.action_size))
        if A_t is None:
            Action_vertices = torch.broadcast_to(self.extremal_actions, (N_trajs, self.nb_actions, self.action_size))
            # Action_vertices = np.broadcast_to(self.extremal_actions, (N_trajs, self.nb_actions, self.action_size))
        else:
            A_t = A_t.reshape((N_trajs, 1, self.action_size)).repeat_interleave(self.nb_actions, dim=1)
            Action_vertices = (A_t + 0.1*self.extremal_actions).clip(self.a_min, self.a_max) # action
        
        R = self.reachable_vertices(S_t, Action_vertices)
        
        for traj_id in range(N_trajs): # Can't be parallelized because of the projection
            tol = sigma[traj_id] + 1e-6 # error tolerance for the convex optimization
            
            if self.states_projected == "vel": # update the velocity states
                s_pred = S_t_dt[traj_id, self.vel].clone()
                Vertices = R[traj_id, :, self.vel].T.to(self.device) # vertices of the reachable set in velocity space
            elif self.states_projected == "pos": # update the position states
                s_pred = S_t_dt[traj_id, self.pos].clone()
                Vertices = R[traj_id, :, self.pos].T.to(self.device) # vertices of the reachable set in position space
            else: # update the full state
                s_pred = S_t_dt[traj_id].clone()
                Vertices = R[traj_id].T.to(self.device) # vertices of the reachable set
            
            ### Convex optimization: closest point to s_vel in the convex hull of the vertices of the reachable set
            if self.reference and sigma[traj_id] < self.sigma_min: # at small noise level keep angular velocity of the top joint close to the reference
                angle_ub = min(Ref_t_dt[traj_id, 2] + 0.1*abs(Ref_t_dt[traj_id, 2]) + 0.01,  1.) # upper bound on the top angle
                angle_lb = max(Ref_t_dt[traj_id, 2] - 0.1*abs(Ref_t_dt[traj_id, 2]) - 0.01, -1.) # lower bound on the top angle
                ub = (angle_ub - S_t[traj_id, [2]])/self.dt # upper bound on the top angle velocity
                lb = (angle_lb - S_t[traj_id, [2]])/self.dt # lower bound on the top angle velocity
                ref = (Ref_t_dt[traj_id, [2]] - S_t[traj_id, [2]])/self.dt
                if all( (Vertices[2,:] > ub) + (Vertices[2,:] < lb) ): # reachable set entirely out of the tube (vertices could be on both sides)
                    if self.target:
                        solution, = self.penalized_cvxpylayer(Vertices, s_pred, lb, ub, ref, solver_args={"eps":tol})
                    else:
                        solution, = self.penalized_cvxpylayer(Vertices, s_pred, lb, ub, solver_args={"eps":tol})
                else: # some of the reachable set is in the tube, we can enforce exactly the constraint
                    if self.target:
                        solution, = self.constrained_cvxpylayer(Vertices, s_pred, lb, ub, ref, solver_args={"eps":tol})
                    else:
                        solution, = self.constrained_cvxpylayer(Vertices, s_pred, lb, ub, solver_args={"eps":tol})
            
            else: # either not reference or too much noise
                solution, = self.base_cvxpylayer(Vertices, s_pred)
           
            adm_A_t[traj_id] = solution @ Action_vertices[traj_id]
            if self.states_projected == "vel":
                # Makes the transition of the position states admissible    
                adm_S_t_dt[traj_id, self.vel] = Vertices @ solution
                x_t = S_t[traj_id, self.pos]
                v_t_dt = adm_S_t_dt[traj_id, self.vel]
                adm_S_t_dt[traj_id, self.pos] = x_t + self.dt * v_t_dt # = x(t+dt) = x(t) + dt * v(t+dt)
                
            elif self.states_projected == "pos":
                # Makes the transition of the position states admissible by choosing correct velocity
                adm_S_t_dt[traj_id, self.pos] = Vertices @ solution
                x_t = S_t[traj_id, self.pos]
                x_t_dt = adm_S_t_dt[traj_id, self.pos]
                adm_S_t_dt[traj_id, self.vel] = (x_t_dt - x_t) / self.dt # =  v(t+dt) = (x(t+dt) - x(t)) / dt
             
            else: # projected the full state
                adm_S_t_dt[traj_id] = Vertices @ solution
        
            if self.correction:
                adm_S_t_dt[traj_id] = self.post_projection_correction(S_t[traj_id], adm_S_t_dt[traj_id], R[traj_id], sigma[traj_id])
        
        if normalized: # Normalize admissible state
            adm_S_t_dt = self.normalizer.normalize(adm_S_t_dt)
        
        return adm_S_t_dt, adm_A_t
    
    
    def post_projection_correction(self, S_t, adm_S_t_dt, R, sigma):
        """Implemented in specific projectors"""
        return adm_S_t_dt



#%%



class Admissible_Projector(Projector):
    def __init__(self, env, normalizer, sigma_min:float = 0.0021,
                 states_projected: str = 'vel'):
        """Class to project states and whole trajectories into their closest
        admissible set. Can be incorporated as a differentiable layer.
        exact: whether the reachable set is exact or enlarged
        states_projected: in 'vel', 'pos', or 'full' """
        
        sigma_max = sigma_min # Full projection for sigma < sigma_min
        super().__init__(env, normalizer, sigma_min, sigma_max, states_projected,
                         reference=False, target=False)
        
        self.name = "admissible_proj_sigma_" + str(sigma_min)
        






class Reference_Projector(Projector):
    def __init__(self, env, normalizer, sigma_min:float = 0.0021,
                 sigma_max:float = 0.1, target:bool = False):
        """Class to project states and whole trajectories into their closest
        admissible set guided by the reference trajectory. 
        Can be incorporated as a differentiable layer.
        sigma > sigma_max: no projections
        sigma in [sigma_min, sigma_max]: projection probability proportional to sigma
        sigma < sigma_min: 100% projection"""
        
        super().__init__(env, normalizer, sigma_min, sigma_max, states_projected="vel",
                         reference=True, target=target, correction=False)
        self.name = "reference_proj_sigma_" + str(sigma_min) + "_" + str(sigma_max)
        if target:
            self.name = self.name + "_target"

        
    
  



#%% State-Action Projector
import os
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt


class SA_Projector():
    def __init__(self, env, normalizer, sigma_min=0.0021, sigma_max=0.2, width=64):
        """State-Action projector without convex optimization.
        sigma > sigma_max: no projections
        sigma in [sigma_min, sigma_max]: projection probability proportional to sigma
        sigma < sigma_min: 100% projection
        """

        self.env = env
        self.normalizer = normalizer
        assert sigma_min >= 0., "Projections happen for sigma <= sigma_min which must be non-negative"
        self.sigma_min = sigma_min
        assert sigma_max >= sigma_min
        self.sigma_max = sigma_max
        
        self.name = "SA_proj_" + str(width) + "_sigma_"+ str(sigma_min) + "_" + str(sigma_max) 
        
        self.device = normalizer.mean.device
        self.state_size = self.env.state_size
        self.action_size = self.env.action_size
        self.a_min = torch.FloatTensor(self.env.action_min)
        self.a_max = torch.FloatTensor(self.env.action_max)
        
        self.vel = self.env.velocity_states
       
        self.dt = env.dt # time step of the environment
        self.eps = 1e-3 # precision of the projector

        # Network to predict the difference in action given the difference in velocity between two states obtained from a fixed initial state
        self.width = width
        self.net = nn.Sequential(nn.Linear(len(self.vel), self.width), nn.ReLU(),
                                 nn.Linear(self.width, self.width), nn.ReLU(),
                                 nn.Linear(self.width, self.action_size))
        self.optim = torch.optim.AdamW(self.net.parameters(), lr=2e-4, weight_decay=1e-4)


    def train(self, Trajs: torch.Tensor, Actions: torch.Tensor, batch_size=32,
              n_gradient_steps=10000, normalized=False, extra=""):
        """Train the neural network of the projector to reconstitute the action
        given a desired change in final state
        Trajs   (N_trajs, H, state_size)
        Actions (N_trajs, H, action_size)
        Trajs are assumed to be NOT normalized"""
        
        if normalized:
            Trajs = self.normalizer.unnormalize(Trajs)
        N_trajs, H, _ = Trajs.shape
        assert Trajs.shape[2] == self.state_size
        N = N_trajs*(H-1) # number of training points
    
        S_t   =  Trajs[:, :-1].reshape((N_trajs*(H-1), self.state_size)) # current state
        S_t_dt =  Trajs[:, 1:].reshape((N_trajs*(H-1), self.state_size)) # desired next state
        A_t = Actions[:, :H-1].reshape((N_trajs*(H-1), self.action_size)) # desired action
        S_t = S_t.numpy()
        
        print("Noising dataset")
        Noise = torch.randn_like(A_t) * (self.sigma_min + self.sigma_max/5) # sigma_min + 5*sigma < sigma_max
        Noised_A_t = (A_t + Noise).clip(self.a_min, self.a_max)
        Noised_S_t_dt = np.zeros((N, self.state_size))
        for i in range(N):
            self.env.reset_to(S_t[i])
            Noised_S_t_dt[i] = self.env.step(Noised_A_t[i].numpy())[0] # noised next state
        
        # Training inputs to the neural network
        Vel_dif = S_t_dt[:, self.vel] - torch.FloatTensor(Noised_S_t_dt[:, self.vel]) # difference in velocities
        # Target outputs to the neural network
        Action_dif = A_t - Noised_A_t
        
        print("Training the State-Action Projector")
        pbar = tqdm(range(n_gradient_steps))
        loss_avg = 0.
        self.training_loss = torch.zeros(n_gradient_steps)
        for step in range(n_gradient_steps):
            
            idx = np.random.randint(0, N, batch_size) # sample a random batch
            pred = self.net(Vel_dif[idx])
            loss = ((Action_dif[idx] - pred)**2).mean()
            self.optim.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.)
            self.optim.step()
            self.training_loss[step] = loss
            loss_avg += loss.item()
            if (step+1) % 100 == 0:
                pbar.set_description(f'step: {step+1} loss: {loss_avg / 100.:.4f} grad_norm: {grad_norm:.4f}')
                pbar.update(100)
                loss_avg = 0.
                self.save(extra)
                
        print('\nTraining completed!')
        self.save(extra)
        plt.plot(self.training_loss.detach().numpy())
        plt.title("Training loss for " + self.name)
        plt.show()
        self._freeze()


    def save(self, extra:str = ""):
         torch.save({'net': self.net.state_dict()}, "trained_models/"+ self.name+extra+".pt")
         
         
    def load(self, extra:str = ""):
         name = "trained_models/" + self.name + extra + ".pt"
         if os.path.isfile(name):
             print("Loading " + name)
             checkpoint = torch.load(name, map_location=self.device, weights_only=True)
             self.net.load_state_dict(checkpoint['net'])
             self._freeze()
             return True # loaded
         else:
             print("File " + name + " doesn't exist. Not loading anything.")
             return False # not loaded



    def projection_probabilities(self, sigma: torch.Tensor):
        """Calculates the probability of projection given sigma"""
        prob = torch.zeros_like(sigma)
        prob += sigma < self.sigma_min # probability = 1 for sigma < sigma_min
        idx = (sigma >= self.sigma_min) * (sigma < self.sigma_max)
        prob[idx] += (self.sigma_max - sigma[idx])/(self.sigma_max - self.sigma_min)
        return prob


    def reshape_sigma(self, sigma, size):
        """Reshape the noise level sigma into a tensor of desired size"""
        if type(sigma) == float:
            sigma = torch.ones((size))*sigma
        elif type(sigma) == torch.Tensor:
            sigma = sigma.reshape((-1))
        elif type(sigma) == np.ndarray:
            sigma = torch.FloatTensor(sigma)
        else:
            raise Exception(f"sigma is neither a float, array, or a tensor but a {type(sigma)}")
        return sigma
    
    
    def project_traj(self, Trajs: torch.Tensor, Actions: torch.Tensor,
                     sigma: float = 0., normalized=True):
        """Projects trajectories onto an admissible set at noise scale sigma, 
        trying to keep them close to their reference trajectories
        Trajs:     Tensor (N_trajs, horizon, state_size) assumed to be normalized
        Actions:   Tensor or Array (N_trajs, horizon, action_size)
        Return projected_trajs:   Tensor (N_trajs, horizon, state_size) 
               projected_actions: Tensor (N_trajs, horizon, action_size) # last action unchanged
        """
        
        N_trajs = Trajs.shape[0]
        N_steps = Trajs.shape[1]-1
        Proj_Trajs = Trajs.clone()
        
        assert Actions.shape == (N_trajs, N_steps+1, self.action_size), f"Actions should be of shape ({N_trajs}, {N_steps+1}, {self.action_size})"
        if type(Actions) == torch.Tensor:
            Proj_Actions = Actions.clone() # return a tensor of actions
        else:
            Actions = torch.FloatTensor(Actions)
            Proj_Actions = Actions.clone()
            
        sigma = self.reshape_sigma(sigma, Trajs.shape[0])
        
        if min(sigma) > self.sigma_max: # no projections to be done
            return Proj_Trajs, Proj_Actions
        
        rand = torch.rand(N_steps) # random steps where projections are needed
        pp = self.projection_probabilities(sigma) # vector (N_trajs, 1) of probabilities
        idx = torch.arange(sigma.shape[0], device=self.device)
        
        for t_id in range(N_steps):
            # indices of the trajectories that need projections
            idx_proj = idx[rand[t_id] < pp]    
            if len(idx_proj) > 0: # needs exact projection
                S_t = Proj_Trajs[idx_proj, t_id, :] # projected current state
                S_t_dt = Trajs[idx_proj, t_id+1, :] # predicted next state to be made admissible
                A_t = Actions[idx_proj, t_id]
                    
                Proj_Trajs[idx_proj, t_id+1, :], Proj_Actions[idx_proj, t_id, :] = self.make_next_state_admissible(S_t, S_t_dt, A_t, sigma[idx_proj], normalized)
                
        return Proj_Trajs, Proj_Actions
    


    def make_next_state_admissible(self, S_t: torch.Tensor, S_t_dt: torch.Tensor,
                                   A_t: torch.Tensor, sigma: float = 0.,
                                   normalized=True):
        """Makes S_t_dt admissible from S_t using a reachable set approximation
        for the actuated states. 
        Uses semi-implicit Euler integrator to assign the value of the unactuated states. 
        States are assumed to be normalized
        S_t, S_t_dt  Tensor(N_trajs, state_size)
        A_t: candidate action  Tensor(N_trajs, action_size)"""
        
        N_trajs = S_t.shape[0]
        sigma = self.reshape_sigma(sigma, N_trajs)
        
        if normalized: # Unnormalize states to use the dynamics
            S_t = self.normalizer.unnormalize(S_t)
            S_t_dt = self.normalizer.unnormalize(S_t_dt)
        
        np_S_t_dt = S_t_dt.clone().detach().numpy() # next state computed with simulator (numpy)
        np_A_t = A_t.clone().detach().numpy()
        S_t = S_t.detach().numpy()
        
        
        for traj_id in range(N_trajs): # Can't be parallelized because of the environment calls
            
            self.env.reset_to(S_t[traj_id])
            s_ol = self.env.step(np_A_t[traj_id])[0] # open-loop next state
            # velocity difference between state prediction and applying the action prediction
            dif = S_t_dt[traj_id, self.vel] - torch.FloatTensor(s_ol[self.vel])
            if norm(dif) < self.eps:
                continue # S_t_dt matches A_t, next state is already admissible
            
            # with torch.no_grad(): # keep gradient here
            da_t = self.net(dif) # action correction
           
            A_t[traj_id] = (A_t[traj_id] + da_t).clip(self.a_min, self.a_max) # corrected action
            np_A_t[traj_id] += da_t.detach().numpy()
            self.env.reset_to(S_t[traj_id])
            np_S_t_dt[traj_id] = self.env.step(np_A_t[traj_id])[0] # corrected next state
        
        S_t_dt += torch.FloatTensor(np_S_t_dt) - S_t_dt.detach() # i.e. S_t_dt = adm_S_t_dt but without losing gradients
        
        if normalized: # Normalize admissible state
            S_t_dt = self.normalizer.normalize(S_t_dt)
        
        return S_t_dt, A_t
    
    
    def _freeze(self):
        """Freeze the Neural Network after training or loading"""
        for param in self.net.parameters():
            param.requires_grad = False




#%% Action Projector


class Action_Projector():
    def __init__(self, env, normalizer, sigma_min=0.0021, sigma_max=0.2):
        """Action projector without convex optimization: 
            open-loop application of the predicted actions.
        sigma > sigma_max: no projections
        sigma in [sigma_min, sigma_max]: projection probability proportional to sigma
        sigma < sigma_min: 100% projection
        """

        self.env = env
        self.normalizer = normalizer
        assert sigma_min >= 0., "Projections happen for sigma <= sigma_min which must be non-negative"
        self.sigma_min = sigma_min
        assert sigma_max >= sigma_min
        self.sigma_max = sigma_max
        
        self.name = "Action_proj_sigma_"+ str(sigma_min) + "_" + str(sigma_max) 
        
        self.device = normalizer.mean.device
        self.state_size = self.env.state_size
        self.action_size = self.env.action_size
        self.a_min = torch.FloatTensor(self.env.action_min)
        self.a_max = torch.FloatTensor(self.env.action_max)
       
        self.dt = env.dt # time step of the environment
        self.eps = 1e-3 # precision of the projector


    def projection_probabilities(self, sigma: torch.Tensor):
        """Calculates the probability of projection given sigma"""
        prob = torch.zeros_like(sigma)
        prob += sigma < self.sigma_min # probability = 1 for sigma < sigma_min
        idx = (sigma >= self.sigma_min) * (sigma < self.sigma_max)
        prob[idx] += (self.sigma_max - sigma[idx])/(self.sigma_max - self.sigma_min)
        return prob


    def reshape_sigma(self, sigma, size):
        """Reshape the noise level sigma into a tensor of desired size"""
        if type(sigma) == float:
            sigma = torch.ones((size))*sigma
        elif type(sigma) == torch.Tensor:
            sigma = sigma.reshape((-1))
        elif type(sigma) == np.ndarray:
            sigma = torch.FloatTensor(sigma)
        else:
            raise Exception(f"sigma is neither a float, array, or a tensor but a {type(sigma)}")
        return sigma
    
    
    def project_traj(self, Trajs: torch.Tensor, Actions: torch.Tensor,
                     sigma: float = 0., normalized=True):
        """Projects trajectories onto an admissible set at noise scale sigma, 
        trying to keep them close to their reference trajectories
        Trajs:     Tensor (N_trajs, horizon, state_size) assumed to be normalized
        Actions:   Tensor or Array (N_trajs, horizon, action_size)
        Return projected_trajs:   Tensor (N_trajs, horizon, state_size) 
               projected_actions: Tensor (N_trajs, horizon, action_size) # last action unchanged
        """
        
        N_trajs = Trajs.shape[0]
        N_steps = Trajs.shape[1]-1
        Proj_Trajs = Trajs.clone()
        
        assert Actions.shape == (N_trajs, N_steps+1, self.action_size), f"Actions should be of shape ({N_trajs}, {N_steps+1}, {self.action_size})"
        if type(Actions) == torch.Tensor:
            Proj_Actions = Actions.clone() # return a tensor of actions
            # Actions = np.array(Actions)    # need an array of actions to propage in gym
        else:
            Actions = torch.FloatTensor(Actions)
            Proj_Actions = Actions.clone()
            
        sigma = self.reshape_sigma(sigma, Trajs.shape[0])
        
        if min(sigma) > self.sigma_max: # no projections to be done
            return Proj_Trajs, Proj_Actions
        
        rand = torch.rand(N_steps) # random steps where projections are needed
        pp = self.projection_probabilities(sigma) # vector (N_trajs, 1) of probabilities
        idx = torch.arange(sigma.shape[0], device=self.device)
        
        for t_id in range(N_steps):
            # indices of the trajectories that need projections
            idx_proj = idx[rand[t_id] < pp]    
            if len(idx_proj) > 0: # needs exact projection
                S_t = Proj_Trajs[idx_proj, t_id, :] # projected current state
                S_t_dt = Trajs[idx_proj, t_id+1, :] # predicted next state to be made admissible
                A_t = Actions[idx_proj, t_id]
                    
                Proj_Trajs[idx_proj, t_id+1, :], Proj_Actions[idx_proj, t_id, :] = self.make_next_state_admissible(S_t, S_t_dt, A_t, sigma[idx_proj], normalized)
                
        return Proj_Trajs, Proj_Actions
    


    def make_next_state_admissible(self, S_t: torch.Tensor, S_t_dt: torch.Tensor,
                                   A_t: torch.Tensor, sigma: float = 0.,
                                   normalized=True):
        """Modifies S_t_dt to make it admissible from S_t using A_t
        States are assumed to be normalized
        S_t, S_t_dt  Tensor(N_trajs, state_size)
        A_t: candidate action  Tensor(N_trajs, action_size)"""
        
        N_trajs = S_t.shape[0]
        sigma = self.reshape_sigma(sigma, N_trajs)
        
        if normalized: # Unnormalize states to use the dynamics
            S_t = self.normalizer.unnormalize(S_t)
            S_t_dt = self.normalizer.unnormalize(S_t_dt)
        
        np_A_t = A_t.clone().detach().numpy()
        S_t = S_t.detach().numpy()
        
        for traj_id in range(N_trajs): # Can't be parallelized because of the environment calls
            
            self.env.reset_to(S_t[traj_id])
            s_ol = self.env.step(np_A_t[traj_id])[0] # open-loop next state
            dif = torch.FloatTensor(s_ol) - S_t_dt[traj_id]
            S_t_dt[traj_id] += dif.detach() # i.e. S_t_dt = s_ol but without losing gradients
        
        if normalized: # Normalize admissible state
            S_t_dt = self.normalizer.normalize(S_t_dt)
        
        return S_t_dt, A_t
     