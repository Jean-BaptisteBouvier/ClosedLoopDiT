# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:45:36 2024

@author: Jean-Baptiste Bouvier

Wrapper for the Mujoco Hopper Environment with a new function: reset_to(state),
and a list of actuated and unactuated states.
Actions are clamped in [-1, 1]
Uses semi-implicit Euler integrator for step
12 states: 6 positions and 6 velocities,
the x-position is included so that each velocity has its corresponding position
"""

import copy
import torch
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import EnvSpec


class HopperEnv():
    """
    Wrapper for the Hopper Environment    
    
    The hopper is a two-dimensional one-legged figure that consist of four body parts:
    torso at the top, thigh in the middle, leg at the bottom, and a single foot on
    which the entire body rests. The goal is to make hops that move in the
    forward (right) direction by applying torques on the three hinges
    connecting the four body parts.
    
    | Num | Action                             | Min | Max |     Name    | Joint | Unit         |
    |-----|------------------------------------|-----|-----|-------------|-------|--------------|
    | 0   | Torque applied on the thigh rotor  | -1  | 1   | thigh_joint | hinge | torque (N m) |
    | 1   | Torque applied on the leg rotor    | -1  | 1   | leg_joint   | hinge | torque (N m) |
    | 2   | Torque applied on the foot rotor   | -1  | 1   | foot_joint  | hinge | torque (N m) |

    | Num | Observation                              | Min  | Max |    Name     | Joint | Unit                     |
    | --- | -----------------------------------------| ---- | --- | ----------- | ----- | ------------------------ |
    | 0   | x-coordinate of the top (location)       | -Inf | Inf | rootx       | slide | position (m)             |
    | 1   | z-coordinate of the top (height)         | -Inf | Inf | rootz       | slide | position (m)             |
    | 2   | angle of the top                         | -Inf | Inf | rooty       | hinge | angle (rad)              |
    | 3   | angle of the thigh joint                 | -Inf | Inf | thigh_joint | hinge | angle (rad)              |
    | 4   | angle of the leg joint                   | -Inf | Inf | leg_joint   | hinge | angle (rad)              |
    | 5   | angle of the foot joint                  | -Inf | Inf | foot_joint  | hinge | angle (rad)              |
    | 6   | velocity of the x-coordinate of the top  | -Inf | Inf | rootx       | slide | velocity (m/s)           |
    | 7   | velocity of the z-coordinate of the top  | -Inf | Inf | rootz       | slide | velocity (m/s)           |
    | 8   | angular velocity of the angle of the top | -Inf | Inf | rooty       | hinge | angular velocity (rad/s) |
    | 9   | angular velocity of the thigh hinge      | -Inf | Inf | thigh_joint | hinge | angular velocity (rad/s) |
    | 10  | angular velocity of the leg hinge        | -Inf | Inf | leg_joint   | hinge | angular velocity (rad/s) |
    | 11  | angular velocity of the foot hinge       | -Inf | Inf | foot_joint  | hinge | angular velocity (rad/s) |

    The reward consists of three parts:
    - *healthy_reward*: Every timestep that the hopper is healthy, it gets a reward of fixed value `healthy_reward`.
    - *forward_reward*: A reward of hopping forward which is measured
    as *`forward_reward_weight` * (x-coordinate before action - x-coordinate after action)/dt*.
    - *ctrl_cost*: A cost for penalising the hopper if it takes actions that are too large. 

    """

    def __init__(self, render_mode=None):
        
        self.name = "Hopper"
        self.parallel = False # cannot run several systems in parallel
        self.max_episode_steps = 2000
        self.reward_threshold = 1.3*self.max_episode_steps
        self.render_mode = render_mode
        
        # Specify the specs to remove all env wrappers to access the Mujoco env directly
        self.spec = EnvSpec(id = "Hopper-v4",           # The string used to create the environment with :meth:`gymnasium.make`
                       entry_point='gymnasium.envs.mujoco.hopper_v4:HopperEnv', # A string for the environment location
                       # reward_threshold=360.,      # The reward threshold for completing the environment.
                       nondeterministic=False,      # If the observation of an environment cannot be repeated with the same initial state, random number generator state and actions.
                       max_episode_steps=None,      # The max number of steps that the environment can take before truncation
                       order_enforce=False,         # If to enforce the order of :meth:`gymnasium.Env.reset` before :meth:`gymnasium.Env.step` and :meth:`gymnasium.Env.render` functions
                       disable_env_checker=True,    # If to disable the environment checker wrapper in :meth:`gymnasium.make`, by default False (runs the environment checker)
                       kwargs={'render_mode': render_mode}, # Additional keyword arguments passed to the environment during initialisation
                       additional_wrappers=(),      #  A tuple of additional wrappers applied to the environment (WrapperSpec)
                       vector_entry_point=None)     # The location of the vectorized environment to create from
         
        # higher initial noise to generate more varied trajectories
        self.noise_scale = 5e-2 # base value is 5e-3
        self.env = gym.make(self.spec, reset_noise_scale=self.noise_scale,
                            exclude_current_positions_from_observation=False) # adds the x-position
        # Note: These assertions are commented out for compatibility
        # The MuJoCo XML files in newer Gymnasium versions may have different integrator settings
        # assert self.env.model.opt.integrator == 0, "Select 'Euler' for the integrator in the XML file"
        # assert self.env.frame_skip == 1, "Need frame_skip = 1, i.e., single time step between states"
        self.metadata = self.env.metadata
        
        self._seed = 12
        self.env.action_space.seed(self._seed)
        
        self.env.reset(seed = self._seed)
        self.state_size = 12
        self.action_size = 3
        
        self.action_max = np.array([[1., 1., 1.]])
        self.action_min = -self.action_max
        self.min_z = self.env._healthy_z_range[0] # 0.7, reset at 1.25
        self.angle_range = self.env._healthy_angle_range # (-0.2, 0.2), reset at 0
        self.state_range = self.env._healthy_state_range # (-100, 100), reset at 0
        self.dt = self.env.dt # 0.008s # time step
        
        self.actuated_states = [6, 7, 8, 9, 10, 11] # states that are directly actuated
        self.unactuated_states = [0, 1, 2, 3, 4, 5] # states that are not directly actuated
        self.velocity_states = [6, 7, 8, 9, 10, 11] 
        self.position_states = [0, 1, 2, 3, 4, 5] 
        self.state_labels = ["x", "z", "top", "thigh", "leg", "foot", "x", "z", "top", "thigh", "leg", "foot"]
        self.reset_state = np.array([0.0, 1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ### For random reset
        self.min_state = np.array([-10., self.min_z,  self.angle_range[0], self.angle_range[0], self.angle_range[0], self.angle_range[0], -1., -1., -1., -1., -1., -1.])
        self.max_state = np.array([10., 2*self.min_z, self.angle_range[1], self.angle_range[1], self.angle_range[1], self.angle_range[1],  1.,  1.,  1.,  1.,  1.,  1.])
        ### Bounds on admissible state range
        self.low_bound = torch.tensor([-np.inf, self.min_z, -0.2, -100., -100., -100., -100., -100., -100., -100., -100., -100.])
        self.high_bound = torch.tensor([np.inf, np.inf, 0.2, 100., 100., 100., 100., 100., 100., 100., 100., 100.])
        
        
    def reset(self):
        self.episode_step = 0
        self.state, _ = self.env.reset()
        return copy.deepcopy(self.state)
     
    
    def step(self, action):
        self.episode_step += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.state = np.concatenate((self.env.data.qpos, self.env.data.qvel))
        done = terminated or truncated or self.episode_step > self.max_episode_steps
        
        return copy.deepcopy(self.state), reward, done, False, None
    
    
    def reset_to(self, state):
        """New function: reset the state to the one provided.
        Split the state into qpos and qvel to use  env.set_state(qpos, qvel)"""
        
        qpos = state[:6] # desired positions (x, z, theta_top, theta_thigh, theta_leg, theta_foot)  
        qvel = state[6:] # desired velocities (d/dt positions)
        self.env.reset()
        self.env.set_state(qpos, qvel) # Mujoco method to set state
        self.state = np.concatenate((self.env.data.qpos, self.env.data.qvel))
        if np.linalg.norm(self.state - state) > 1e-5:
            print("states dont match") # observation is clipped at 10 (except maybe x)
        self.episode_step = 0
        return copy.deepcopy(self.state)
        
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()
 
    
    

def rollout(env, agent, state_norm, title="", s0=None, display=True):
    """Test the policy on a rollout with trajectory plot"""
    
    if s0 is None:
        state = env.reset()
    else:
        state = env.reset_to(s0)
    N_step = env.max_episode_steps
    episode_reward = 0.
    Traj = np.zeros((N_step, env.state_size))
    Traj[0] = state
    Actions = np.zeros((N_step-1, env.action_size))
    
    for t in range(1, N_step):
        with torch.no_grad():
            action = agent.evaluate(state, state_norm)
        next_state, reward, done = env.step(action)
        episode_reward += reward
        state = next_state
        Traj[t] = state
        Actions[t-1] = action
        if done: break
    
    if display:
        print(title + f" total reward: {episode_reward:.1f}")
        plot_traj(env, Traj[:t+1], title=title)
    return Traj[:t+1], Actions[:t]
   



#%% Original Hopper environment


# from gym import utils
# from gym.envs.mujoco import MujocoEnv
# from gym.spaces import Box

# DEFAULT_CAMERA_CONFIG = {
#     "trackbodyid": 2,
#     "distance": 3.0,
#     "lookat": np.array((0.0, 0.0, 1.15)),
#     "elevation": -20.0,
# }


# class HopperEnv(MujocoEnv, utils.EzPickle):
#     """
#     ### Description

#     This environment is based on the work done by Erez, Tassa, and Todorov in
#     ["Infinite Horizon Model Predictive Control for Nonlinear Periodic Tasks"](http://www.roboticsproceedings.org/rss07/p10.pdf). The environment aims to
#     increase the number of independent state and control variables as compared to
#     the classic control environments. The hopper is a two-dimensional
#     one-legged figure that consist of four main body parts - the torso at the
#     top, the thigh in the middle, the leg in the bottom, and a single foot on
#     which the entire body rests. The goal is to make hops that move in the
#     forward (right) direction by applying torques on the three hinges
#     connecting the four body parts.

#     ### Action Space
#     The action space is a `Box(-1, 1, (3,), float32)`. An action represents the torques applied between *links*

#     | Num | Action                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
#     |-----|------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
#     | 0   | Torque applied on the thigh rotor  | -1          | 1           | thigh_joint                      | hinge | torque (N m) |
#     | 1   | Torque applied on the leg rotor    | -1          | 1           | leg_joint                        | hinge | torque (N m) |
#     | 3   | Torque applied on the foot rotor   | -1          | 1           | foot_joint                       | hinge | torque (N m) |

#     ### Observation Space

#     Observations consist of positional values of different body parts of the
#     hopper, followed by the velocities of those individual parts
#     (their derivatives) with all the positions ordered before all the velocities.

#     By default, observations do not include the x-coordinate of the hopper. It may
#     be included by passing `exclude_current_positions_from_observation=False` during construction.
#     In that case, the observation space will have 12 dimensions where the first dimension
#     represents the x-coordinate of the hopper.
#     Regardless of whether `exclude_current_positions_from_observation` was set to true or false, the x-coordinate
#     will be returned in `info` with key `"x_position"`.

#     However, by default, the observation is a `ndarray` with shape `(11,)` where the elements
#     correspond to the following:

#     | Num | Observation                                      | Min  | Max | Name (in corresponding XML file) | Joint | Unit                     |
#     | --- | ------------------------------------------------ | ---- | --- | -------------------------------- | ----- | ------------------------ |
#     | 0   | z-coordinate of the top (height of hopper)       | -Inf | Inf | rootz                            | slide | position (m)             |
#     | 1   | angle of the top                                 | -Inf | Inf | rooty                            | hinge | angle (rad)              |
#     | 2   | angle of the thigh joint                         | -Inf | Inf | thigh_joint                      | hinge | angle (rad)              |
#     | 3   | angle of the leg joint                           | -Inf | Inf | leg_joint                        | hinge | angle (rad)              |
#     | 4   | angle of the foot joint                          | -Inf | Inf | foot_joint                       | hinge | angle (rad)              |
#     | 5   | velocity of the x-coordinate of the top          | -Inf | Inf | rootx                            | slide | velocity (m/s)           |
#     | 6   | velocity of the z-coordinate (height) of the top | -Inf | Inf | rootz                            | slide | velocity (m/s)           |
#     | 7   | angular velocity of the angle of the top         | -Inf | Inf | rooty                            | hinge | angular velocity (rad/s) |
#     | 8   | angular velocity of the thigh hinge              | -Inf | Inf | thigh_joint                      | hinge | angular velocity (rad/s) |
#     | 9   | angular velocity of the leg hinge                | -Inf | Inf | leg_joint                        | hinge | angular velocity (rad/s) |
#     | 10  | angular velocity of the foot hinge               | -Inf | Inf | foot_joint                       | hinge | angular velocity (rad/s) |


#     ### Rewards
#     The reward consists of three parts:
#     - *healthy_reward*: Every timestep that the hopper is healthy (see definition in section "Episode Termination"), it gets a reward of fixed value `healthy_reward`.
#     - *forward_reward*: A reward of hopping forward which is measured
#     as *`forward_reward_weight` * (x-coordinate before action - x-coordinate after action)/dt*. *dt* is
#     the time between actions and is dependent on the frame_skip parameter
#     (fixed to 4), where the frametime is 0.002 - making the
#     default *dt = 4 * 0.002 = 0.008*. This reward would be positive if the hopper
#     hops forward (positive x direction).
#     - *ctrl_cost*: A cost for penalising the hopper if it takes
#     actions that are too large. It is measured as *`ctrl_cost_weight` *
#     sum(action<sup>2</sup>)* where *`ctrl_cost_weight`* is a parameter set for the
#     control and has a default value of 0.001

#     The total reward returned is ***reward*** *=* *healthy_reward + forward_reward - ctrl_cost* and `info` will also contain the individual reward terms

#     ### Starting State
#     All observations start in state
#     (0.0, 1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) with a uniform noise
#      in the range of [-`reset_noise_scale`, `reset_noise_scale`] added to the values for stochasticity.

#     ### Episode End
#     The hopper is said to be unhealthy if any of the following happens:

#     1. An element of `observation[1:]` (if  `exclude_current_positions_from_observation=True`, else `observation[2:]`) is no longer contained in the closed interval specified by the argument `healthy_state_range`
#     2. The height of the hopper (`observation[0]` if  `exclude_current_positions_from_observation=True`, else `observation[1]`) is no longer contained in the closed interval specified by the argument `healthy_z_range` (usually meaning that it has fallen)
#     3. The angle (`observation[1]` if  `exclude_current_positions_from_observation=True`, else `observation[2]`) is no longer contained in the closed interval specified by the argument `healthy_angle_range`

#     If `terminate_when_unhealthy=True` is passed during construction (which is the default),
#     the episode ends when any of the following happens:

#     1. Truncation: The episode duration reaches a 1000 timesteps
#     2. Termination: The hopper is unhealthy

#     If `terminate_when_unhealthy=False` is passed, the episode is ended only when 1000 timesteps are exceeded.

#     ### Arguments

#     No additional arguments are currently supported in v2 and lower.

#     ```
#     env = gym.make('Hopper-v2')
#     ```

#     v3 and v4 take gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc.

#     ```
#     env = gym.make('Hopper-v4', ctrl_cost_weight=0.1, ....)
#     ```

#     | Parameter                                    | Type      | Default               | Description                                                                                                                                                                     |
#     | -------------------------------------------- | --------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
#     | `xml_file`                                   | **str**   | `"hopper.xml"`        | Path to a MuJoCo model                                                                                                                                                          |
#     | `forward_reward_weight`                      | **float** | `1.0`                 | Weight for _forward_reward_ term (see section on reward)                                                                                                                        |
#     | `ctrl_cost_weight`                           | **float** | `0.001`               | Weight for _ctrl_cost_ reward (see section on reward)                                                                                                                           |
#     | `healthy_reward`                             | **float** | `1`                   | Constant reward given if the ant is "healthy" after timestep                                                                                                                    |
#     | `terminate_when_unhealthy`                   | **bool**  | `True`                | If true, issue a done signal if the hopper is no longer healthy                                                                                                                 |
#     | `healthy_state_range`                        | **tuple** | `(-100, 100)`         | The elements of `observation[1:]` (if `exclude_current_positions_from_observation=True`, else `observation[2:]`) must be in this range for the hopper to be considered healthy  |
#     | `healthy_z_range`                            | **tuple** | `(0.7, float("inf"))` | The z-coordinate must be in this range for the hopper to be considered healthy                                                                                                  |
#     | `healthy_angle_range`                        | **tuple** | `(-0.2, 0.2)`         | The angle given by `observation[1]` (if `exclude_current_positions_from_observation=True`, else `observation[2]`) must be in this range for the hopper to be considered healthy |
#     | `reset_noise_scale`                          | **float** | `5e-3`                | Scale of random perturbations of initial position and velocity (see section on Starting State)                                                                                  |
#     | `exclude_current_positions_from_observation` | **bool**  | `True`                | Whether or not to omit the x-coordinate from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies               |

#     ### Version History

#     * v4: all mujoco environments now use the mujoco bindings in mujoco>=2.1.3
#     * v3: support for gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc. rgb rendering comes from tracking camera (so agent does not run away from screen)
#     * v2: All continuous control environments now use mujoco_py >= 1.50
#     * v1: max_time_steps raised to 1000 for robot based tasks. Added reward_threshold to environments.
#     * v0: Initial versions release (1.0.0)
#     """

#     metadata = {
#         "render_modes": [
#             "human",
#             "rgb_array",
#             "depth_array",
#         ],
#         "render_fps": 125,
#     }

#     def __init__(
#         self,
#         forward_reward_weight=1.0,
#         ctrl_cost_weight=1e-3,
#         healthy_reward=1.0,
#         terminate_when_unhealthy=True,
#         healthy_state_range=(-100.0, 100.0),
#         healthy_z_range=(0.7, float("inf")),
#         healthy_angle_range=(-0.2, 0.2),
#         reset_noise_scale=5e-3,
#         exclude_current_positions_from_observation=True,
#         **kwargs
#     ):
#         utils.EzPickle.__init__(
#             self,
#             forward_reward_weight,
#             ctrl_cost_weight,
#             healthy_reward,
#             terminate_when_unhealthy,
#             healthy_state_range,
#             healthy_z_range,
#             healthy_angle_range,
#             reset_noise_scale,
#             exclude_current_positions_from_observation,
#             **kwargs
#         )

#         self._forward_reward_weight = forward_reward_weight

#         self._ctrl_cost_weight = ctrl_cost_weight

#         self._healthy_reward = healthy_reward
#         self._terminate_when_unhealthy = terminate_when_unhealthy

#         self._healthy_state_range = healthy_state_range
#         self._healthy_z_range = healthy_z_range
#         self._healthy_angle_range = healthy_angle_range

#         self._reset_noise_scale = reset_noise_scale

#         self._exclude_current_positions_from_observation = (
#             exclude_current_positions_from_observation
#         )

#         if exclude_current_positions_from_observation:
#             observation_space = Box(
#                 low=-np.inf, high=np.inf, shape=(11,), dtype=np.float64
#             )
#         else:
#             observation_space = Box(
#                 low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64
#             )

#         MujocoEnv.__init__(
#             self, "hopper.xml", 4, observation_space=observation_space, **kwargs
#         )

#     @property
#     def healthy_reward(self):
#         return (
#             float(self.is_healthy or self._terminate_when_unhealthy)
#             * self._healthy_reward
#         )

#     def control_cost(self, action):
#         control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
#         return control_cost

#     @property
#     def is_healthy(self):
#         z, angle = self.data.qpos[1:3]
#         state = self.state_vector()[2:]

#         min_state, max_state = self._healthy_state_range
#         min_z, max_z = self._healthy_z_range
#         min_angle, max_angle = self._healthy_angle_range

#         healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
#         healthy_z = min_z < z < max_z
#         healthy_angle = min_angle < angle < max_angle

#         is_healthy = all((healthy_state, healthy_z, healthy_angle))

#         return is_healthy

#     @property
#     def terminated(self):
#         terminated = not self.is_healthy if self._terminate_when_unhealthy else False
#         return terminated

#     def _get_obs(self):
#         position = self.data.qpos.flat.copy()
#         velocity = np.clip(self.data.qvel.flat.copy(), -10, 10)

#         if self._exclude_current_positions_from_observation:
#             position = position[1:]

#         observation = np.concatenate((position, velocity)).ravel()
#         return observation

#     def step(self, action):
#         x_position_before = self.data.qpos[0]
#         self.do_simulation(action, self.frame_skip)
#         x_position_after = self.data.qpos[0]
#         x_velocity = (x_position_after - x_position_before) / self.dt

#         ctrl_cost = self.control_cost(action)

#         forward_reward = self._forward_reward_weight * x_velocity
#         healthy_reward = self.healthy_reward

#         rewards = forward_reward + healthy_reward
#         costs = ctrl_cost

#         observation = self._get_obs()
#         reward = rewards - costs
#         terminated = self.terminated
#         info = {
#             "x_position": x_position_after,
#             "x_velocity": x_velocity,
#         }

#         if self.render_mode == "human":
#             self.render()
#         return observation, reward, terminated, False, info

#     def reset_model(self):
#         noise_low = -self._reset_noise_scale
#         noise_high = self._reset_noise_scale

#         qpos = self.init_qpos + self.np_random.uniform(
#             low=noise_low, high=noise_high, size=self.model.nq
#         )
#         qvel = self.init_qvel + self.np_random.uniform(
#             low=noise_low, high=noise_high, size=self.model.nv
#         )

#         self.set_state(qpos, qvel)

#         observation = self._get_obs()
#         return observation

#     def viewer_setup(self):
#         assert self.viewer is not None
#         for key, value in DEFAULT_CAMERA_CONFIG.items():
#             if isinstance(value, np.ndarray):
#                 getattr(self.viewer.cam, key)[:] = value
#             else:
#                 setattr(self.viewer.cam, key, value)

