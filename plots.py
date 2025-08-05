# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:32:09 2024

@author: Jean-Baptiste Bouvier

Hopper specific plotting functions
"""

import numpy as np
import matplotlib.pyplot as plt



def plot_traj(env, Traj, Actions=None, title="", plot_all=False):
    """Plots the trajectory of the hopper.
    state = [positions, velocities]
    positions = [x, z, top_angle, thigh_angle, leg_angle, foot_angle]
    velocities = d/dt positions         
    action = [thigh torque, leg torque, foot torque] """
        
    N = Traj.shape[0] # number of steps of the trajectory before terminating
    time = np.arange(N)
    
    fig, ax = nice_plot()
    plt.title(title)
    plt.scatter(time, Traj[:, 1], s=10)
    plt.plot([0., time[-1]], [env.min_z, env.min_z], color="red")
    plt.ylabel("Hopper height (m)")
    plt.xlabel("timesteps")
    plt.show()
    
    fig, ax = nice_plot()
    plt.title(title)
    plt.scatter(time, Traj[:, 2]*180/np.pi, s=10)
    min_angle = env.angle_range[0]*180/np.pi
    max_angle = env.angle_range[1]*180/np.pi
    plt.plot([0., time[-1]], [min_angle, min_angle], color="red")
    plt.plot([0., time[-1]], [max_angle, max_angle], color="red")
    plt.ylabel("Top angle (deg)")
    plt.xlabel("timesteps")
    plt.show()
    
    fig, ax = nice_plot()
    plt.title(title)
    plt.scatter(time, Traj[:, 6], s=10)
    plt.ylabel("Forward velocity (m/s)")
    plt.xlabel("timesteps")
    plt.show()

    if plot_all: # plot all the states
        fig, ax = nice_plot()
        plt.title(title)
        plt.scatter(time, Traj[:, 0], s=10)
        plt.ylabel("Hopper position (m)")
        plt.xlabel("timesteps")
        plt.show()
    
        fig, ax = nice_plot()
        plt.title(title)
        plt.scatter(time, Traj[:, 2], s=10, label="top")
        plt.scatter(time, Traj[:, 3], s=10, label="thigh")
        plt.scatter(time, Traj[:, 4], s=10, label="leg")
        plt.scatter(time, Traj[:, 5], s=10, label="foot")
        plt.ylabel("Angles (rad)")
        plt.xlabel("timesteps")
        plt.legend(frameon=False, labelspacing=0.3, handletextpad=0.2, handlelength=0.9)
        plt.show()
        
        fig, ax = nice_plot()
        plt.title(title)
        plt.scatter(time, Traj[:, 7], s=10)
        plt.ylabel("Vertical velocity (m/s)")
        plt.xlabel("timesteps")
        plt.show()
    
        fig, ax = nice_plot()
        plt.title(title)
        plt.scatter(time, Traj[:, 8], s=10, label="top")
        plt.scatter(time, Traj[:, 9], s=10, label="thigh")
        plt.scatter(time, Traj[:, 10], s=10, label="leg")
        plt.scatter(time, Traj[:, 11], s=10, label="foot")
        plt.ylabel("Angular velocities (rad/s)")
        plt.xlabel("timesteps")
        plt.legend(frameon=False, labelspacing=0.3, handletextpad=0.2, handlelength=0.9)
        plt.show()

    if Actions is not None:
        
        fig, ax = nice_plot()
        plt.title(title)
        plt.scatter(time[:-1], Actions[:, 0], s=10, label="thigh")
        plt.scatter(time[:-1], Actions[:, 1], s=10, label="leg")
        plt.scatter(time[:-1], Actions[:, 2], s=10, label="foot")
        max_torque = env.action_max[0]
        plt.plot([0., time[-1]], [ max_torque,  max_torque], color="red")
        plt.plot([0., time[-1]], [-max_torque, -max_torque], color="red")
        plt.ylabel("Torques (N m)")
        plt.xlabel("timesteps")
        plt.legend(frameon=False, labelspacing=0.3, handletextpad=0.2, handlelength=0.9)
        plt.show()        
   
 
def traj_comparison(env, traj_1, label_1, traj_2, label_2, title="",
                    traj_3=None, label_3=None, traj_4=None, label_4=None,
                    saveas: str = None, plot_height=True, legend_loc='best',
                    horizon:int = None):
    """Compares given hopper trajectories.
    Optional argument 'saveas' takes the filename to save the plots if desired"""
    
    if horizon is not None:
        assert type(horizon) == int
    assert len(traj_1.shape) == 2, "Trajectory 1 must be a 2D array"
    assert len(traj_2.shape) == 2, "Trajectory 2 must be a 2D array"
    if traj_3 is not None:
        assert len(traj_3.shape) == 2, "Trajectory 3 must be a 2D array"
    if traj_4 is not None:
        assert len(traj_4.shape) == 2, "Trajectory 4 must be a 2D array"
    
    time_1 = np.arange(traj_1.shape[0])
    time_2 = np.arange(traj_2.shape[0])
    time_max = max(time_1[-1], time_2[-1])
    
    fig, ax = nice_plot()
    if title is not None:
        plt.title(title)
    if horizon is not None:
        t1 = min(time_1[-1], horizon)
        plt.plot(time_1[0:t1], traj_1[0:t1, 2]*180/np.pi, label=label_1, linewidth=3, color="tab:blue")
        h = 1
        while h*horizon < time_1[-1]:
            t0 = h*horizon
            t1 = min(time_1[-1], (h+1)*horizon)
            plt.plot(time_1[t0:t1], traj_1[t0:t1, 2]*180/np.pi, linewidth=3, color="tab:blue")
            h += 1

        t1 = min(time_2[-1], horizon)
        plt.plot(time_2[0:t1], traj_2[0:t1, 2]*180/np.pi, label=label_2, linewidth=3, color="tab:orange")
        h = 1
        while h*horizon < time_2[-1]:
            t0 = h*horizon
            t1 = min(time_2[-1], (h+1)*horizon)
            plt.plot(time_2[t0:t1], traj_2[t0:t1, 2]*180/np.pi, linewidth=3, color="tab:orange")
            h += 1
    else:
        plt.plot(time_1, traj_1[:, 2]*180/np.pi, label=label_1, linewidth=3, color="tab:blue")
        plt.plot(time_2, traj_2[:, 2]*180/np.pi, label=label_2, linewidth=3, color="tab:orange")
    y_max = max(traj_1[:, 2].max(), traj_2[:, 2].max())
    y_min = min(traj_1[:, 2].min(), traj_2[:, 2].min())
    if traj_3 is not None:
        time_3 = np.arange(traj_3.shape[0])
        time_max = max(time_max, time_3[-1])
        plt.plot(time_3, traj_3[:, 2]*180/np.pi, label=label_3, linewidth=3)
        y_max = max(traj_3[:, 2].max(), y_max)
        y_min = min(traj_3[:, 2].min(), y_min)
    if traj_4 is not None:
        time_4 = np.arange(traj_4.shape[0])
        time_max = max(time_max, time_4[-1])
        plt.plot(time_4, traj_4[:, 2]*180/np.pi, label=label_4, linewidth=3)
        y_max = max(traj_4[:, 2].max(), y_max)
        y_min = min(traj_4[:, 2].min(), y_min)

    if horizon is not None:
        num_h = time_max//horizon
        for h in range(1, num_h +2):
            plt.plot([h*horizon, h*horizon], [y_min*180/np.pi, y_max*180/np.pi], color="black", linestyle="dotted", linewidth=1)

    min_angle = env.angle_range[0]*180/np.pi
    max_angle = env.angle_range[1]*180/np.pi
    plt.plot([0., time_max], [min_angle, min_angle], color="red", linestyle="dashed", linewidth=1)
    plt.plot([0., time_max], [max_angle, max_angle], color="red", linestyle="dashed", linewidth=1)
    plt.ylabel("Top angle (deg)")
    plt.xlabel("timesteps")
    ax.set_ylim([max(-30, (y_min-0.1*abs(y_min))*180/np.pi), min(30, y_max*180/np.pi*1.1)])
    plt.legend(frameon=False, labelspacing=0.3, handletextpad=0.2, handlelength=0.9, loc=legend_loc)
    if saveas is not None:
        plt.savefig(saveas + "_angle.svg", bbox_inches='tight', format="svg", dpi=1200)
    plt.show()       

    
    if plot_height:
        fig, ax = nice_plot()
        if title is not None:
            plt.title(title)
        plt.plot(time_1, traj_1[:, 1], label=label_1, linewidth=3)
        plt.plot(time_2, traj_2[:, 1], label=label_2, linewidth=3)
        y_max = max(traj_1[:, 1].max(), traj_2[:, 1].max())
        y_min = min(traj_1[:, 1].min(), traj_2[:, 1].min())
        if traj_3 is not None:
            plt.plot(time_3, traj_3[:, 1], label=label_3, linewidth=3)
            y_max = max(traj_3[:, 1].max(), y_max)
            y_min = min(traj_3[:, 1].min(), y_min)
        if traj_4 is not None:
            plt.plot(time_4, traj_4[:, 1], label=label_4, linewidth=3)
            y_max = max(traj_4[:, 1].max(), y_max)
            y_min = min(traj_4[:, 1].min(), y_min)
        plt.plot([0., time_max], [env.min_z, env.min_z], color="red", linestyle="dashed", linewidth=1)
        plt.ylabel("Hopper height (m)")
        plt.xlabel("timesteps")
        ax.set_ylim([max(-5, y_min-0.1*abs(y_min)), min(5, y_max*1.1)])
        plt.legend(frameon=False, labelspacing=0.3, handletextpad=0.2, handlelength=0.9)
        if saveas is not None:
            plt.savefig(saveas + "_height.svg", bbox_inches='tight', format="svg", dpi=1200)
        plt.show()


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


