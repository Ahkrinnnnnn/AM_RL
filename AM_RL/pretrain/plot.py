import os
import numpy as np
import matplotlib.pyplot as plt

import AM_RL

package_path = os.path.dirname(os.path.abspath(AM_RL.__file__))
dataset_path = package_path + "/pretrain/dataset.npy"
dataset = np.load(dataset_path, allow_pickle=True)

def deal_data(dataset):
    target = dataset["target_pos"]
    plan = dataset["planned_path"]
    real_state = dataset["track_state"]
    control = dataset["track_control"]
    return target, plan, real_state, control

def draw_plan(d, vs=False):
    target, plan, real_state, _ = deal_data(d)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(target[0], target[1], target[2], 
            marker='*', linestyle='-', markersize=4,
            color='green', label='Target')

    ax.plot(plan[:, 0], plan[:, 1], plan[:, 2], 
            marker='o', linestyle='-', markersize=4,
            color='blue', label='Planned Trajectory')
    
    if vs:
        ax.plot(real_state[:, 0], real_state[:, 1], real_state[:, 2], 
            marker='s', linestyle='-', markersize=4,
            color='red', label='True Trajectory')
        
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    
    ax.legend(loc='upper right', fontsize=10)
    
    ax.set_xlim(-1, 8)
    ax.set_ylim(-1, 8)
    ax.set_zlim(0, 2)
    
    ax.view_init(elev=20, azim=30)
    
    plt.show()

def draw_control(d):
    _, _, _, control = deal_data(d)

    time_points = np.linspace(0, 20, len(control))

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    axs[0].plot(time_points, control[:, 0], marker='o', linestyle='-', color='blue', markersize=4, label='p1')
    axs[0].plot(time_points, control[:, 1], marker='o', linestyle='-', color='red', markersize=4, label='p2')
    axs[0].plot(time_points, control[:, 2], marker='o', linestyle='-', color='green', markersize=4, label='p3')
    axs[0].plot(time_points, control[:, 3], marker='o', linestyle='-', color='cyan', markersize=4, label='p4')
    axs[0].plot(time_points, control[:, 4], marker='o', linestyle='-', color='black', markersize=4, label='p5')
    axs[0].plot(time_points, control[:, 5], marker='o', linestyle='-', color='yellow', markersize=4, label='p6')

    axs[0].set_ylabel("Thrust [N]", fontsize=12)
    axs[0].legend(loc='upper right', fontsize=10)
    axs[0].grid(True, linestyle='--', alpha=0.7)

    axs[1].plot(time_points, control[:, 6], marker='o', linestyle='-', color='blue', markersize=4, label='j1')
    axs[1].plot(time_points, control[:, 7], marker='o', linestyle='-', color='red', markersize=4, label='j2')
    axs[1].plot(time_points, control[:, 8], marker='o', linestyle='-', color='green', markersize=4, label='j3')

    axs[1].set_xlabel("Time (s)", fontsize=12)
    axs[1].set_ylabel("Torques [Nm]", fontsize=12)
    axs[1].legend(loc='upper right', fontsize=10)
    axs[1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    plt.show()

for i in range(250):
    draw_plan(dataset[i], vs=True)

# draw_control()