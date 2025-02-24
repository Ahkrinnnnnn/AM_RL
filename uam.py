import genesis as gs
import numpy as np
import torch
from src.uam_env import UamEnv
from src.models.planner import Node, APlanner
from src.models.controller import PPO, Memory
from cfgs.robotCfg import *
from cfgs.hyperparaCfg import *
from cfgs.rewardCfg import *

def main():
    gs.init(backend=gs.cpu)
    env = UamEnv(urdf=urdfPath, work_area=workArea, show_viewer=True, device="cpu")
    obj = env.obj.get_pos() + [0, 0, 2]
    # print(env.robot.links)
    catPath = []
    start = Node(tuple((env.robot.get_pos().detach().tolist())))
    goal = Node(tuple((env.obj.get_pos().detach().tolist())))
    aStarPlanner1 = APlanner(start, goal, env.obstacles, env.work_area)
    aStarPlanner2 = APlanner(goal, obj, env.obstacles, env.work_area).plan()
    catPath = torch.cat([aStarPlanner1.plan(), aStarPlanner2.plan()], dim=0)
    # aStarPlanner.vis(env.scene, catPath[i])

    state_dim = 19 + 3 + 3
    action_dim = 9
    ppo = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, save_freq)

    ref = 0
    memory = Memory()
    for episode in range(1000):
        state = torch.cat([env.get_full_states(), 
                        env.end_effector.get_pos()-env.obj.get_pos(),
                        catPath[ref]], dim=0)
        done = False
        while not done:
            action = ppo.select_action(state)
            env.step(action)
            if (state[:3]-state[22:]).norm(p=2) < rewardsCfg["traj_thresh"]:
                ref += 1
                if ref == len(catPath):
                    done = True
            next_state = torch.cat(env.get_full_states(), 
                        env.obj.get_pos(),
                        catPath[ref])
            reward = env.compute_reward(rewardsCfg, state, next_state)
            memory.states.append(state)
            memory.actions.append(action)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            state = next_state
        ppo.update(memory)
        memory.clear_memory()
        env.scene.reset()
        if episode == ppo.save_freq:
            torch.save(ppo.state_dict(), savePath)

if __name__ == "__main__":
    main()
