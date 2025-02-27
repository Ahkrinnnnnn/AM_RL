import genesis as gs
import torch
from src.uam_env import UamEnv
from cfgs.robotCfg import *
from cfgs.hyperparaCfg import *
from cfgs.rewardCfg import *

def main():
    gs.init(backend=gs.cpu)
    env = UamEnv(urdf=urdfPath, work_area=workArea, show_viewer=True, device="cpu")
    obj = env.obj.get_pos() + [0, 0, 2]

    for episode in range(1000):
        state = env.get_img()
        done = False
        while not done:
            action = ppo.select_action(state)
            env.step(action)
            if (env.obj.get_pos()-obj).norm(p=2) < rewardsCfg["traj_thresh"]:
                done = True
            
            next_state = env.get_img()
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
