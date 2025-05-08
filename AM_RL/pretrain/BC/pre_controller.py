import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import AM_RL
from AM_RL.Controller.model.controller import ControllerNetwork

def load_data(dataset_path):
    try:
        dataset = np.load(dataset_path, allow_pickle=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file {dataset_path} not found!")

    states_mid = np.array([
        0, 0, 5,
        0, 0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,

        0, 0, 5,

        0, 0, 5,
        0, 0, 0, 0,
        0, 0, 0
    ])
    states_half_range = np.array([
        20, 20, 10,
        2, 2, 2, 2,
        20, 20, 20,
        20, 20, 20, 
        3.3415926535897932, 3.3415926535897932, 3.3415926535897932,
        20, 20, 20, 
        
        20, 20, 10,

        20, 20, 10,
        2, 2, 2, 2,
        3.3415926535897932, 3.3415926535897932, 3.3415926535897932,
    ]) / 2

    action_mid = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    action_half_range = np.array([20, 20, 20, 20, 20, 20, 3, 3, 1])

    np.savez(
        norm_params_save_path, 
        states_mid=states_mid, states_range=states_half_range,
        action_mid=action_mid, action_range=action_half_range
    )

    states, actions = [], []
    for trajectory in dataset:
        dx = len(trajectory["planned_path"]) // len(trajectory["track_state"])
        for i in range(len(trajectory["track_state"])-1):
            s = np.concatenate((
                    trajectory["track_state"][i],
                    trajectory["target_pos"],
                    trajectory["planned_path"][dx*i][:7],
                    trajectory["planned_path"][dx*i][13:16]
                    ), axis=0
                )
            if all(-1 <= x <= 1 for x in (s-states_mid)/states_half_range):
                states.append(s)
                actions.append(trajectory["track_control"][i])

    states = np.array(states)
    actions = np.array(actions)
    states_tensor = torch.tensor((states-states_mid)/states_half_range, dtype=torch.float32, device="cpu")
    actions_tensor = torch.tensor((actions-action_mid)/action_half_range, dtype=torch.float32, device="cpu")

    # np.set_printoptions(threshold=np.inf)
    # print(np.sum((states-states_mid)/states_range > 1))
    # print(np.max(states, axis=0))
    # print(np.min(states, axis=0))
    # print(np.max(actions, axis=0))
    # print(np.min(actions, axis=0))

    return states_tensor, actions_tensor

def train_behavior_cloning(states_tensor, actions_tensor, model, model_save_path, batch_size=64, epochs=100):
    dataset = TensorDataset(states_tensor, actions_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for param in model.value_net.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(model.policy_net.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        train_loss = train_step(model, train_dataloader, optimizer, criterion)
        val_loss = val_step(model, val_dataloader, criterion)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.2e}, Val Loss: {val_loss:.2e}")

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")

def train_step(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for state_batch, action_batch in dataloader:
        optimizer.zero_grad()
        predicted_action = model(state_batch)[0]
        loss = criterion(predicted_action, action_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def val_step(model, dataloader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for state_batch, action_batch in dataloader:
            predicted_action = model(state_batch)[0]
            loss = criterion(predicted_action, action_batch)
            val_loss += loss.item()
    return val_loss / len(dataloader)

if __name__ == "__main__":
    package_path = os.path.dirname(os.path.abspath(AM_RL.__file__))
    dataset_path = package_path + "/pretrain/dataset.npy"
    model_save_path = package_path + "/Controller/model/pretraining_controller.pth"
    norm_params_save_path = package_path + "/Controller/model/cnorm_params.npz"

    states_tensor, actions_tensor = load_data(dataset_path)

    actor_model = ControllerNetwork(pretraining=True)

    train_behavior_cloning(states_tensor, actions_tensor, actor_model, model_save_path)
