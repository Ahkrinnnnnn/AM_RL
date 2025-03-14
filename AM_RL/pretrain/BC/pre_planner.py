import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from gymnasium import spaces
import AM_RL
from AM_RL.Planner.model.planner import PlannerNetwork

def load_data(dataset_path):
    try:
        dataset = np.load(dataset_path, allow_pickle=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file {dataset_path} not found!")

    states_mid = np.array([
        0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,

        0, 0, 0,
    ])
    states_range = np.array([
        20, 20, 20,
        2, 2, 2, 2,
        20, 20, 20,
        20, 20, 20, 
        3.3415926535897932, 3.3415926535897932, 3.3415926535897932,
        20, 20, 20, 
        
        20, 20, 20
    ])

    action_mid = np.array([
        0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    ])
    action_range = np.array([
        20, 20, 20,
        2, 2, 2, 2,
        20, 20, 20,
        20, 20, 20, 
        3.3415926535897932, 3.3415926535897932, 3.3415926535897932,
        20, 20, 20, 
    ])

    # np.savez(
    #     norm_params_save_path, 
    #     states_mid=states_mid, states_range=states_range,
    #     action_mid=action_mid, action_range=action_range
    # )

    states, actions = [], []
    for trajectory in dataset:
        for i in range(len(trajectory["planned_path"])-1):
            s = np.concatenate((
                    trajectory["planned_path"][i],
                    trajectory["target_pos"]), axis=0
                )
            a = trajectory["planned_path"][i+1]
            if all(-1 <= x <= 1 for x in (s-states_mid)/states_range):
                states.append(s)
                actions.append(a)

    states = np.array(states)
    actions = np.array(actions)
    states_tensor = torch.tensor((states-states_mid)/states_range, dtype=torch.float32)
    actions_tensor = torch.tensor((actions-action_mid)/action_range, dtype=torch.float32)

    # np.set_printoptions(threshold=np.inf)
    # print(np.sum((states-states_mid)/states_range > 1))
    # print(np.max(states, axis=0))
    # print(np.min(states, axis=0))
    # print(np.max(states, axis=0))
    # print(np.min(states, axis=0))

    return states_tensor, actions_tensor

def train_behavior_cloning(states_tensor, actions_tensor, model, model_save_path, batch_size=64, epochs=20):
    dataset = TensorDataset(states_tensor, actions_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
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
        predicted_action = model(state_batch)
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
            predicted_action = model(state_batch)
            loss = criterion(predicted_action, action_batch)
            val_loss += loss.item()
    return val_loss / len(dataloader)

if __name__ == "__main__":
    package_path = os.path.dirname(os.path.abspath(AM_RL.__file__))
    dataset_path = package_path + "/pretrain/dataset.npy"
    model_save_path = package_path + "/Planner/model/pretraining_planner.pth"
    norm_params_save_path = package_path + "/Planner/model/pnorm_params.npz"

    states_tensor, actions_tensor = load_data(dataset_path)

    state_dim = spaces.Box(low=-1, high=1, shape=(states_tensor.shape[1],), dtype=np.float32)
    action_dim = spaces.Box(low=-1, high=1, shape=(actions_tensor.shape[1],), dtype=np.float32)

    td3_model = PlannerNetwork(state_dim, action_dim)

    train_behavior_cloning(states_tensor, actions_tensor, td3_model, model_save_path)
