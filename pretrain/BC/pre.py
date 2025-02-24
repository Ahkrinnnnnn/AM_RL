# 标准库
import numpy as np
import os

# 第三方库
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# 常量定义
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
NUM_EPOCHS = 150
DATA_PATH = os.path.dirname(__file__) + "/../dataset.npy"
MODEL_SAVE_PATH = os.path.dirname(__file__) + "/pretraining_model.pth"
NORM_PARAMS_SAVE_PATH = os.path.dirname(__file__) + "/norm_params.npz"

# 自定义 Dataset
class BCDataset(Dataset):
    def __init__(self, states, targets, actions):
        self.states = states
        self.targets = targets
        self.actions = actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            "state": self.states[idx],
            "target": self.targets[idx],
            "expert_action": self.actions[idx]
        }

# 定义行为克隆策略网络
class BehaviorCloningPolicy(nn.Module):
    def __init__(self, state_dim, target_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + target_dim, 256),
            nn.Softmax(),
            nn.Linear(256, 128),
            nn.Softmax(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state, target):
        x = torch.cat([state, target], dim=-1)
        return self.fc(x)

# 评估函数
def evaluate(model, test_dataloader, criterion):
    model.eval()
    total_mse = 0.0
    with torch.no_grad():
        for batch in test_dataloader:
            state = batch["state"]
            target = batch["target"]
            expert_action = batch["expert_action"]
            pred_action = model(state, target)
            total_mse += criterion(pred_action, expert_action).item()
    avg_mse = total_mse / len(test_dataloader)
    print(f"Test MSE: {avg_mse:.4f}")
    return avg_mse

# 主程序
if __name__ == "__main__":
    # 加载数据集
    try:
        dataset = np.load(DATA_PATH, allow_pickle=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file {DATA_PATH} not found!")

    # 初始化列表以存储数据
    states, targets, expert_actions = [], [], []

    # 提取数据
    for traj in dataset:
        target_pos = traj["target_pos"]
        track_states = traj["track_state"]   # 形状: [T, 19]
        track_controls = traj["track_control"]  # 形状: [T, 9]
    # 同步遍历每个时间步
        for t in range(len(track_states)-1):
            states.append(track_states[t])
            targets.append(target_pos)        # 目标位置固定
            expert_actions.append(track_controls[t])


    # 转换为 NumPy 数组
    states = np.array(states)
    targets = np.array(targets)
    expert_actions = np.array(expert_actions)

    # 数据增强：添加噪声
    noisy_states = states + np.random.randn(*states.shape) * 0.01

    # 数据归一化
    state_mean, state_std = noisy_states.mean(axis=0), noisy_states.std(axis=0)
    target_mean, target_std = targets.mean(axis=0), targets.std(axis=0)
    states_normalized = (noisy_states - state_mean) / (state_std + 1e-8)
    targets_normalized = (targets - target_mean) / (target_std + 1e-8)

    print("State mean:", state_mean)
    print("State std:", state_std)
    print("Target mean:", target_mean)
    print("Target std:", target_std)


    # 保存归一化参数（关键！用于后续部署）
    np.savez(
        NORM_PARAMS_SAVE_PATH,
        state_mean=state_mean,
        state_std=state_std,
        target_mean=target_mean,
        target_std=target_std
    )

    # 转换为 PyTorch Tensor
    states = torch.FloatTensor(states_normalized)
    targets = torch.FloatTensor(targets_normalized)
    expert_actions = torch.FloatTensor(expert_actions)

    # 划分训练集和测试集
    train_states, test_states, train_targets, test_targets, train_actions, test_actions = train_test_split(
        states, targets, expert_actions, test_size=0.2, random_state=42
    )

    # 创建 DataLoader
    train_dataset = BCDataset(train_states, train_targets, train_actions)
    test_dataset = BCDataset(test_states, test_targets, test_actions)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型
    state_dim = train_states.shape[1]
    target_dim = train_targets.shape[1]
    action_dim = train_actions.shape[1]
    model = BehaviorCloningPolicy(state_dim, target_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    criterion = nn.MSELoss()

    # 训练循环
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for batch in train_dataloader:
            state = batch["state"]
            target = batch["target"]
            expert_action = batch["expert_action"]

            optimizer.zero_grad()
            pred_action = model(state, target)
            loss = criterion(pred_action, expert_action)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss/len(train_dataloader):.4f}")

    # 测试模型
    evaluate(model, test_dataloader, criterion)

    # 保存模型参数
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
