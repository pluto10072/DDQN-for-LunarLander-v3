import torch
import torch.nn as nn
import random
import numpy as np
import gymnasium as gym
from collections import deque

class QNet(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_actions)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, n_states, n_actions, lr=1e-3, gamma=0.99):
        self.net = QNet(n_states, n_actions)
        self.target_net = QNet(n_states, n_actions)
        self.target_net.load_state_dict(self.net.state_dict())
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.memory = deque(maxlen=10_000)
        self.criterion = nn.MSELoss()

        self.learn_step = 0
        self.target_update_freq = 100  # 目标网络更新频率

    def get_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, env.action_space.n - 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return self.net(state).argmax().item()

    def save2memory(self, transition):
        self.memory.append(transition)

    def learn(self, batch_size):
        if len(self.memory) < batch_size:
            return

        # 从经验池采样
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([t[0] for t in batch]))
        actions = torch.LongTensor(np.array([t[1] for t in batch]))
        rewards = torch.FloatTensor(np.array([t[2] for t in batch]))
        next_states = torch.FloatTensor(np.array([t[3] for t in batch]))
        dones = torch.FloatTensor(np.array([t[4] for t in batch]))

        current_q = self.net(states).gather(1, actions.unsqueeze(1))
        # 使用在线网络选择动作
        next_actions = self.net(next_states).argmax(1, keepdim=True)
        # 使用目标网络评估动作
        next_q = self.target_net(next_states).gather(1, next_actions).detach()
        target_q = rewards.unsqueeze(1) + self.gamma * next_q * (1 - dones.unsqueeze(1))

        # 计算损失
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 定期更新目标网络
        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.net.state_dict())


# 超参数
batch_size = 128
lr = 1e-4
episodes = 500
target_score = 250.0
gamma = 0.99
max_steps = 1000
eps_start = 1.0
eps_decay = 0.995
eps_min = 0.05

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 环境初始化
env = gym.make('LunarLander-v3')
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n
agent = DQNAgent(n_states, n_actions, lr=lr, gamma=gamma)

# 训练循环
score_hist = []
epsilon = eps_start

for epoch in range(episodes):
    state, _ = env.reset()
    score = 0
    for _ in range(max_steps):
        action = agent.get_action(state, epsilon)
        next_state, reward, done, truncated, _ = env.step(action)
        agent.save2memory((state, action, reward, next_state, done))
        state = next_state
        score += reward
        agent.learn(batch_size)

        if done or truncated:
            break

    score_hist.append(score)
    print(f"Ep {epoch + 1:4d} | Score: {score:7.2f} | Eps: {epsilon:.3f} "
          f"| Avg: {np.mean(score_hist[-100:]):.2f}")
    epsilon = max(eps_min, epsilon * eps_decay)

    if (epoch + 1) % 100 == 0:
        avg_score = np.mean(score_hist[-100:])
        print(f"Avg Score ({epoch - 99}-{epoch + 1}): {avg_score:.2f}")
        if avg_score >= target_score:
            print("\nTarget Reached!")
            break

torch.save(agent.net.state_dict(), 'LunarLander_DDQN.pth')

import matplotlib.pyplot as plt

# 计算滑动平均奖励（窗口大小为100）
window_size = 100
avg_scores = [np.mean(score_hist[max(0, i-window_size+1):i+1])
             for i in range(len(score_hist))]

plt.figure(figsize=(12, 6))

# 绘制单局奖励（浅色）
plt.plot(score_hist,
         color='skyblue',
         alpha=0.4,
         linewidth=0.8,
         label='Episode Reward')

# 绘制滑动平均（深色）
plt.plot(avg_scores,
         color='darkblue',
         linewidth=2,
         label=f'Moving Average ({window_size} episodes)')

# 标注目标线
plt.axhline(y=target_score,
            color='red',
            linestyle='--',
            linewidth=1,
            label='Target Score')

# 美化图表
plt.title('LunarLander DDQN Training Progress', fontsize=14, pad=20)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.legend(loc='upper left', frameon=False)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存高清图片
plt.savefig('training_progress.png', dpi=300)
plt.show()