import gymnasium as gym
import torch
import torch.nn as nn

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

env = gym.make("LunarLander-v3", render_mode='human')
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
net = QNet(input_dim, output_dim)
net.load_state_dict(torch.load('LunarLander_DDQN.pth', weights_only=True))  # 加载训练好的模型
net.eval()  # 切换到评估模式

while True:
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # 将状态转换为张量
        with torch.no_grad():
            q_values = net(state_tensor)
        action = q_values.argmax().item()  # 选择最优动作

        # 执行动作并获取下一状态
        next_state, reward, done, truncated, _ = env.step(action)

        total_reward += reward
        state = next_state  # 状态转移
        done = done or truncated

    print(f"Score: {int(total_reward)}")