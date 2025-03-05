"""《DDPG算法的代码》
    时间：2024.12
    环境：gym
    作者：不去幼儿园
"""

import gym  # 导入 Gym 库，用于创建和管理强化学习环境
import numpy as np  # 导入 NumPy，用于处理数组和数学运算
import torch  # 导入 PyTorch，用于构建和训练神经网络
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torch.optim as optim  # 导入 PyTorch 的优化器模块
from collections import deque  # 导入双端队列，用于实现经验回放池
import random  # 导入随机模块，用于从经验池中采样


# 定义 Actor 网络（策略网络）
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256)  # 输入层到隐藏层1，大小为 256
        self.layer2 = nn.Linear(256, 256)  # 隐藏层1到隐藏层2，大小为 256
        self.layer3 = nn.Linear(256, action_dim)  # 隐藏层2到输出层，输出动作维度
        self.max_action = max_action  # 动作的最大值，用于限制输出范围

    def forward(self, state):
        x = torch.relu(self.layer1(state))  # 使用 ReLU 激活函数处理隐藏层1
        x = torch.relu(self.layer2(x))  # 使用 ReLU 激活函数处理隐藏层2
        x = torch.tanh(self.layer3(x)) * self.max_action  # 使用 Tanh 激活函数，并放大到动作范围
        return x  # 返回输出动作


# 定义 Critic 网络（价值网络）
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 256)  # 将状态和动作拼接后输入到隐藏层1
        self.layer2 = nn.Linear(256, 256)  # 隐藏层1到隐藏层2，大小为 256
        self.layer3 = nn.Linear(256, 1)  # 隐藏层2到输出层，输出 Q 值

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)  # 将状态和动作拼接为单个输入
        x = torch.relu(self.layer1(x))  # 使用 ReLU 激活函数处理隐藏层1
        x = torch.relu(self.layer2(x))  # 使用 ReLU 激活函数处理隐藏层2
        x = self.layer3(x)  # 输出 Q 值
        return x  # 返回 Q 值


# 定义经验回放池
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)  # 初始化一个双端队列，设置最大容量

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))  # 将经验存入队列

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)  # 随机采样一个小批量数据
        states, actions, rewards, next_states, dones = zip(*batch)  # 解压采样数据
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))  # 返回 NumPy 数组格式的数据

    def size(self):
        return len(self.buffer)  # 返回经验池中当前存储的样本数量


# DDPG智能体类定义
class DDPGAgent:
    # 初始化方法，设置智能体的参数和模型
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005, buffer_size=100000, batch_size=64):
        # 定义actor网络（策略网络）及其目标网络
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        # 将目标actor网络的参数初始化为与actor网络一致
        self.actor_target.load_state_dict(self.actor.state_dict())
        # 定义actor网络的优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        # 定义critic网络（值网络）及其目标网络
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        # 将目标critic网络的参数初始化为与critic网络一致
        self.critic_target.load_state_dict(self.critic.state_dict())
        # 定义critic网络的优化器
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # 保存动作的最大值，用于限制动作范围
        self.max_action = max_action
        # 折扣因子，用于奖励的时间折扣
        self.gamma = gamma
        # 软更新系数，用于目标网络的更新
        self.tau = tau
        # 初始化经验回放池
        self.replay_buffer = ReplayBuffer(buffer_size)
        # 每次训练的批量大小
        self.batch_size = batch_size

    # 选择动作的方法
    def select_action(self, state):
        # 将状态转换为张量
        state = torch.FloatTensor(state.reshape(1, -1))
        # 使用actor网络预测动作，并将结果转换为NumPy数组
        action = self.actor(state).detach().cpu().numpy().flatten()
        return action

    # 训练方法
    def train(self):
        # 如果回放池中样本数量不足，直接返回
        if self.replay_buffer.size() < self.batch_size:
            return

        # 从回放池中采样一批数据
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # 将采样的数据转换为张量
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)  # 添加一个维度以匹配Q值维度
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)  # 添加一个维度以匹配Q值维度

        # 计算critic的损失
        with torch.no_grad():  # 关闭梯度计算
            next_actions = self.actor_target(next_states)  # 使用目标actor网络预测下一步动作
            target_q = self.critic_target(next_states, next_actions)  # 目标Q值
            # 使用贝尔曼方程更新目标Q值
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # 当前Q值
        current_q = self.critic(states, actions)
        # 均方误差损失
        critic_loss = nn.MSELoss()(current_q, target_q)

        # 优化critic网络
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 计算actor的损失
        actor_loss = -self.critic(states, self.actor(states)).mean()  # 策略梯度目标为最大化Q值

        # 优化actor网络
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新目标网络参数（软更新）
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # 将样本添加到回放池中
    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)


# 绘制学习曲线的方法
import matplotlib.pyplot as plt


def train_ddpg(env_name, episodes=1000, max_steps=200):
    # 创建环境
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]  # 状态空间维度
    action_dim = env.action_space.shape[0]  # 动作空间维度
    max_action = float(env.action_space.high[0])  # 动作最大值

    # 初始化DDPG智能体
    agent = DDPGAgent(state_dim, action_dim, max_action)
    rewards = []  # 用于存储每个episode的奖励

    for episode in range(episodes):
        state, _ = env.reset()  # 重置环境，获取初始状态
        episode_reward = 0  # 初始化每轮奖励为0
        for step in range(max_steps):
            # 选择动作
            action = agent.select_action(state)
            # 执行动作，获取环境反馈
            next_state, reward, done, _, _ = env.step(action)
            # 将样本存入回放池
            agent.add_to_replay_buffer(state, action, reward, next_state, done)

            # 训练智能体
            agent.train()
            # 更新当前状态
            state = next_state
            # 累加奖励
            episode_reward += reward

            if done:  # 如果完成（到达终止状态），结束本轮
                break

        # 记录每轮的累计奖励
        rewards.append(episode_reward)
        print(f"Episode: {episode + 1}, Reward: {episode_reward}")

    # 绘制学习曲线
    plt.plot(rewards)
    plt.title("Learning Curve")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.show()

    env.close()  # 关闭环境


# 主函数运行
if __name__ == "__main__":
    # 定义环境名称和训练轮数
    env_name = "Pendulum-v1"
    episodes = 500
    # 开始训练
    train_ddpg(env_name, episodes=episodes)