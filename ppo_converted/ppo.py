import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class PolicyNetwork(nn.Module):
    """策略网络：输入状态，输出动作概率分布"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        action_probs = torch.softmax(self.fc3(x), dim=-1)
        return action_probs
    
    def get_action(self, state):
        """根据策略采样动作"""
        with torch.no_grad():
            probs = self.forward(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()


class ValueNetwork(nn.Module):
    """价值网络：输入状态，输出状态价值"""
    def __init__(self, state_dim, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        value = self.fc3(x)
        return value


class PPO:
    """PPO算法实现"""
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 epsilon=0.2, c1=1.0, c2=0.01, k_epochs=10, batch_size=64):
        # 网络
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        self.policy_old = PolicyNetwork(state_dim, action_dim)
        
        # 优化器
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        # 超参数
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 裁剪参数
        self.c1 = c1  # 价值函数损失系数
        self.c2 = c2  # 熵奖励系数
        self.k_epochs = k_epochs  # 优化周期数
        self.batch_size = batch_size  # 小批量大小
        
        # 同步旧策略
        self.policy_old.load_state_dict(self.policy_net.state_dict())
    
    def select_action(self, state):
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob = self.policy_net.get_action(state_tensor)
        return action, log_prob
    
    def compute_gae(self, rewards, values, next_value, dones):
        """计算广义优势估计（GAE）"""
        advantages = []
        gae = 0
        next_value = next_value
        
        for step in reversed(range(len(rewards))):
            if dones[step]:
                delta = rewards[step] - values[step]
                gae = delta
            else:
                delta = rewards[step] + self.gamma * values[step + 1] - values[step]
                gae = delta + self.gamma * 0.95 * gae  # lambda = 0.95
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns
    
    def update(self, states, actions, old_log_probs, rewards, next_states, dones):
        """更新策略和价值网络"""
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        rewards = np.array(rewards)
        dones = np.array(dones)
        
        # 计算价值函数
        with torch.no_grad():
            values = self.value_net(states).squeeze().numpy()
            next_value = self.value_net(torch.FloatTensor(next_states[-1:])).item()
        
        # 计算优势函数和回报
        advantages, returns = self.compute_gae(rewards, values, next_value, dones)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 多次优化
        for epoch in range(self.k_epochs):
            # 随机打乱数据
            indices = np.random.permutation(len(states))
            
            # 小批量训练
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 计算当前策略的概率
                action_probs = self.policy_net(batch_states)
                dist = torch.distributions.Categorical(action_probs)
                log_probs = dist.log_prob(batch_actions)
                
                # 计算概率比
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # 计算L^CLIP
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 计算熵奖励
                entropy = dist.entropy().mean()
                
                # 计算价值函数损失
                value_pred = self.value_net(batch_states).squeeze()
                value_loss = nn.MSELoss()(value_pred, batch_returns)
                
                # 总损失
                total_loss = policy_loss + self.c1 * value_loss - self.c2 * entropy
                
                # 更新策略网络
                self.policy_optimizer.zero_grad()
                policy_loss.backward(retain_graph=True)
                self.policy_optimizer.step()
                
                # 更新价值网络
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()
        
        # 更新旧策略
        self.policy_old.load_state_dict(self.policy_net.state_dict())


class SimpleEnv:
    """简单的环境用于测试"""
    def __init__(self):
        self.state_dim = 4
        self.action_dim = 2
        self.state = np.random.randn(4)
        self.step_count = 0
        self.max_steps = 200
    
    def reset(self):
        self.state = np.random.randn(4)
        self.step_count = 0
        return self.state
    
    def step(self, action):
        self.step_count += 1
        # 简单的奖励函数
        reward = 1.0 if action == 0 else 0.5
        # 简单的状态转移
        self.state = self.state + np.random.randn(4) * 0.1
        done = self.step_count >= self.max_steps
        return self.state, reward, done, {}


def train_ppo(env, agent, num_episodes=1000, max_steps=200):
    """训练PPO算法"""
    episode_rewards = []
    
    for episode in range(num_episodes):
        # 收集数据
        states = []
        actions = []
        old_log_probs = []
        rewards = []
        next_states = []
        dones = []
        
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # 选择动作
            action, log_prob = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 存储数据
            states.append(state)
            actions.append(action)
            old_log_probs.append(log_prob)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # 更新策略
        agent.update(states, actions, old_log_probs, rewards, next_states, dones)
        
        episode_rewards.append(episode_reward)
        
        # 打印进度
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")
    
    return episode_rewards


# 主函数
if __name__ == "__main__":
    # 创建环境
    env = SimpleEnv()
    
    # 创建PPO智能体
    agent = PPO(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        lr=3e-4,
        gamma=0.99,
        epsilon=0.2,
        c1=1.0,
        c2=0.01,
        k_epochs=10,
        batch_size=64
    )
    
    # 训练
    print("开始训练PPO算法...")
    rewards = train_ppo(env, agent, num_episodes=1000, max_steps=200)
    print("训练完成！")
    
    # 绘制训练曲线（可选）
    try:
        import matplotlib.pyplot as plt
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('PPO Training Curve')
        plt.show()
    except ImportError:
        print("matplotlib未安装，跳过绘图")