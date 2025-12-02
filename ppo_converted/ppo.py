# -*- coding: utf-8 -*-
# 文件编码声明，支持中文字符

import torch  # PyTorch深度学习框架，用于构建神经网络和自动求导
import torch.nn as nn  # PyTorch神经网络模块，包含各种层和激活函数
import torch.optim as optim  # PyTorch优化器模块，包含Adam、SGD等优化算法
import numpy as np  # NumPy数值计算库，用于数组操作和数学运算
from collections import deque  # Python双端队列，用于存储经验回放缓冲区（本代码中未使用）

class PolicyNetwork(nn.Module):
    """
    策略网络（Policy Network）：输入状态，输出动作概率分布
    策略网络 π(a|s) 表示在状态 s 下选择动作 a 的概率
    公式：π_θ(a|s) = softmax(f_θ(s))
    其中 f_θ(s) 是神经网络的输出，θ 是网络参数
    """
    def __init__(self, state_dim, action_dim, hidden_dim=4):
        """
        初始化策略网络
        Args:
            state_dim: 状态维度，本环境中为3（x, a_est, b_est）
            action_dim: 动作维度，本环境中为4（增加a/减少a/增加b/减少b）
            hidden_dim: 隐藏层维度，默认4
        """
        super(PolicyNetwork, self).__init__()  # 调用父类nn.Module的初始化方法
        
        # 第一层全连接层：state_dim -> hidden_dim
        # 公式：h1 = W1 * s + b1，其中W1是权重矩阵，b1是偏置向量
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        
        # 第二层全连接层：hidden_dim -> hidden_dim
        # 公式：h2 = W2 * h1 + b2
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 第三层全连接层（输出层）：hidden_dim -> action_dim
        # 公式：logits = W3 * h2 + b3，输出未归一化的动作分数
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        """
        前向传播：计算动作概率分布
        公式：π(a|s) = softmax(tanh(W3 * tanh(W2 * tanh(W1 * s + b1) + b2) + b3))
        Args:
            state: 输入状态，形状为 (batch_size, state_dim)
        Returns:
            action_probs: 动作概率分布，形状为 (batch_size, action_dim)
        """
        # 第一层：线性变换 + tanh激活函数
        # 公式：h1 = tanh(W1 * s + b1)
        # tanh函数：tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))，输出范围[-1, 1]
        x = torch.tanh(self.fc1(state))
        
        # 第二层：线性变换 + tanh激活函数
        # 公式：h2 = tanh(W2 * h1 + b2)
        x = torch.tanh(self.fc2(x))
        
        # 输出层：线性变换 + softmax归一化
        # 公式：action_probs = softmax(W3 * h2 + b3)
        # softmax函数：softmax(x_i) = exp(x_i) / Σ_j exp(x_j)，将logits转换为概率分布
        action_probs = torch.softmax(self.fc3(x), dim=-1)
        
        return action_probs  # 返回动作概率分布，所有动作概率之和为1
    
    def get_action(self, state):
        """
        根据策略采样动作
        使用Categorical分布进行采样：a ~ π_θ(·|s)
        公式：P(a=i|s) = π_θ(a=i|s)，动作i的采样概率等于策略网络输出的概率
        Args:
            state: 输入状态
        Returns:
            action: 采样的动作（整数）
            log_prob: 动作的对数概率 log π_θ(a|s)
        """
        with torch.no_grad():  # 禁用梯度计算，因为只是采样动作，不需要反向传播
            # 计算动作概率分布
            # 公式：π(a|s) = forward(state)
            probs = self.forward(state)
            
            # 创建分类分布（Categorical Distribution）
            # 公式：P(a=i) = probs[i]，每个动作的概率由probs给出
            dist = torch.distributions.Categorical(probs)
            
            # 从分布中采样一个动作
            # 公式：a ~ Categorical(π(a|s))，根据概率分布随机采样
            action = dist.sample()
            
            # 计算采样动作的对数概率
            # 公式：log_prob = log π_θ(a|s)，用于后续的PPO损失计算
            log_prob = dist.log_prob(action)
        
        # 将tensor转换为Python标量并返回
        return action.item(), log_prob.item()


class ValueNetwork(nn.Module):
    """
    价值网络（Value Network）：输入状态，输出状态价值
    价值网络 V(s) 估计状态 s 的期望累积回报
    公式：V_φ(s) = E[Σ_{t=0}^∞ γ^t * r_t | s_0=s]
    其中 γ 是折扣因子，r_t 是时刻t的奖励，φ 是网络参数
    """
    def __init__(self, state_dim, hidden_dim=2):
        """
        初始化价值网络
        Args:
            state_dim: 状态维度，本环境中为3
            hidden_dim: 隐藏层维度，默认2
        """
        super(ValueNetwork, self).__init__()  # 调用父类nn.Module的初始化方法
        
        # 第一层全连接层：state_dim -> hidden_dim
        # 公式：h1 = W1 * s + b1
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        
        # 第二层全连接层：hidden_dim -> hidden_dim
        # 公式：h2 = W2 * h1 + b2
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 输出层：hidden_dim -> 1（输出标量价值）
        # 公式：V(s) = W3 * h2 + b3
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        """
        前向传播：计算状态价值
        公式：V_φ(s) = W3 * tanh(W2 * tanh(W1 * s + b1) + b2) + b3
        Args:
            state: 输入状态，形状为 (batch_size, state_dim)
        Returns:
            value: 状态价值，形状为 (batch_size, 1)
        """
        # 第一层：线性变换 + tanh激活函数
        # 公式：h1 = tanh(W1 * s + b1)
        x = torch.tanh(self.fc1(state))
        
        # 第二层：线性变换 + tanh激活函数
        # 公式：h2 = tanh(W2 * h1 + b2)
        x = torch.tanh(self.fc2(x))
        
        # 输出层：线性变换，输出状态价值（标量）
        # 公式：V(s) = W3 * h2 + b3
        value = self.fc3(x)
        
        return value  # 返回状态价值估计


class PPO:
    """
    PPO（Proximal Policy Optimization）算法实现
    PPO是一种策略梯度方法，通过限制策略更新幅度来稳定训练
    
    核心公式：
    1. 策略损失（Clipped Surrogate Objective）：
       L^CLIP(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]
       其中 r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) 是重要性采样比率
       A_t 是优势函数，ε 是裁剪参数（通常0.1或0.2）
    
    2. 价值函数损失：
       L^VF(φ) = E[(V_φ(s_t) - R_t)^2]
       其中 R_t 是实际回报
    
    3. 总损失：
       L(θ,φ) = L^CLIP(θ) - c1 * L^VF(φ) + c2 * H[π_θ(·|s_t)]
       其中 H[π_θ] 是策略熵，c1和c2是系数
    """
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 epsilon=0.2, c1=1.0, c2=0.01, k_epochs=10, batch_size=64):
        """
        初始化PPO算法
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            lr: 学习率（learning rate），默认3e-4
            gamma: 折扣因子，用于计算未来奖励的现值，默认0.99
                   公式：G_t = r_t + γ*r_{t+1} + γ^2*r_{t+2} + ...
            epsilon: PPO裁剪参数，限制策略更新幅度，默认0.2
            c1: 价值函数损失系数，默认1.0
            c2: 熵奖励系数，鼓励探索，默认0.01
            k_epochs: 每次更新时对同一批数据重复优化的次数，默认10
            batch_size: 小批量大小，用于批量训练，默认64
        """
        # 创建策略网络（当前策略）
        # 公式：π_θ(a|s)，θ是网络参数
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        
        # 创建价值网络
        # 公式：V_φ(s)，φ是网络参数
        self.value_net = ValueNetwork(state_dim)
        
        # 创建旧策略网络（用于重要性采样）
        # 公式：π_θ_old(a|s)，在每次更新后同步当前策略
        self.policy_old = PolicyNetwork(state_dim, action_dim)
        
        # 创建策略网络优化器（Adam优化器）
        # Adam优化器公式：
        # m_t = β1 * m_{t-1} + (1-β1) * g_t  （一阶矩估计）
        # v_t = β2 * v_{t-1} + (1-β2) * g_t^2  （二阶矩估计）
        # θ_t = θ_{t-1} - lr * m_t / (sqrt(v_t) + ε)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # 创建价值网络优化器
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        # 折扣因子 γ：用于计算未来奖励的现值
        # 公式：G_t = Σ_{k=0}^∞ γ^k * r_{t+k}
        self.gamma = gamma
        
        # PPO裁剪参数 ε：限制策略更新幅度
        # 在PPO损失中使用：clip(r_t(θ), 1-ε, 1+ε)
        self.epsilon = epsilon
        
        # 价值函数损失系数 c1：平衡策略损失和价值损失
        # 在总损失中使用：L = L^CLIP - c1 * L^VF
        self.c1 = c1
        
        # 熵奖励系数 c2：鼓励策略探索
        # 在总损失中使用：L = L^CLIP - c1 * L^VF + c2 * H[π]
        # 熵公式：H[π] = -Σ_a π(a|s) * log π(a|s)
        self.c2 = c2
        
        # 优化周期数 k：每次更新时对同一批数据重复优化的次数
        # 这样可以更充分地利用收集到的数据
        self.k_epochs = k_epochs
        
        # 小批量大小：每次优化时使用的样本数量
        self.batch_size = batch_size
        
        # 同步旧策略：将当前策略的参数复制到旧策略
        # 公式：θ_old = θ，用于计算重要性采样比率 r_t(θ)
        self.policy_old.load_state_dict(self.policy_net.state_dict())
    
    def select_action(self, state):
        """
        选择动作：根据当前策略采样动作
        Args:
            state: 当前状态
        Returns:
            action: 采样的动作
            log_prob: 动作的对数概率 log π_θ(a|s)
        """
        # 将numpy数组转换为PyTorch tensor，并添加batch维度
        # state形状从(state_dim,)变为(1, state_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # 使用策略网络采样动作
        # 公式：a ~ π_θ(·|s)，log_prob = log π_θ(a|s)
        action, log_prob = self.policy_net.get_action(state_tensor)
        
        return action, log_prob
    
    def compute_gae(self, rewards, values, next_value, dones):
        """
        计算广义优势估计（Generalized Advantage Estimation, GAE）
        GAE结合了TD误差和蒙特卡洛估计，平衡偏差和方差
        
        公式：
        1. TD误差：δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        2. GAE：A_t^GAE = δ_t + (γ*λ) * δ_{t+1} + (γ*λ)^2 * δ_{t+2} + ...
        3. 回报：R_t = A_t + V(s_t)
        
        其中 λ 是GAE参数（本代码中设为0.95），γ 是折扣因子
        Args:
            rewards: 奖励序列 [r_0, r_1, ..., r_{T-1}]
            values: 价值函数估计 [V(s_0), V(s_1), ..., V(s_{T-1})]
            next_value: 下一个状态的价值 V(s_T)
            dones: 终止标志序列，表示每个步骤是否结束
        Returns:
            advantages: 优势函数序列 [A_0, A_1, ..., A_{T-1}]
            returns: 回报序列 [R_0, R_1, ..., R_{T-1}]
        """
        advantages = []  # 存储优势函数值
        gae = 0  # 初始化GAE累积值
        next_value = next_value  # 下一个状态的价值（用于计算最后一个TD误差）
        
        # 从后往前遍历，计算GAE
        # 公式：A_t = δ_t + (γ*λ) * A_{t+1}
        for step in reversed(range(len(rewards))):
            if dones[step]:  # 如果当前步骤结束，重置GAE
                # TD误差：δ_t = r_t - V(s_t)（没有下一状态）
                delta = rewards[step] - values[step]
                gae = delta  # 重置GAE为当前TD误差
            else:  # 如果未结束，计算完整的TD误差
                # TD误差：δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
                delta = rewards[step] + self.gamma * values[step + 1] - values[step]
                # GAE：A_t = δ_t + (γ*λ) * A_{t+1}
                # 其中 λ = 0.95（GAE参数）
                gae = delta + self.gamma * 0.95 * gae
        
            # 将GAE插入列表开头（因为是从后往前遍历）
            advantages.insert(0, gae)
        
        # 计算回报：R_t = A_t + V(s_t)
        # 回报是优势函数加上状态价值
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        return advantages, returns
    
    def update(self, states, actions, old_log_probs, rewards, next_states, dones):
        """
        更新策略和价值网络
        这是PPO算法的核心更新步骤
        
        更新流程：
        1. 计算价值函数估计
        2. 计算GAE优势函数和回报
        3. 归一化优势函数
        4. 对同一批数据进行k次优化（k_epochs）
        5. 每次优化时使用小批量（batch_size）进行训练
        6. 计算PPO损失并更新网络参数
        
        Args:
            states: 状态序列
            actions: 动作序列
            old_log_probs: 旧策略的对数概率 log π_θ_old(a|s)
            rewards: 奖励序列
            next_states: 下一个状态序列
            dones: 终止标志序列
        """
        # 将数据转换为PyTorch tensor
        states = torch.FloatTensor(states)  # 状态tensor
        actions = torch.LongTensor(actions)  # 动作tensor（整数类型）
        old_log_probs = torch.FloatTensor(old_log_probs)  # 旧对数概率tensor
        rewards = np.array(rewards)  # 奖励数组
        dones = np.array(dones)  # 终止标志数组
        
        # 计算价值函数估计（不需要梯度）
        with torch.no_grad():  # 禁用梯度计算，因为只是评估
            # 计算所有状态的价值：V_φ(s_t)
            # 公式：V(s) = value_net(s)
            values = self.value_net(states).squeeze().numpy()  # 移除维度并转为numpy
            
            # 计算最后一个下一个状态的价值：V_φ(s_T)
            # 用于计算最后一个TD误差
            next_value = self.value_net(torch.FloatTensor(next_states[-1:])).item()
        
        # 计算优势函数和回报
        # 公式：A_t = GAE(rewards, values)，R_t = A_t + V(s_t)
        advantages, returns = self.compute_gae(rewards, values, next_value, dones)
        
        # 转换为tensor
        advantages = torch.FloatTensor(advantages)  # 优势函数tensor
        returns = torch.FloatTensor(returns)  # 回报tensor
        
        # 归一化优势函数：减少方差，提高训练稳定性
        # 公式：A_normalized = (A - mean(A)) / (std(A) + ε)
        # 其中 ε = 1e-8 是为了避免除以0
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 多次优化：对同一批数据进行k次优化
        # 这样可以更充分地利用收集到的数据
        for epoch in range(self.k_epochs):
            # 随机打乱数据：每次优化时随机打乱，避免过拟合
            indices = np.random.permutation(len(states))
            
            # 小批量训练：将数据分成多个小批量
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size  # 批量结束索引
                batch_indices = indices[start:end]  # 当前批量的索引
                
                # 提取当前批量的数据
                batch_states = states[batch_indices]  # 批量状态
                batch_actions = actions[batch_indices]  # 批量动作
                batch_old_log_probs = old_log_probs[batch_indices]  # 批量旧对数概率
                batch_advantages = advantages[batch_indices]  # 批量优势函数
                batch_returns = returns[batch_indices]  # 批量回报
                
                # 计算当前策略的动作概率分布
                # 公式：π_θ(a|s) = policy_net(s)
                action_probs = self.policy_net(batch_states)
                
                # 创建分类分布
                # 公式：P(a=i|s) = action_probs[i]
                dist = torch.distributions.Categorical(action_probs)
                
                # 计算当前策略的对数概率
                # 公式：log π_θ(a|s) = log P(a|s)
                log_probs = dist.log_prob(batch_actions)
                
                # 计算重要性采样比率
                # 公式：r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
                # 等价于：r_t(θ) = exp(log π_θ(a_t|s_t) - log π_θ_old(a_t|s_t))
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # 计算PPO裁剪损失（Clipped Surrogate Objective）
                # 公式：L^CLIP(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]
                # surr1 = r_t(θ) * A_t（未裁剪的损失）
                surr1 = ratio * batch_advantages
                # surr2 = clip(r_t(θ), 1-ε, 1+ε) * A_t（裁剪后的损失）
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                # 取最小值并取负号（因为要最大化，所以损失函数取负）
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 计算熵奖励：鼓励策略探索
                # 熵公式：H[π] = -Σ_a π(a|s) * log π(a|s)
                # 熵越大，策略越随机，探索越多
                entropy = dist.entropy().mean()
                
                # 计算价值函数损失：最小化价值函数估计误差
                # 公式：L^VF(φ) = E[(V_φ(s_t) - R_t)^2]
                # 其中 R_t 是实际回报（从GAE计算得到）
                value_pred = self.value_net(batch_states).squeeze()  # 价值函数预测
                value_loss = nn.MSELoss()(value_pred, batch_returns)  # 均方误差损失
                
                # 总损失函数
                # 公式：L(θ,φ) = L^CLIP(θ) - c1 * L^VF(φ) + c2 * H[π_θ(·|s_t)]
                # 注意：policy_loss已经是负号，所以这里是相加
                # value_loss前面加负号是因为要最大化价值函数精度（最小化误差）
                # entropy前面加负号是因为要最大化熵（鼓励探索）
                total_loss = policy_loss + self.c1 * value_loss - self.c2 * entropy
                
                # 更新策略网络
                self.policy_optimizer.zero_grad()  # 清零梯度
                policy_loss.backward(retain_graph=True)  # 反向传播计算梯度
                # retain_graph=True是因为后面还要对value_loss进行反向传播
                self.policy_optimizer.step()  # 更新参数：θ = θ - lr * ∇_θ L^CLIP
                
                # 更新价值网络
                self.value_optimizer.zero_grad()  # 清零梯度
                value_loss.backward()  # 反向传播计算梯度
                self.value_optimizer.step()  # 更新参数：φ = φ - lr * ∇_φ L^VF
        
        # 更新旧策略：将当前策略的参数复制到旧策略
        # 公式：θ_old = θ，用于下次更新时计算重要性采样比率
        self.policy_old.load_state_dict(self.policy_net.state_dict())


class SimpleEnv:
    """
    基于线性公式 y = ax + b 的环境
    
    环境设计：
    - 真实公式：y_true = a_true * x + b_true（固定参数，智能体不知道）
    - 预测公式：y_pred = a_est * x + b_est（智能体需要学习的参数）
    - 目标：通过调整 a_est 和 b_est，使 y_pred 尽可能接近 y_true
    - 奖励函数：reward = 1 / (1 + |y_true - y_pred|)，误差越小奖励越大
    """
    def __init__(self, a_true=2.0, b_true=1.0, x_range=(-5, 5)):
        """
        初始化环境
        Args:
            a_true: 真实斜率参数 a，默认2.0
                   真实公式：y_true = a_true * x + b_true
            b_true: 真实截距参数 b，默认1.0
            x_range: x的取值范围，默认(-5, 5)
        """
        # 存储真实参数（智能体不知道这些值）
        self.a_true = a_true  # 真实斜率 a_true = 2.0
        self.b_true = b_true  # 真实截距 b_true = 1.0
        self.x_range = x_range  # x的取值范围
        
        # 状态定义：[x, a_est, b_est]
        # x: 当前输入值
        # a_est: 当前估计的斜率参数
        # b_est: 当前估计的截距参数
        self.state_dim = 3  # 状态维度为3
        
        # 动作定义：4个离散动作
        # 0: 增加 a_est（a_est = a_est + step_size）
        # 1: 减少 a_est（a_est = a_est - step_size）
        # 2: 增加 b_est（b_est = b_est + step_size）
        # 3: 减少 b_est（b_est = b_est - step_size）
        self.action_dim = 4
        
        self.step_count = 0  # 当前步数计数器
        self.max_steps = 10  # 每个episode的最大步数
        
        # 初始化估计参数（智能体从0开始学习）
        self.a_est = 0.0  # 初始估计斜率
        self.b_est = 0.0  # 初始估计截距
        self.x = 0.0  # 初始x值
    
    def reset(self):
        """
        重置环境：开始新的episode
        Returns:
            state: 初始状态 [x, a_est, b_est]
        """
        # 随机初始化 x：在x_range范围内均匀采样
        # 公式：x ~ Uniform(x_range[0], x_range[1])
        self.x = np.random.uniform(self.x_range[0], self.x_range[1])
        
        # 重置估计参数为0（智能体重新开始学习）
        self.a_est = 0.0
        self.b_est = 0.0
        
        self.step_count = 0  # 重置步数计数器
        
        # 构建初始状态：[x, a_est, b_est]
        self.state = np.array([self.x, self.a_est, self.b_est], dtype=np.float32)
        
        return self.state.copy()  # 返回状态的副本
    
    def step(self, action):
        """
        执行动作：根据动作更新参数，计算奖励
        Args:
            action: 动作（0/1/2/3）
        Returns:
            next_state: 下一个状态 [x_new, a_est_new, b_est_new]
            reward: 奖励值
            done: 是否结束
            info: 额外信息（空字典）
        """
        self.step_count += 1  # 增加步数
        
        # 根据动作更新参数
        step_size = 0.1  # 参数调整步长（学习率）
        if action == 0:  # 动作0：增加 a_est
            # 公式：a_est = a_est + step_size
            self.a_est += step_size
        elif action == 1:  # 动作1：减少 a_est
            # 公式：a_est = a_est - step_size
            self.a_est -= step_size
        elif action == 2:  # 动作2：增加 b_est
            # 公式：b_est = b_est + step_size
            self.b_est += step_size
        elif action == 3:  # 动作3：减少 b_est
            # 公式：b_est = b_est - step_size
            self.b_est -= step_size
        
        # 计算真实值：使用真实参数和当前x值
        # 公式：y_true = a_true * x + b_true
        # 例如：y_true = 2.0 * x + 1.0
        y_true = self.a_true * self.x + self.b_true
        
        # 计算预测值：使用估计参数和当前x值
        # 公式：y_pred = a_est * x + b_est
        # 智能体的目标是使 y_pred 接近 y_true
        y_pred = self.a_est * self.x + self.b_est
        
        # 计算预测误差：真实值与预测值的绝对差
        # 公式：error = |y_true - y_pred|
        error = abs(y_true - y_pred)
        
        # 奖励函数：误差越小，奖励越大
        # 公式：reward = 1 / (1 + error)
        # 当 error = 0 时，reward = 1（最大奖励）
        # 当 error → ∞ 时，reward → 0（最小奖励）
        # 这是一个单调递减函数，鼓励智能体减小误差
        reward = 1.0 / (1.0 + error)
        
        # 更新 x：每次步进一个小的增量
        # 公式：x_new = x + 0.1
        self.x += 0.1
        
        # 如果x超出范围，重置到起始值（循环）
        if self.x > self.x_range[1]:
            self.x = self.x_range[0]
        
        # 更新状态：[x_new, a_est_new, b_est_new]
        self.state = np.array([self.x, self.a_est, self.b_est], dtype=np.float32)
        
        # 判断是否结束：达到最大步数
        done = self.step_count >= self.max_steps
        
        return self.state.copy(), reward, done, {}  # 返回下一个状态、奖励、结束标志和额外信息


def train_ppo(env, agent, num_episodes=1000, max_steps=200):
    """
    训练PPO算法
    
    训练流程：
    1. 对每个episode：
       a. 重置环境
       b. 收集一个episode的数据（状态、动作、奖励等）
       c. 使用收集的数据更新策略和价值网络
    2. 记录每个episode的累积奖励
    3. 定期打印训练进度
    
    Args:
        env: 环境对象
        agent: PPO智能体
        num_episodes: 训练episode数量，默认1000
        max_steps: 每个episode的最大步数，默认200
    Returns:
        episode_rewards: 每个episode的累积奖励列表
    """
    episode_rewards = []  # 存储每个episode的累积奖励
    
    # 训练循环：遍历每个episode
    for episode in range(num_episodes):
        # 初始化数据收集列表
        states = []  # 状态序列
        actions = []  # 动作序列
        old_log_probs = []  # 旧策略对数概率序列
        rewards = []  # 奖励序列
        next_states = []  # 下一个状态序列
        dones = []  # 终止标志序列
        
        # 重置环境，获取初始状态
        state = env.reset()
        episode_reward = 0  # 当前episode的累积奖励
        
        # 收集一个episode的数据
        for step in range(max_steps):
            # 选择动作：根据当前策略采样动作
            # 公式：a_t ~ π_θ(·|s_t)，log_prob = log π_θ(a_t|s_t)
            action, log_prob = agent.select_action(state)
            
            # 执行动作：在环境中执行动作，获得下一个状态和奖励
            # 公式：s_{t+1}, r_t, done = env.step(a_t)
            next_state, reward, done, _ = env.step(action)
            
            # 存储数据：保存当前步骤的所有信息
            states.append(state)  # 当前状态
            actions.append(action)  # 执行的动作
            old_log_probs.append(log_prob)  # 动作的对数概率
            rewards.append(reward)  # 获得的奖励
            next_states.append(next_state)  # 下一个状态
            dones.append(done)  # 是否结束
            
            # 更新状态和累积奖励
            state = next_state  # 状态转移：s_t = s_{t+1}
            episode_reward += reward  # 累积奖励：R = Σ r_t
            
            # 如果episode结束，提前退出
            if done:
                break
        
        # 更新策略：使用收集的数据更新PPO网络
        # 公式：θ, φ = PPO.update(states, actions, old_log_probs, rewards, ...)
        agent.update(states, actions, old_log_probs, rewards, next_states, dones)
        
        # 记录当前episode的累积奖励
        episode_rewards.append(episode_reward)
        
        # 打印训练进度：每100个episode打印一次
        if (episode + 1) % 100 == 0:
            # 计算最近100个episode的平均奖励
            avg_reward = np.mean(episode_rewards[-100:])
            # 打印进度信息
            print("Episode {}, Average Reward: {:.2f}".format(episode + 1, avg_reward))
    
    return episode_rewards  # 返回所有episode的奖励列表


# 主函数：程序入口
if __name__ == "__main__":
    # 创建环境：基于 y = 2x + 1 的线性环境
    # 真实公式：y_true = 2.0 * x + 1.0
    env = SimpleEnv()
    
    # 创建PPO智能体
    agent = PPO(
        state_dim=env.state_dim,  # 状态维度：3（x, a_est, b_est）
        action_dim=env.action_dim,  # 动作维度：4（增加/减少a/b）
        lr=3e-4,  # 学习率：0.0003
        gamma=0.99,  # 折扣因子：0.99
        epsilon=0.2,  # PPO裁剪参数：0.2
        c1=1.0,  # 价值函数损失系数：1.0
        c2=0.01,  # 熵奖励系数：0.01
        k_epochs=10,  # 优化周期数：10
        batch_size=64  # 小批量大小：64
    )
    
    # 开始训练
    print("开始训练PPO算法...")
    # 训练1000个episode，每个episode最多200步
    rewards = train_ppo(env, agent, num_episodes=10000, max_steps=10)
    print("训练完成！")
    
    # 绘制训练曲线（可选）
    try:
        import matplotlib.pyplot as plt  # 导入matplotlib绘图库
        plt.plot(rewards)  # 绘制奖励曲线：x轴是episode，y轴是累积奖励
        plt.xlabel('Episode')  # x轴标签：Episode
        plt.ylabel('Reward')  # y轴标签：Reward
        plt.title('PPO Training Curve')  # 图表标题：PPO训练曲线
        plt.show()  # 显示图表
    except ImportError:
        # 如果matplotlib未安装，跳过绘图
        print("matplotlib未安装，跳过绘图")
