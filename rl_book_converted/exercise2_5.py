#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 2.5: 非平稳问题的样本平均方法困难演示

设计并实施一个实验，展示样本平均方法在非平稳问题中的困难。
使用修改版的10-armed testbed，其中所有q*(a)开始时相等，然后进行独立随机游走。
"""

import numpy as np
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
except ImportError:
    # 如果没有tqdm，使用简单的进度显示
    def tqdm(iterable, **kwargs):
        return iterable


class NonstationaryBandit:
    """
    非平稳的10-armed bandit环境
    
    特点：
    - 所有q*(a)开始时相等（设为0）
    - 每一步，所有q*(a)都加上一个正态分布的增量（均值0，标准差0.01）
    """
    
    def __init__(self, k=10, initial_value=0.0, random_walk_std=0.01):
        """
        初始化非平稳bandit
        
        Args:
            k: 动作数量（默认10）
            initial_value: 初始q*值（默认0.0）
            random_walk_std: 随机游走的标准差（默认0.01）
        """
        self.k = k
        self.q_star = np.full(k, initial_value)  # 所有动作的初始真实值相等
        self.random_walk_std = random_walk_std
        
    def step(self, action):
        """
        执行动作并返回奖励
        
        Args:
            action: 选择的动作索引
            
        Returns:
            reward: 从N(q*(action), 1)分布中采样的奖励
        """
        # 奖励从N(q*(action), 1)分布中采样
        reward = np.random.normal(self.q_star[action], 1.0)
        return reward
    
    def update_q_star(self):
        """
        更新所有动作的真实值（随机游走）
        每一步都加上一个正态分布的增量
        """
        increments = np.random.normal(0.0, self.random_walk_std, self.k)
        self.q_star += increments
    
    def get_optimal_action(self):
        """返回当前最优动作的索引"""
        return np.argmax(self.q_star)


class SampleAverageAgent:
    """
    使用样本平均方法估计动作值的智能体
    公式：Q_n = (R_1 + R_2 + ... + R_{n-1}) / (n-1)
    """
    
    def __init__(self, k=10, epsilon=0.1):
        """
        初始化智能体
        
        Args:
            k: 动作数量
            epsilon: ε-greedy参数
        """
        self.k = k
        self.epsilon = epsilon
        self.Q = np.zeros(k)  # 动作值估计
        self.N = np.zeros(k)  # 每个动作被选择的次数
        
    def select_action(self):
        """
        使用ε-greedy策略选择动作
        
        Returns:
            action: 选择的动作索引
        """
        if np.random.random() < self.epsilon:
            # 探索：随机选择
            return np.random.randint(self.k)
        else:
            # 利用：选择估计值最大的动作
            return np.argmax(self.Q)
    
    def update(self, action, reward):
        """
        使用样本平均方法更新动作值估计
        增量更新：Q(A) = Q(A) + (R - Q(A)) / N(A)
        
        Args:
            action: 选择的动作
            reward: 获得的奖励
        """
        self.N[action] += 1
        # 增量更新公式：Q_n = Q_{n-1} + (R_n - Q_{n-1}) / n
        self.Q[action] += (reward - self.Q[action]) / self.N[action]


class ConstantStepSizeAgent:
    """
    使用常数步长参数估计动作值的智能体
    公式：Q_n = Q_{n-1} + α(R_n - Q_{n-1})
    """
    
    def __init__(self, k=10, epsilon=0.1, alpha=0.1):
        """
        初始化智能体
        
        Args:
            k: 动作数量
            epsilon: ε-greedy参数
            alpha: 常数步长参数
        """
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q = np.zeros(k)  # 动作值估计
        
    def select_action(self):
        """
        使用ε-greedy策略选择动作
        
        Returns:
            action: 选择的动作索引
        """
        if np.random.random() < self.epsilon:
            # 探索：随机选择
            return np.random.randint(self.k)
        else:
            # 利用：选择估计值最大的动作
            return np.argmax(self.Q)
    
    def update(self, action, reward):
        """
        使用常数步长参数更新动作值估计
        公式：Q(A) = Q(A) + α(R - Q(A))
        
        Args:
            action: 选择的动作
            reward: 获得的奖励
        """
        # 常数步长更新公式
        self.Q[action] += self.alpha * (reward - self.Q[action])


def run_experiment(agent_class, num_steps=10000, num_runs=2000, **agent_kwargs):
    """
    运行实验
    
    Args:
        agent_class: 智能体类（SampleAverageAgent 或 ConstantStepSizeAgent）
        num_steps: 每次运行的步数
        num_runs: 运行次数
        **agent_kwargs: 传递给智能体的参数
        
    Returns:
        rewards: 形状为(num_runs, num_steps)的奖励数组
        optimal_actions: 形状为(num_runs, num_steps)的最优动作选择数组
    """
    rewards = np.zeros((num_runs, num_steps))
    optimal_actions = np.zeros((num_runs, num_steps))
    
    print(f"运行 {num_runs} 次实验，每次 {num_steps} 步...")
    
    for run in tqdm(range(num_runs)):
        # 创建新的bandit环境
        bandit = NonstationaryBandit()
        # 创建智能体
        agent = agent_class(**agent_kwargs)
        
        for step in range(num_steps):
            # 选择动作
            action = agent.select_action()
            
            # 执行动作，获得奖励
            reward = bandit.step(action)
            
            # 更新智能体
            agent.update(action, reward)
            
            # 记录奖励
            rewards[run, step] = reward
            
            # 记录是否选择了最优动作
            optimal_action = bandit.get_optimal_action()
            optimal_actions[run, step] = (action == optimal_action)
            
            # 更新bandit的真实值（随机游走）
            bandit.update_q_star()
    
    return rewards, optimal_actions


def plot_results(sample_avg_rewards, sample_avg_optimal, 
                 const_step_rewards, const_step_optimal):
    """
    绘制结果图表（类似Figure 2.2）
    
    Args:
        sample_avg_rewards: 样本平均方法的奖励数组
        sample_avg_optimal: 样本平均方法的最优动作选择数组
        const_step_rewards: 常数步长方法的奖励数组
        const_step_optimal: 常数步长方法的最优动作选择数组
    """
    # 计算平均值
    sample_avg_rewards_mean = np.mean(sample_avg_rewards, axis=0)
    sample_avg_optimal_mean = np.mean(sample_avg_optimal, axis=0) * 100  # 转换为百分比
    
    const_step_rewards_mean = np.mean(const_step_rewards, axis=0)
    const_step_optimal_mean = np.mean(const_step_optimal, axis=0) * 100
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # 上图：平均奖励
    steps = np.arange(1, len(sample_avg_rewards_mean) + 1)
    ax1.plot(steps, sample_avg_rewards_mean, label='Sample Average', linewidth=1.5)
    ax1.plot(steps, const_step_rewards_mean, label='Constant Step-Size (α=0.1)', 
             linewidth=1.5, linestyle='--')
    ax1.set_xlabel('Steps', fontsize=12)
    ax1.set_ylabel('Average Reward', fontsize=12)
    ax1.set_title('Average Performance: Sample Average vs Constant Step-Size\n'
                  '(Nonstationary 10-armed Bandit)', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 下图：最优动作百分比
    ax2.plot(steps, sample_avg_optimal_mean, label='Sample Average', linewidth=1.5)
    ax2.plot(steps, const_step_optimal_mean, label='Constant Step-Size (α=0.1)', 
             linewidth=1.5, linestyle='--')
    ax2.set_xlabel('Steps', fontsize=12)
    ax2.set_ylabel('% Optimal Action', fontsize=12)
    ax2.set_title('Optimal Action Selection Percentage', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig('exercise2_5_results.png', dpi=300, bbox_inches='tight')
    print("\n图表已保存为: exercise2_5_results.png")
    plt.show()


def main():
    """主函数"""
    print("=" * 80)
    print("Exercise 2.5: 非平稳问题的样本平均方法困难演示")
    print("=" * 80)
    print("\n实验设置：")
    print("- 10-armed bandit（非平稳）")
    print("- q*(a)初始值相等，然后进行随机游走（标准差=0.01）")
    print("- ε-greedy策略（ε=0.1）")
    print("- 运行10,000步，重复2,000次")
    print("- 比较：样本平均方法 vs 常数步长方法（α=0.1）")
    print("=" * 80)
    
    # 实验参数
    # 注意：完整实验使用 num_steps=10000, num_runs=2000
    # 快速测试可以使用较小的值，例如：num_steps=1000, num_runs=10
    num_steps = 10000
    num_runs = 2000
    epsilon = 0.1
    alpha = 0.1
    
    # 快速测试模式（取消注释以启用）
    # num_steps = 1000
    # num_runs = 10
    
    # 运行样本平均方法
    print("\n[1/2] 运行样本平均方法...")
    sample_avg_rewards, sample_avg_optimal = run_experiment(
        SampleAverageAgent,
        num_steps=num_steps,
        num_runs=num_runs,
        k=10,
        epsilon=epsilon
    )
    
    # 运行常数步长方法
    print("\n[2/2] 运行常数步长方法...")
    const_step_rewards, const_step_optimal = run_experiment(
        ConstantStepSizeAgent,
        num_steps=num_steps,
        num_runs=num_runs,
        k=10,
        epsilon=epsilon,
        alpha=alpha
    )
    
    # 绘制结果
    print("\n绘制结果图表...")
    plot_results(
        sample_avg_rewards, sample_avg_optimal,
        const_step_rewards, const_step_optimal
    )
    
    # 打印最终统计
    print("\n" + "=" * 80)
    print("最终统计（最后1000步的平均值）：")
    print("=" * 80)
    final_steps = 1000
    
    sample_avg_final_reward = np.mean(sample_avg_rewards[:, -final_steps:])
    sample_avg_final_optimal = np.mean(sample_avg_optimal[:, -final_steps:]) * 100
    
    const_step_final_reward = np.mean(const_step_rewards[:, -final_steps:])
    const_step_final_optimal = np.mean(const_step_optimal[:, -final_steps:]) * 100
    
    print(f"\n样本平均方法：")
    print(f"  平均奖励: {sample_avg_final_reward:.4f}")
    print(f"  最优动作百分比: {sample_avg_final_optimal:.2f}%")
    
    print(f"\n常数步长方法（α={alpha}）：")
    print(f"  平均奖励: {const_step_final_reward:.4f}")
    print(f"  最优动作百分比: {const_step_final_optimal:.2f}%")
    
    print(f"\n性能提升：")
    print(f"  奖励提升: {const_step_final_reward - sample_avg_final_reward:.4f}")
    print(f"  最优动作百分比提升: {const_step_final_optimal - sample_avg_final_optimal:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()

