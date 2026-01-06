#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gridworld价值迭代算法演示
演示如何通过价值迭代计算最优价值函数表格
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class Gridworld:
    """5x5 Gridworld环境"""
    
    def __init__(self, gamma=0.9):
        self.size = 5
        self.gamma = gamma
        self.num_states = self.size * self.size
        
        # 特殊状态定义（状态编号从0开始，行优先）
        # 状态A: (0, 1) -> 状态1
        # 状态B: (0, 3) -> 状态3
        # 状态A': (1, 1) -> 状态6
        # 状态B': (1, 3) -> 状态8
        self.state_A = 1   # 第1行第2列
        self.state_B = 3   # 第1行第4列
        self.state_A_prime = 6   # 第2行第2列
        self.state_B_prime = 8   # 第2行第4列
        
        # 动作定义：0=北, 1=南, 2=东, 3=西
        self.actions = ['北', '南', '东', '西']
        self.num_actions = 4
        
    def get_coords(self, state):
        """从状态编号获取坐标 (row, col)"""
        row = state // self.size
        col = state % self.size
        return row, col
    
    def get_state(self, row, col):
        """从坐标获取状态编号"""
        return row * self.size + col
    
    def is_special_state(self, state):
        """判断是否是特殊状态"""
        return state == self.state_A or state == self.state_B
    
    def get_transition(self, state, action):
        """
        获取状态转移和奖励
        返回: (next_state, reward, probability)
        """
        # 特殊状态A
        if state == self.state_A:
            return self.state_A_prime, 10, 1.0
        
        # 特殊状态B
        if state == self.state_B:
            return self.state_B_prime, 5, 1.0
        
        # 普通状态
        row, col = self.get_coords(state)
        
        if action == 0:  # 北
            if row == 0:  # 撞墙
                return state, -1, 1.0
            else:
                next_state = self.get_state(row - 1, col)
                return next_state, 0, 1.0
        
        elif action == 1:  # 南
            if row == self.size - 1:  # 撞墙
                return state, -1, 1.0
            else:
                next_state = self.get_state(row + 1, col)
                return next_state, 0, 1.0
        
        elif action == 2:  # 东
            if col == self.size - 1:  # 撞墙
                return state, -1, 1.0
            else:
                next_state = self.get_state(row, col + 1)
                return next_state, 0, 1.0
        
        elif action == 3:  # 西
            if col == 0:  # 撞墙
                return state, -1, 1.0
            else:
                next_state = self.get_state(row, col - 1)
                return next_state, 0, 1.0


class ValueIteration:
    """价值迭代算法"""
    
    def __init__(self, env, epsilon=0.01):
        self.env = env
        self.epsilon = epsilon
        self.value_history = []  # 存储每次迭代的价值函数
        self.policy_history = []  # 存储每次迭代的策略
        
    def compute_q_value(self, state, action, value_func):
        """计算动作价值函数 q(s, a)"""
        next_state, reward, prob = self.env.get_transition(state, action)
        return prob * (reward + self.env.gamma * value_func[next_state])
    
    def value_iteration(self, max_iterations=100):
        """执行价值迭代"""
        # 初始化价值函数
        V = np.zeros(self.env.num_states)
        self.value_history.append(V.copy())
        
        for iteration in range(max_iterations):
            V_new = np.zeros(self.env.num_states)
            policy = np.zeros(self.env.num_states, dtype=int)
            
            # 对每个状态更新价值
            for state in range(self.env.num_states):
                q_values = []
                for action in range(self.env.num_actions):
                    q = self.compute_q_value(state, action, V)
                    q_values.append(q)
                
                # 选择最优动作和价值
                V_new[state] = max(q_values)
                policy[state] = np.argmax(q_values)
            
            # 保存历史
            self.value_history.append(V_new.copy())
            self.policy_history.append(policy.copy())
            
            # 检查收敛
            delta = np.max(np.abs(V_new - V))
            if delta < self.epsilon:
                print(f"收敛于第 {iteration + 1} 次迭代，最大变化: {delta:.6f}")
                break
            
            V = V_new
        
        return V, policy
    
    def get_value_table(self, value_func):
        """将价值函数转换为5x5表格"""
        table = np.zeros((self.env.size, self.env.size))
        for state in range(self.env.num_states):
            row, col = self.env.get_coords(state)
            table[row, col] = value_func[state]
        return table


def visualize_value_function(env, value_func, iteration, ax=None, save_path=None):
    """可视化价值函数"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # 创建价值表格
    value_table = np.zeros((env.size, env.size))
    for state in range(env.num_states):
        row, col = env.get_coords(state)
        value_table[row, col] = value_func[state]
    
    # 绘制热力图
    im = ax.imshow(value_table, cmap='YlOrRd', aspect='auto', vmin=0, vmax=25)
    
    # 添加数值标注
    for i in range(env.size):
        for j in range(env.size):
            state = env.get_state(i, j)
            value = value_func[state]
            text_color = 'white' if value > 12.5 else 'black'
            ax.text(j, i, f'{value:.1f}', ha='center', va='center', 
                   color=text_color, fontsize=12, fontweight='bold')
    
    # 标记特殊状态
    # 状态A
    rect_A = patches.Rectangle((env.state_A % env.size - 0.5, 
                                env.state_A // env.size - 0.5), 
                               1, 1, linewidth=3, edgecolor='blue', 
                               facecolor='none', linestyle='--')
    ax.add_patch(rect_A)
    ax.text(env.state_A % env.size, env.state_A // env.size - 0.3, 
           'A', ha='center', va='center', color='blue', 
           fontsize=14, fontweight='bold')
    
    # 状态B
    rect_B = patches.Rectangle((env.state_B % env.size - 0.5, 
                                env.state_B // env.size - 0.5), 
                               1, 1, linewidth=3, edgecolor='green', 
                               facecolor='none', linestyle='--')
    ax.add_patch(rect_B)
    ax.text(env.state_B % env.size, env.state_B // env.size - 0.3, 
           'B', ha='center', va='center', color='green', 
           fontsize=14, fontweight='bold')
    
    # 状态A'
    ax.text(env.state_A_prime % env.size, env.state_A_prime // env.size + 0.3, 
           "A'", ha='center', va='center', color='blue', 
           fontsize=10, fontweight='bold')
    
    # 状态B'
    ax.text(env.state_B_prime % env.size, env.state_B_prime // env.size + 0.3, 
           "B'", ha='center', va='center', color='green', 
           fontsize=10, fontweight='bold')
    
    ax.set_title(f'价值迭代 - 第 {iteration} 次迭代', fontsize=16, fontweight='bold')
    ax.set_xlabel('列', fontsize=12)
    ax.set_ylabel('行', fontsize=12)
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.set_xticklabels(range(1, env.size + 1))
    ax.set_yticklabels(range(1, env.size + 1))
    
    # 添加颜色条
    plt.colorbar(im, ax=ax, label='状态价值', fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return ax


def visualize_policy(env, policy, iteration, ax=None, save_path=None):
    """可视化策略（用箭头表示）"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # 动作到箭头的映射
    arrows = {
        0: '↑',  # 北
        1: '↓',  # 南
        2: '→',  # 东
        3: '←'   # 西
    }
    
    # 绘制网格
    for i in range(env.size + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5)
        ax.axvline(i - 0.5, color='gray', linewidth=0.5)
    
    # 绘制策略箭头
    for state in range(env.num_states):
        row, col = env.get_coords(state)
        action = policy[state]
        arrow = arrows[action]
        ax.text(col, row, arrow, ha='center', va='center', 
               fontsize=20, fontweight='bold')
    
    # 标记特殊状态
    # 状态A
    rect_A = patches.Rectangle((env.state_A % env.size - 0.5, 
                                env.state_A // env.size - 0.5), 
                               1, 1, linewidth=3, edgecolor='blue', 
                               facecolor='lightblue', alpha=0.3, linestyle='--')
    ax.add_patch(rect_A)
    ax.text(env.state_A % env.size, env.state_A // env.size - 0.3, 
           'A', ha='center', va='center', color='blue', 
           fontsize=14, fontweight='bold')
    
    # 状态B
    rect_B = patches.Rectangle((env.state_B % env.size - 0.5, 
                                env.state_B // env.size - 0.5), 
                               1, 1, linewidth=3, edgecolor='green', 
                               facecolor='lightgreen', alpha=0.3, linestyle='--')
    ax.add_patch(rect_B)
    ax.text(env.state_B % env.size, env.state_B // env.size - 0.3, 
           'B', ha='center', va='center', color='green', 
           fontsize=14, fontweight='bold')
    
    ax.set_title(f'最优策略 - 第 {iteration} 次迭代', fontsize=16, fontweight='bold')
    ax.set_xlabel('列', fontsize=12)
    ax.set_ylabel('行', fontsize=12)
    ax.set_xlim(-0.5, env.size - 0.5)
    ax.set_ylim(env.size - 0.5, -0.5)
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.set_xticklabels(range(1, env.size + 1))
    ax.set_yticklabels(range(1, env.size + 1))
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return ax


def create_animation(env, vi, output_file='value_iteration_animation.gif'):
    """创建价值迭代过程的动画"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    def animate(frame):
        for ax in axes:
            ax.clear()
        
        if frame < len(vi.value_history):
            value_func = vi.value_history[frame]
            policy = vi.policy_history[frame] if frame < len(vi.policy_history) else np.zeros(env.num_states)
            
            # 左图：价值函数
            visualize_value_function(env, value_func, frame, ax=axes[0])
            
            # 右图：策略
            visualize_policy(env, policy, frame, ax=axes[1])
    
    anim = FuncAnimation(fig, animate, frames=len(vi.value_history), 
                        interval=1000, repeat=True)
    anim.save(output_file, writer='pillow', fps=1)
    print(f"动画已保存到: {output_file}")
    return anim


def print_value_table(value_func, env):
    """打印价值函数表格"""
    print("\n最优价值函数表格:")
    print("=" * 60)
    for i in range(env.size):
        row_values = []
        for j in range(env.size):
            state = env.get_state(i, j)
            row_values.append(f"{value_func[state]:.1f}")
        print(" | ".join(f"{v:>6}" for v in row_values))
    print("=" * 60)


def main():
    """主函数"""
    print("=" * 60)
    print("Gridworld价值迭代算法演示")
    print("=" * 60)
    
    # 创建环境
    env = Gridworld(gamma=0.9)
    print(f"\n环境设置:")
    print(f"  - 网格大小: {env.size}x{env.size}")
    print(f"  - 折扣因子: {env.gamma}")
    print(f"  - 状态A: 第1行第2列 (状态 {env.state_A})")
    print(f"  - 状态B: 第1行第4列 (状态 {env.state_B})")
    print(f"  - 状态A': 第2行第2列 (状态 {env.state_A_prime})")
    print(f"  - 状态B': 第2行第4列 (状态 {env.state_B_prime})")
    
    # 创建价值迭代算法
    vi = ValueIteration(env, epsilon=0.01)
    
    # 执行价值迭代
    print("\n开始价值迭代...")
    optimal_value, optimal_policy = vi.value_iteration(max_iterations=100)
    
    # 打印结果
    print_value_table(optimal_value, env)
    
    # 可视化前几次迭代
    print("\n生成可视化...")
    num_visualizations = min(6, len(vi.value_history))
    
    fig, axes = plt.subplots(2, num_visualizations, figsize=(5*num_visualizations, 10))
    if num_visualizations == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(num_visualizations):
        # 价值函数
        visualize_value_function(env, vi.value_history[i], i, ax=axes[0, i])
        
        # 策略（只在最后几次迭代显示）
        if i >= num_visualizations - 3:
            visualize_policy(env, vi.policy_history[i], i, ax=axes[1, i])
        else:
            axes[1, i].axis('off')
    
    plt.savefig('value_iteration_process.png', dpi=150, bbox_inches='tight')
    print("可视化已保存到: value_iteration_process.png")
    
    # 创建最终结果的详细可视化
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    visualize_value_function(env, optimal_value, len(vi.value_history) - 1, ax=axes[0])
    visualize_policy(env, optimal_policy, len(vi.value_history) - 1, ax=axes[1])
    plt.savefig('final_result.png', dpi=150, bbox_inches='tight')
    print("最终结果已保存到: final_result.png")
    
    # 绘制收敛曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    max_deltas = []
    for i in range(1, len(vi.value_history)):
        delta = np.max(np.abs(vi.value_history[i] - vi.value_history[i-1]))
        max_deltas.append(delta)
    
    ax.plot(range(1, len(max_deltas) + 1), max_deltas, 'b-o', linewidth=2, markersize=6)
    ax.axhline(y=vi.epsilon, color='r', linestyle='--', label=f'收敛阈值 ({vi.epsilon})')
    ax.set_xlabel('迭代次数', fontsize=12)
    ax.set_ylabel('最大价值变化', fontsize=12)
    ax.set_title('价值迭代收敛曲线', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig('convergence_curve.png', dpi=150, bbox_inches='tight')
    print("收敛曲线已保存到: convergence_curve.png")
    
    # 显示一些关键状态的收敛过程
    key_states = [env.state_A, env.state_A_prime, env.state_B, env.state_B_prime, 12]  # 中心状态
    state_names = ['A', "A'", 'B', "B'", '中心']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    for state, name in zip(key_states, state_names):
        values = [v[state] for v in vi.value_history]
        ax.plot(values, 'o-', linewidth=2, markersize=6, label=f'状态{name} (s={state})')
    
    ax.set_xlabel('迭代次数', fontsize=12)
    ax.set_ylabel('状态价值', fontsize=12)
    ax.set_title('关键状态的价值收敛过程', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig('key_states_convergence.png', dpi=150, bbox_inches='tight')
    print("关键状态收敛过程已保存到: key_states_convergence.png")
    
    print("\n所有可视化已完成！")
    print("\n生成的文件:")
    print("  - value_iteration_process.png: 迭代过程可视化")
    print("  - final_result.png: 最终价值函数和策略")
    print("  - convergence_curve.png: 收敛曲线")
    print("  - key_states_convergence.png: 关键状态收敛过程")
    
    # 显示图表
    plt.show()


if __name__ == '__main__':
    main()

