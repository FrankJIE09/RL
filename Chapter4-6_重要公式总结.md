# 强化学习第4-6章重要公式总结

## 第4章：动态规划

### 4.4 策略评估

#### 迭代策略评估（状态价值函数）
$$v_\pi^{k+1}(s) = \sum_{a} \pi(a | s) \sum_{s', r} p(s', r | s, a) [r + \gamma v_\pi^k(s')]$$

#### 迭代策略评估（动作价值函数）
$$q_\pi^{k+1}(s, a) = \sum_{s', r} p(s', r | s, a) \left[r + \gamma \sum_{a'} \pi(a' | s') q_\pi^k(s', a')\right]$$

### 4.5 策略改进

#### 策略改进定理
如果 $q_\pi(s, \pi'(s)) \geq v_\pi(s)$ 对所有 $s$ 成立，则 $v_{\pi'}(s) \geq v_\pi(s)$ 对所有 $s$ 成立。

#### 策略改进（基于状态价值函数）
$$\pi'(s) = \arg\max_{a} \sum_{s', r} p(s', r | s, a) [r + \gamma v_\pi(s')]$$

#### 策略改进（基于动作价值函数）
$$\pi'(s) = \arg\max_{a} q_\pi(s, a)$$

### 4.6 策略迭代

#### 策略迭代算法
1. **策略评估**：计算 $v_\pi$ 或 $q_\pi$
2. **策略改进**：基于价值函数改进策略
3. **重复**：直到策略不再改变

### 4.7 价值迭代

#### 价值迭代更新
$$v_{k+1}(s) = \max_{a \in \mathcal{A}(s)} \sum_{s', r} p(s', r | s, a) [r + \gamma v_k(s')]$$

#### 价值迭代（动作价值函数形式）
$$q_{k+1}(s, a) = \sum_{s', r} p(s', r | s, a) \left[r + \gamma \max_{a'} q_k(s', a')\right]$$

### 4.8 期望更新

#### 期望更新（策略评估）
$$v_\pi^{k+1}(s) = \sum_{a} \pi(a | s) \sum_{s', r} p(s', r | s, a) [r + \gamma v_\pi^k(s')]$$

#### 期望更新（价值迭代）
$$v_{k+1}(s) = \max_{a} \sum_{s', r} p(s', r | s, a) [r + \gamma v_k(s')]$$

---

## 第5章：蒙特卡洛方法

### 5.1 回报定义

#### 回报（Return）
$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots + \gamma^{T-t-1} R_T$$

#### 回报的递归形式
$$G_t = R_{t+1} + \gamma G_{t+1}$$

### 5.2 蒙特卡洛预测

#### 状态价值函数的估计
$$v_\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$$

#### 显式平均（批量方法）
$$V(s) = \frac{1}{n} \sum_{i=1}^{n} G_i$$

其中 $n$ 是访问状态 $s$ 的次数，$G_i$ 是第 $i$ 次访问后的回报。

#### 增量式平均（在线方法）
$$V(S_t) \gets V(S_t) + \alpha [G_t - V(S_t)]$$

- 如果 $\alpha = \frac{1}{n}$（$n$ 是访问次数），这是**算术平均**
- 如果 $\alpha$ 是固定值，这是**指数移动平均**

### 5.3 蒙特卡洛动作价值估计

#### 动作价值函数的估计
$$q_\pi(s, a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]$$

#### 增量式更新
$$Q(S_t, A_t) \gets Q(S_t, A_t) + \alpha [G_t - Q(S_t, A_t)]$$

### 5.4 首次访问 vs 每次访问

#### 首次访问蒙特卡洛
只计算**首次访问**状态 $s$ 后的回报。

#### 每次访问蒙特卡洛
计算**每次访问**状态 $s$ 后的回报。

### 5.5 蒙特卡洛控制

#### $\varepsilon$-贪婪策略
$$\pi(a | s) = \begin{cases}
1 - \varepsilon + \frac{\varepsilon}{|\mathcal{A}(s)|}, & \text{如果 } a = \arg\max_{a'} Q(s, a') \\
\frac{\varepsilon}{|\mathcal{A}(s)|}, & \text{否则}
\end{cases}$$

#### 策略改进（基于动作价值函数）
$$\pi(s) \gets \arg\max_{a} Q(s, a)$$

### 5.6 重要性采样（Off-policy）

#### 重要性采样比率
$$\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k | S_k)}{b(A_k | S_k)}$$

其中：
- $\pi$ 是目标策略（要评估的策略）
- $b$ 是行为策略（生成数据的策略）

#### 加权回报（普通重要性采样）
$$V(s) \gets V(s) + \alpha [\rho_{t:T-1} G_t - V(s)]$$

#### 加权回报（加权重要性采样）
$$V(s) \gets V(s) + \frac{W}{C(s)} [G_t - V(s)]$$

其中：
- $W$ 是重要性采样权重
- $C(s)$ 是累积权重

#### 加权重要性采样更新
$$Q(S_t, A_t) \gets Q(S_t, A_t) + \frac{W}{C(S_t, A_t)} [G_t - Q(S_t, A_t)]$$

其中：
$$C(S_t, A_t) \gets C(S_t, A_t) + W$$

$$W \gets W \frac{\pi(A_t | S_t)}{b(A_t | S_t)}$$

---

## 第6章：时序差分学习

### 6.1 TD(0)预测

#### TD(0)更新公式
$$V(S_t) \gets V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$

#### TD误差
$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

#### TD目标
$$\text{目标} = R_{t+1} + \gamma V(S_{t+1})$$

### 6.2 TD(0) vs 蒙特卡洛

#### 蒙特卡洛更新
$$V(S_t) \gets V(S_t) + \alpha [G_t - V(S_t)]$$

其中 $G_t$ 是完整回报。

#### TD(0)更新
$$V(S_t) \gets V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$

**关键区别**：
- **蒙特卡洛**：使用完整回报 $G_t$（需要等待回合结束）
- **TD(0)**：使用一步前瞻 $R_{t+1} + \gamma V(S_{t+1})$（可以立即更新）

### 6.3 TD误差与蒙特卡洛误差的关系

如果价值函数在回合中不改变，蒙特卡洛误差可以写成TD误差的折扣和：

$$G_t - V(S_t) = \delta_t + \gamma \delta_{t+1} + \gamma^2 \delta_{t+2} + \cdots$$

其中：
$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

### 6.4 SARSA（On-policy TD控制）

#### SARSA更新公式
$$Q(S_t, A_t) \gets Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$$

#### SARSA的关键特征
- **On-policy**：使用当前策略选择动作
- **五元组**：$(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$
- **自举法**：使用 $Q(S_{t+1}, A_{t+1})$ 更新 $Q(S_t, A_t)$

#### SARSA算法流程
1. 初始化 $Q(s, a)$ 和策略 $\pi$（$\varepsilon$-贪婪）
2. 对每个时间步：
   - 执行动作 $A_t$，观察 $R_{t+1}, S_{t+1}$
   - 选择下一动作 $A_{t+1}$（根据策略 $\pi$）
   - 更新：$Q(S_t, A_t) \gets Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$
   - 策略改进：$\pi(S_t) \gets \varepsilon$-贪婪策略，基于 $Q(S_t, \cdot)$

### 6.5 Q-learning（Off-policy TD控制）

#### Q-learning更新公式
$$Q(S_t, A_t) \gets Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t)]$$

#### Q-learning的关键特征
- **Off-policy**：学习最优动作价值函数，但可以使用任何策略收集数据
- **最大化操作**：使用 $\max_{a} Q(S_{t+1}, a)$ 而不是 $Q(S_{t+1}, A_{t+1})$
- **直接学习最优策略**：不需要显式策略改进步骤

#### Q-learning算法流程
1. 初始化 $Q(s, a)$
2. 对每个时间步：
   - 执行动作 $A_t$（行为策略选择），观察 $R_{t+1}, S_{t+1}$
   - 更新：$Q(S_t, A_t) \gets Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t)]$
3. 提取最优策略：$\pi_*(s) = \arg\max_{a} Q(s, a)$

### 6.6 SARSA vs Q-learning

#### 更新公式对比

**SARSA（On-policy）**：
$$Q(S_t, A_t) \gets Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$$

**Q-learning（Off-policy）**：
$$Q(S_t, A_t) \gets Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t)]$$

#### 关键区别

| 特征 | SARSA | Q-learning |
|------|-------|------------|
| 更新目标 | $Q(S_{t+1}, A_{t+1})$ | $\max_{a} Q(S_{t+1}, a)$ |
| 依赖实际动作 | 是 | 否 |
| 学习目标 | 当前策略的价值函数 | 最优策略的价值函数 |
| 策略类型 | On-policy | Off-policy |
| 探索考虑 | 考虑探索风险 | 不考虑探索风险 |

### 6.7 Expected SARSA

#### Expected SARSA更新公式
$$Q(S_t, A_t) \gets Q(S_t, A_t) + \alpha \left[R_{t+1} + \gamma \sum_{a} \pi(a | S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t)\right]$$

#### Expected SARSA的特点
- 使用期望值而不是采样值
- 可以用于 On-policy 或 Off-policy
- 当目标策略是贪婪策略时，Expected SARSA 就是 Q-learning

---

## 公式关系总结

### 价值函数关系链

1. **状态价值函数**：
   $$v_\pi(s) = \sum_{a} \pi(a | s) q_\pi(s, a)$$

2. **动作价值函数**：
   $$q_\pi(s, a) = \sum_{s', r} p(s', r | s, a) [r + \gamma v_\pi(s')]$$

3. **贝尔曼方程**（状态价值）：
   $$v_\pi(s) = \sum_{a} \pi(a | s) \sum_{s', r} p(s', r | s, a) [r + \gamma v_\pi(s')]$$

4. **贝尔曼方程**（动作价值）：
   $$q_\pi(s, a) = \sum_{s', r} p(s', r | s, a) \left[r + \gamma \sum_{a'} \pi(a' | s') q_\pi(s', a')\right]$$

### 方法对比

| 方法 | 更新公式 | 需要模型 | 自举法 | 采样 |
|------|---------|---------|--------|------|
| **动态规划** | 期望更新 | 是 | 是 | 否 |
| **蒙特卡洛** | $V(S_t) \gets V(S_t) + \alpha [G_t - V(S_t)]$ | 否 | 否 | 是 |
| **TD(0)** | $V(S_t) \gets V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$ | 否 | 是 | 是 |
| **SARSA** | $Q(S_t, A_t) \gets Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$ | 否 | 是 | 是 |
| **Q-learning** | $Q(S_t, A_t) \gets Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t)]$ | 否 | 是 | 是 |

### 关键概念

1. **自举法（Bootstrapping）**：使用估计值来更新估计值
   - 动态规划：使用 $v_k(s')$ 更新 $v_{k+1}(s)$
   - TD方法：使用 $V(S_{t+1})$ 更新 $V(S_t)$

2. **采样（Sampling）**：使用样本而不是期望值
   - 蒙特卡洛：使用样本回报 $G_t$
   - TD方法：使用样本转移 $(S_t, R_{t+1}, S_{t+1})$

3. **On-policy vs Off-policy**：
   - **On-policy**：评估和改进的是同一个策略（如 SARSA）
   - **Off-policy**：评估一个策略，但使用另一个策略的数据（如 Q-learning）

---

## 符号说明

- $S_t$：时刻 $t$ 的状态
- $A_t$：时刻 $t$ 的动作
- $R_{t+1}$：时刻 $t+1$ 的奖励
- $G_t$：从时刻 $t$ 开始的回报
- $\gamma$：折扣因子，$\gamma \in [0, 1]$
- $\alpha$：步长参数（学习率），$\alpha \in (0, 1]$
- $\pi$：策略
- $v_\pi(s)$：策略 $\pi$ 下状态 $s$ 的价值函数
- $q_\pi(s, a)$：策略 $\pi$ 下状态-动作对 $(s, a)$ 的价值函数
- $v_*(s)$：最优状态价值函数
- $q_*(s, a)$：最优动作价值函数
- $\pi_*$：最优策略
- $p(s', r | s, a)$：环境动态函数（转移概率和奖励）
- $\varepsilon$：探索参数（$\varepsilon$-贪婪策略）
- $\delta_t$：TD误差
- $\rho_{t:T-1}$：重要性采样比率

