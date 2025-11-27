#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于指数积的串联机器人标定方法
参考论文：高文斌等，"一种基于指数积的串联机器人标定方法"，机器人，2013

核心公式：
1. POE运动学模型：f(θ) = e^(ξ₁θ₁) e^(ξ₂θ₂) ... e^(ξₙθₙ) e^Γ
2. 线性化模型：(δf·f^(-1))^∨ = Σ[Ji·δηi] + J_{n+1}·δΓ
3. 最小二乘解：x = (L^T L)^(-1) L^T Y
4. 参数更新：ξ_{i,k} = Ad(e^(δη_{i,k-1})) ξ_{i,k-1}
"""

import numpy as np
import matplotlib.pyplot as plt
from spatialmath import SE3
import warnings
warnings.filterwarnings('ignore')

# 导入中文字体配置
try:
    from font_config import init_chinese_font
    init_chinese_font(verbose=False)
except ImportError:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False


class POECalibrationPaper:
    """基于论文方法的POE校准类"""
    
    def __init__(self, true_screw_axes, initial_screw_axes, Gamma_true, Gamma_initial):
        """
        初始化
        
        Args:
            true_screw_axes: 真实关节旋量列表 [ξ₁, ξ₂, ..., ξₙ]，每个ξ是6维向量
            initial_screw_axes: 初始估计关节旋量列表（带误差）
            Gamma_true: 真实零位旋量（6维向量）
            Gamma_initial: 初始估计零位旋量（带误差）
        """
        self.true_screw_axes = true_screw_axes
        self.initial_screw_axes = initial_screw_axes
        self.Gamma_true = Gamma_true
        self.Gamma_initial = Gamma_initial
        
        # 参数向量：所有旋量参数
        self.phi_true = self._params_to_vector(true_screw_axes, Gamma_true)
        self.phi_est = self._params_to_vector(initial_screw_axes, Gamma_initial)
    
    def _params_to_vector(self, screw_axes, Gamma):
        """将参数转换为向量"""
        phi = []
        for xi in screw_axes:
            phi.extend(list(xi))
        phi.extend(list(Gamma))
        return np.array(phi)
    
    def _vector_to_params(self, phi):
        """将向量转换为参数"""
        n_joints = (len(phi) - 6) // 6
        screw_axes = []
        for i in range(n_joints):
            xi = phi[i*6:(i+1)*6]
            screw_axes.append(xi)
        Gamma = phi[-6:]
        return screw_axes, Gamma
    
    def hat_operator(self, xi):
        """
        旋量坐标的hat算子（se(3)的李代数元素）
        
        ξ = [ω, v]^T → ξ^ = [[ω^, v], [0, 0]]
        """
        omega = xi[:3]
        v = xi[3:]
        
        omega_hat = np.array([
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0]
        ])
        
        xi_hat = np.block([
            [omega_hat, v.reshape(3, 1)],
            [np.zeros((1, 4))]
        ])
        return xi_hat
    
    def exp_se3(self, xi_hat):
        """
        计算SE(3)的指数映射：e^(ξ^)
        
        使用Rodrigues公式
        """
        omega_hat = xi_hat[:3, :3]
        v = xi_hat[:3, 3]
        
        # 从反对称矩阵提取ω
        omega = np.array([
            omega_hat[2, 1] - omega_hat[1, 2],
            omega_hat[0, 2] - omega_hat[2, 0],
            omega_hat[1, 0] - omega_hat[0, 1]
        ]) / 2
        
        omega_norm = np.linalg.norm(omega)
        
        if omega_norm < 1e-6:
            # 纯平移
            R = np.eye(3)
            t = v
        else:
            # Rodrigues公式
            omega_unit = omega / omega_norm
            omega_unit_hat = np.array([
                [0, -omega_unit[2], omega_unit[1]],
                [omega_unit[2], 0, -omega_unit[0]],
                [-omega_unit[1], omega_unit[0], 0]
            ])
            
            R = np.eye(3) + np.sin(omega_norm) * omega_unit_hat + \
                (1 - np.cos(omega_norm)) * omega_unit_hat @ omega_unit_hat
            
            # 计算平移（使用V矩阵）
            V = np.eye(3) + (1 - np.cos(omega_norm)) / omega_norm * omega_unit_hat + \
                (omega_norm - np.sin(omega_norm)) / omega_norm * omega_unit_hat @ omega_unit_hat
            t = V @ v
        
        T = np.block([
            [R, t.reshape(3, 1)],
            [np.zeros((1, 3)), 1]
        ])
        return SE3(T)
    
    def exp_se3_from_xi(self, xi, theta=1.0):
        """
        从旋量坐标直接计算指数映射：e^(ξ^θ)
        """
        xi_hat = self.hat_operator(xi * theta)
        return self.exp_se3(xi_hat)
    
    def adjoint(self, T):
        """
        计算Adjoint变换：Ad_T
        
        Ad_T = [R    0]
                [p^R R]
        """
        R = T.R
        p = T.t
        p_hat = np.array([
            [0, -p[2], p[1]],
            [p[2], 0, -p[0]],
            [-p[1], p[0], 0]
        ])
        
        Ad = np.block([
            [R, np.zeros((3, 3))],
            [p_hat @ R, R]
        ])
        return Ad
    
    def compute_K_matrix(self, eta):
        """
        计算K矩阵（公式19）
        
        K = I6 + (4-a*sin(a)-4*cos(a))/(2*a^2) * Ω + ...
        """
        omega = eta[:3]
        v = eta[3:]
        
        a = np.linalg.norm(omega)
        
        if a < 1e-6:
            # 纯平移情况
            return np.eye(6)
        
        omega_hat = np.array([
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0]
        ])
        
        v_hat = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        
        Omega = np.block([
            [omega_hat, np.zeros((3, 3))],
            [v_hat, omega_hat]
        ])
        
        # 计算K矩阵的各项系数
        sin_a = np.sin(a)
        cos_a = np.cos(a)
        
        c1 = (4 - a * sin_a - 4 * cos_a) / (2 * a**2)
        c2 = (4 * a - 5 * sin_a + a * cos_a) / (2 * a**3)
        c3 = (2 - a * sin_a - 2 * cos_a) / (2 * a**4)
        c4 = (2 * a - 3 * sin_a + a * cos_a) / (2 * a**5)
        
        K = np.eye(6) + c1 * Omega + c2 * Omega @ Omega + \
            c3 * Omega @ Omega @ Omega + c4 * Omega @ Omega @ Omega @ Omega
        
        return K
    
    def forward_kinematics(self, q, screw_axes, Gamma):
        """
        正向运动学（公式2）
        
        f(θ) = e^(ξ₁θ₁) e^(ξ₂θ₂) ... e^(ξₙθₙ) e^Γ
        """
        T = SE3()
        
        # 计算各关节的指数映射
        for i, (xi, theta_i) in enumerate(zip(screw_axes, q)):
            exp_xi_theta = self.exp_se3_from_xi(xi, theta_i)
            T = T * exp_xi_theta
        
        # 零位旋量
        exp_Gamma = self.exp_se3_from_xi(Gamma, 1.0)
        T = T * exp_Gamma
        
        return T
    
    def compute_jacobian(self, q, screw_axes, Gamma):
        """
        计算雅可比矩阵（公式26）
        
        H = [J₁, J₂, ..., Jₙ, J_{n+1}]
        """
        n_joints = len(screw_axes)
        J_list = []
        
        # 计算各个中间变换
        transforms = []
        T_accum = SE3()
        
        # 零位旋量变换
        exp_Gamma = self.exp_se3_from_xi(Gamma, 1.0)
        transforms.append(exp_Gamma)
        T_accum = T_accum * exp_Gamma
        
        # 各关节变换
        for i, (xi, theta_i) in enumerate(zip(screw_axes, q)):
            exp_xi_theta = self.exp_se3_from_xi(xi, theta_i)
            T_accum = T_accum * exp_xi_theta
            transforms.append(exp_xi_theta)
        
        # 计算名义旋量（理论值，这里简化使用当前估计值）
        xi_n_list = screw_axes
        
        # 计算各列的雅可比
        for i in range(n_joints):
            if i == 0:
                # J₁ = [I - Ad(e^(η₁)e^(ξ₁ⁿθ₁)e^(-η₁))] K₁
                # 简化：假设η₁=0（初始误差为0）
                eta_i = np.zeros(6)
                exp_eta_i = SE3()
                exp_xi_n_theta = transforms[i+1]  # e^(ξ₁ⁿθ₁)
                T_i = exp_eta_i * exp_xi_n_theta * exp_eta_i.inv()
                Ad_Ti = self.adjoint(T_i)
                K_i = self.compute_K_matrix(eta_i)
                J_i = (np.eye(6) - Ad_Ti) @ K_i
            else:
                # J_i = Ad(∏...) [I - Ad(...)] K_i
                T_prod = SE3()
                for k in range(i):
                    T_prod = T_prod * transforms[k+1]
                
                eta_i = np.zeros(6)
                exp_eta_i = SE3()
                exp_xi_n_theta = transforms[i+1]
                T_i = exp_eta_i * exp_xi_n_theta * exp_eta_i.inv()
                Ad_Ti = self.adjoint(T_i)
                Ad_prod = self.adjoint(T_prod)
                K_i = self.compute_K_matrix(eta_i)
                J_i = Ad_prod @ (np.eye(6) - Ad_Ti) @ K_i
            
            J_list.append(J_i)
        
        # J_{n+1} = Ad(∏...) K_st
        T_prod = SE3()
        for k in range(n_joints):
            T_prod = T_prod * transforms[k+1]
        Ad_prod = self.adjoint(T_prod)
        K_st = self.compute_K_matrix(Gamma)
        J_n1 = Ad_prod @ K_st
        J_list.append(J_n1)
        
        # 组合成H矩阵
        H = np.hstack(J_list)
        
        return H
    
    def compute_error_vector(self, f_actual, f_nominal):
        """
        计算误差向量（公式22）
        
        y = (δf·f^(-1))^∨ = (f_a * f_n^(-1) - I4) 的向量化
        """
        delta_f = f_actual * f_nominal.inv()
        delta_f_matrix = delta_f.A
        
        # 提取旋转部分和平移部分
        R_delta = delta_f_matrix[:3, :3]
        t_delta = delta_f_matrix[:3, 3]
        
        # 旋转部分转换为轴角表示（使用对数映射）
        # log(R) = (θ/(2*sin(θ))) * (R - R^T)
        R_minus_RT = R_delta - R_delta.T
        trace_R = np.trace(R_delta)
        theta = np.arccos(np.clip((trace_R - 1) / 2, -1, 1))
        
        if abs(theta) < 1e-6:
            # 小角度近似
            omega_hat = R_minus_RT / 2
        else:
            omega_hat = theta / (2 * np.sin(theta)) * R_minus_RT
        
        # 从反对称矩阵提取ω
        omega = np.array([
            omega_hat[2, 1] - omega_hat[1, 2],
            omega_hat[0, 2] - omega_hat[2, 0],
            omega_hat[1, 0] - omega_hat[0, 1]
        ]) / 2
        
        # 组合成6维向量 [ω, v]
        y = np.concatenate([omega, t_delta])
        
        return y
    
    def calibrate(self, q_list, measured_poses, max_iterations=50, tolerance=1e-6):
        """
        校准POE参数（基于论文方法）
        
        Args:
            q_list: 关节角度列表
            measured_poses: 测量的末端位姿列表（SE3对象）
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
        """
        screw_axes = [xi.copy() for xi in self.initial_screw_axes]
        Gamma = self.Gamma_initial.copy()
        
        errors = []
        
        print("开始校准（基于论文方法）...")
        print(f"初始参数误差: {np.linalg.norm(self.phi_est - self.phi_true)}")
        
        for iteration in range(max_iterations):
            # 计算所有位姿的误差和雅可比
            Y_list = []
            L_list = []
            
            for q, f_measured in zip(q_list, measured_poses):
                # 计算名义位姿
                f_nominal = self.forward_kinematics(q, screw_axes, Gamma)
                
                # 计算误差向量
                y = self.compute_error_vector(f_measured, f_nominal)
                Y_list.append(y)
                
                # 计算雅可比矩阵
                H = self.compute_jacobian(q, screw_axes, Gamma)
                L_list.append(H)
            
            # 堆叠
            Y = np.vstack(Y_list)  # (k*6,)
            L = np.vstack(L_list)  # (k*6, 6(n+1))
            
            Y = Y.flatten()
            
            # 最小二乘求解（公式28）
            try:
                LTL = L.T @ L
                lambda_reg = 1e-6
                LTL_reg = LTL + lambda_reg * np.eye(LTL.shape[0])
                x = np.linalg.solve(LTL_reg, L.T @ Y)
            except:
                x = np.linalg.pinv(L) @ Y
            
            # 提取参数修正量
            n_joints = len(screw_axes)
            delta_eta_list = []
            for i in range(n_joints):
                delta_eta = x[i*6:(i+1)*6]
                delta_eta_list.append(delta_eta)
            delta_Gamma = x[-6:]
            
            # 限制步长
            max_step = 0.01
            step_norm = np.linalg.norm(x)
            if step_norm > max_step:
                x = x * (max_step / step_norm)
                for i in range(n_joints):
                    delta_eta_list[i] = delta_eta_list[i] * (max_step / step_norm)
                delta_Gamma = delta_Gamma * (max_step / step_norm)
            
            # 更新参数（公式29-30）
            # ξ_{i,k} = Ad(e^(δη_{i,k-1})) ξ_{i,k-1}
            for i, delta_eta in enumerate(delta_eta_list):
                exp_delta_eta = self.exp_se3_from_xi(delta_eta, 1.0)
                Ad_exp = self.adjoint(exp_delta_eta)
                screw_axes[i] = Ad_exp @ screw_axes[i]
            
            # Γ_k = Γ_{k-1} + δΓ_{k-1}
            Gamma = Gamma + delta_Gamma
            
            # 计算误差
            error_norm = np.linalg.norm(Y)
            phi_current = self._params_to_vector(screw_axes, Gamma)
            param_error = np.linalg.norm(phi_current - self.phi_true)
            errors.append(error_norm)
            
            print(f"迭代 {iteration+1}: 位置误差 = {error_norm:.6f}, 参数误差 = {param_error:.6f}, 步长 = {step_norm:.6f}")
            
            # 检查收敛
            if error_norm < tolerance or step_norm < tolerance:
                print(f"收敛于第 {iteration+1} 次迭代")
                break
            
            if len(errors) > 5 and abs(errors[-1] - errors[-5]) < tolerance * 10:
                print(f"位置误差已稳定，停止于第 {iteration+1} 次迭代")
                break
        
        self.screw_axes_est = screw_axes
        self.Gamma_est = Gamma
        return screw_axes, Gamma, errors
    
    def evaluate(self, q_list, measured_poses):
        """评估校准结果"""
        print("\n=== 校准结果评估（基于论文方法）===")
        
        # 真实参数
        print("\n真实关节旋量:")
        for i, xi in enumerate(self.true_screw_axes):
            print(f"  关节 {i+1}: {xi}")
        print(f"\n真实零位旋量: {self.Gamma_true}")
        
        # 估计参数
        print("\n估计关节旋量:")
        for i, xi in enumerate(self.screw_axes_est):
            print(f"  关节 {i+1}: {xi}")
        print(f"\n估计零位旋量: {self.Gamma_est}")
        
        # 参数误差
        print("\n参数误差:")
        for i, (xi_t, xi_e) in enumerate(zip(self.true_screw_axes, self.screw_axes_est)):
            error = np.linalg.norm(xi_t - xi_e)
            print(f"  关节 {i+1} 旋量误差: {error:.6f}")
        print(f"  零位旋量误差: {np.linalg.norm(self.Gamma_true - self.Gamma_est):.6f}")
        
        # 位置误差
        print("\n位置误差:")
        for i, (q, f_measured) in enumerate(zip(q_list, measured_poses)):
            f_theoretical = self.forward_kinematics(q, self.screw_axes_est, self.Gamma_est)
            error = np.linalg.norm(f_measured.t - f_theoretical.t)
            print(f"  位姿 {i+1}: {error:.6f} mm")


def main():
    """主函数"""
    print("=" * 60)
    print("基于指数积的串联机器人标定方法（论文方法）")
    print("=" * 60)
    
    # 定义2关节机器人的旋量参数
    # 旋转关节：ξ = [ω, v]^T，其中ω是旋转轴方向，v = -ω × p（p是轴上一点）
    
    # 真实旋量
    true_screw_axes = [
        np.array([0, 0, 1, 0, 0, 0]),      # 关节1：绕z轴旋转
        np.array([0, 0, 1, 0, -0.3, 0])   # 关节2：绕z轴旋转，轴上一点在y=-0.3
    ]
    Gamma_true = np.array([0, 0, 0, 0.55, 0, 0])  # 零位旋量（位置在x=0.55）
    
    # 初始估计（带误差）
    initial_screw_axes = [
        np.array([0.01, 0.01, 1.0, 0, 0, 0]),
        np.array([0.01, 0.01, 1.0, 0, -0.29, 0])
    ]
    Gamma_initial = np.array([0, 0, 0, 0.54, 0.01, 0])
    
    # 创建校准对象
    calibrator = POECalibrationPaper(
        true_screw_axes=true_screw_axes,
        initial_screw_axes=initial_screw_axes,
        Gamma_true=Gamma_true,
        Gamma_initial=Gamma_initial
    )
    
    # 生成测试位姿
    np.random.seed(42)
    n_poses = 20
    q_list = []
    measured_poses = []
    
    for _ in range(n_poses):
        q = np.random.uniform(-np.pi/2, np.pi/2, 2)
        q_list.append(q)
        
        # 使用真实参数计算测量位姿
        f_measured = calibrator.forward_kinematics(q, true_screw_axes, Gamma_true)
        # 添加噪声
        noise_pos = np.random.normal(0, 0.001, 3)
        f_measured = f_measured * SE3.Tx(noise_pos[0]) * SE3.Ty(noise_pos[1]) * SE3.Tz(noise_pos[2])
        measured_poses.append(f_measured)
    
    print("\n生成测量数据...")
    
    # 执行校准
    print("\n" + "=" * 60)
    screw_axes_final, Gamma_final, errors = calibrator.calibrate(
        q_list,
        measured_poses,
        max_iterations=50,
        tolerance=1e-6
    )
    
    # 评估结果
    calibrator.evaluate(q_list, measured_poses)
    
    # 绘制误差收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(errors, 'b-o', linewidth=2, markersize=4)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('位置误差', fontsize=12)
    plt.title('基于论文方法的POE校准误差收敛曲线', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    output_path = '/home/frank/extra_storage/Frank/doc/RL/springer_robotics_converted/poe_paper_calibration_convergence.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n误差收敛曲线已保存到: poe_paper_calibration_convergence.png")
    
    print("\n" + "=" * 60)
    print("基于论文方法的POE校准完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

