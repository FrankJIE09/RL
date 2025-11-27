#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于第14章方法的DH参数校准
使用Robotics Toolbox实现机器人运动学校准
"""

import numpy as np
import matplotlib.pyplot as plt
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3
import warnings
warnings.filterwarnings('ignore')

# 导入中文字体配置
try:
    from font_config import init_chinese_font
    init_chinese_font(verbose=False)
except ImportError:
    # 如果font_config模块不存在，使用备用方案
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

class DHCalibration:
    """DH参数校准类"""
    
    def __init__(self, true_dh_params, initial_dh_params, camera_params=None):
        """
        初始化
        
        Args:
            true_dh_params: 真实DH参数列表 [(a, d, alpha, theta), ...]
            initial_dh_params: 初始估计DH参数列表（带误差）
            camera_params: 相机参数 {'d0': value, 'theta0': value, 'a1': value, ...}
        """
        self.true_dh_params = true_dh_params
        self.initial_dh_params = initial_dh_params
        self.camera_params = camera_params or {}
        
        # 创建机器人模型
        self.robot_true = self._create_robot(true_dh_params)
        self.robot_est = self._create_robot(initial_dh_params)
        
        # 参数向量（包括相机参数和DH参数）
        self.phi_true = self._params_to_vector(true_dh_params, camera_params)
        self.phi_est = self._params_to_vector(initial_dh_params, camera_params)
        
    def _create_robot(self, dh_params):
        """创建机器人模型"""
        links = []
        for a, d, alpha, theta in dh_params:
            # RevoluteDH参数：a, d, alpha, offset(对应theta)
            links.append(RevoluteDH(a=a, d=d, alpha=alpha, offset=theta))
        return DHRobot(links, name='Robot')
    
    def _params_to_vector(self, dh_params, camera_params):
        """将参数转换为向量"""
        phi = []
        # 相机参数
        phi.extend([camera_params.get('d0', 0), camera_params.get('theta0', 0)])
        phi.extend([camera_params.get('a1', 0), camera_params.get('d1', 0), 
                    camera_params.get('alpha1', 0), camera_params.get('theta1', 0)])
        # DH参数
        for a, d, alpha, theta in dh_params:
            phi.extend([a, d, alpha, theta])
        return np.array(phi)
    
    def _vector_to_params(self, phi):
        """将向量转换为参数"""
        idx = 0
        camera_params = {
            'd0': phi[idx], 'theta0': phi[idx+1],
            'a1': phi[idx+2], 'd1': phi[idx+3], 
            'alpha1': phi[idx+4], 'theta1': phi[idx+5]
        }
        idx += 6
        dh_params = []
        n_links = (len(phi) - idx) // 4
        for i in range(n_links):
            dh_params.append((phi[idx], phi[idx+1], phi[idx+2], phi[idx+3]))
            idx += 4
        return dh_params, camera_params
    
    def forward_kinematics(self, q, phi):
        """
        正向运动学（使用参数phi）
        
        Args:
            q: 关节角度
            phi: 参数向量
        """
        dh_params, camera_params = self._vector_to_params(phi)
        
        # 计算相机到中间坐标系的变换
        d0 = camera_params['d0']
        theta0 = camera_params['theta0']
        T_camera_intermediate = SE3.Tz(d0) * SE3.Rz(theta0)
        
        # 计算中间坐标系到基坐标系的变换
        a1 = camera_params['a1']
        d1 = camera_params['d1']
        alpha1 = camera_params['alpha1']
        theta1 = camera_params['theta1']
        T_intermediate_base = SE3.Rx(alpha1) * SE3.Tx(a1) * SE3.Tz(d1) * SE3.Rz(theta1 + q[0])
        
        # 计算机器人运动学链
        T_base_end = SE3()
        for i, (a, d, alpha, theta) in enumerate(dh_params):
            T_base_end = T_base_end * (
                SE3.Rx(alpha) * SE3.Tx(a) * SE3.Tz(d) * SE3.Rz(theta + q[i])
            )
        
        # 完整变换链（假设末端到测量点的变换为单位变换）
        T_camera_end = T_camera_intermediate * T_intermediate_base * T_base_end
        
        return T_camera_end
    
    def compute_jacobian(self, q, phi):
        """
        计算雅可比矩阵（基于书中公式14.14-14.18）
        
        Args:
            q: 关节角度
            phi: 参数向量
        """
        dh_params, camera_params = self._vector_to_params(phi)
        
        # 计算完整变换链和所有中间变换
        transforms = []
        origins = []  # 存储各个坐标系的原点位置
        
        # 相机到中间坐标系
        d0 = camera_params['d0']
        theta0 = camera_params['theta0']
        T_camera_intermediate = SE3.Tz(d0) * SE3.Rz(theta0)
        transforms.append(T_camera_intermediate)
        origins.append(np.array([0, 0, 0]))  # 相机坐标系原点
        origins.append(T_camera_intermediate.t)  # 中间坐标系原点
        
        # 中间到基坐标系
        a1 = camera_params['a1']
        d1 = camera_params['d1']
        alpha1 = camera_params['alpha1']
        theta1 = camera_params['theta1']
        T_intermediate_base = SE3.Rx(alpha1) * SE3.Tx(a1) * SE3.Tz(d1) * SE3.Rz(theta1 + q[0])
        T_camera_base = T_camera_intermediate * T_intermediate_base
        transforms.append(T_intermediate_base)
        origins.append(T_camera_base.t)  # 基坐标系原点
        
        # 机器人运动学链
        T_camera_current = T_camera_base
        for i, (a, d, alpha, theta) in enumerate(dh_params):
            T_link = SE3.Rx(alpha) * SE3.Tx(a) * SE3.Tz(d) * SE3.Rz(theta + q[i])
            T_camera_current = T_camera_current * T_link
            transforms.append(T_link)
            origins.append(T_camera_current.t)  # 下一个坐标系原点
        
        # 末端位置（测量点）
        p_end = T_camera_current.t
        
        # 计算雅可比矩阵（按照书中公式14.14-14.18）
        J = []
        
        # 计算各个坐标系在相机坐标系中的表示
        R_list = []
        p_list = []
        T_accum = SE3()
        
        # 相机到中间
        T_accum = T_accum * transforms[0]
        R_list.append(T_accum.R)
        p_list.append(T_accum.t)
        
        # 中间到基座
        T_accum = T_accum * transforms[1]
        R_list.append(T_accum.R)
        p_list.append(T_accum.t)
        
        # 机器人链
        for i in range(len(dh_params)):
            T_accum = T_accum * transforms[2+i]
            R_list.append(T_accum.R)
            p_list.append(T_accum.t)
        
        # 计算原点到测量点的向量（在相机坐标系中）
        d_vectors = []
        for p in p_list:
            d_vectors.append(p_end - p)
        
        # 按照书中公式计算雅可比
        
        # 1. 对d0的雅可比（公式14.15类似）
        z0_camera = R_list[0] @ np.array([0, 0, 1])
        J.append(z0_camera)
        
        # 2. 对theta0的雅可比（公式14.17）
        z0_camera_theta = R_list[0] @ np.array([0, 0, 1])
        d_0_end = d_vectors[0]
        J.append(np.cross(z0_camera_theta, d_0_end))
        
        # 3. 对a1的雅可比（公式14.14）
        x0_camera = R_list[0] @ np.array([1, 0, 0])
        J.append(x0_camera)
        
        # 4. 对d1的雅可比（公式14.15）
        z1_camera = R_list[1] @ np.array([0, 0, 1])
        J.append(z1_camera)
        
        # 5. 对alpha1的雅可比（公式14.16）
        x0_alpha = R_list[0] @ np.array([1, 0, 0])
        d_0_end_alpha = d_vectors[0]
        J.append(np.cross(x0_alpha, d_0_end_alpha))
        
        # 6. 对theta1的雅可比（公式14.17）
        z1_theta = R_list[1] @ np.array([0, 0, 1])
        d_1_end = d_vectors[1]
        J.append(np.cross(z1_theta, d_1_end))
        
        # 7. 对DH参数的雅可比
        for i, (a, d, alpha, theta) in enumerate(dh_params):
            idx = 2 + i  # 在R_list和p_list中的索引
            
            # a参数（公式14.14）
            x_i_camera = R_list[idx-1] @ np.array([1, 0, 0])
            J.append(x_i_camera)
            
            # d参数（公式14.15）
            z_i_camera = R_list[idx] @ np.array([0, 0, 1])
            J.append(z_i_camera)
            
            # alpha参数（公式14.16）
            x_i_alpha = R_list[idx-1] @ np.array([1, 0, 0])
            d_i_end = d_vectors[idx-1]
            J.append(np.cross(x_i_alpha, d_i_end))
            
            # theta参数（公式14.17）
            z_i_theta = R_list[idx] @ np.array([0, 0, 1])
            d_i_end_theta = d_vectors[idx]
            J.append(np.cross(z_i_theta, d_i_end_theta))
        
        return np.array(J).T
    
    def calibrate(self, q_list, measured_positions, max_iterations=50, tolerance=1e-6):
        """
        校准DH参数
        
        Args:
            q_list: 关节角度列表
            measured_positions: 测量的末端位置列表
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
        """
        phi = self.phi_est.copy()
        errors = []
        
        print("开始校准...")
        print(f"初始参数误差: {np.linalg.norm(phi - self.phi_true)}")
        
        for iteration in range(max_iterations):
            # 计算所有位姿的误差和雅可比
            J_list = []
            error_list = []
            
            for q, p_measured in zip(q_list, measured_positions):
                # 计算理论位置
                T_theoretical = self.forward_kinematics(q, phi)
                p_theoretical = T_theoretical.t
                
                # 计算误差（确保是1D数组）
                error = np.array(p_measured) - np.array(p_theoretical)
                if error.ndim > 1:
                    error = error.flatten()
                error_list.append(error)
                
                # 计算雅可比
                J = self.compute_jacobian(q, phi)
                J_list.append(J)
            
            # 堆叠
            J_stack = np.vstack(J_list)  # (3*P) × N_params
            error_stack = np.hstack(error_list)  # (3*P,)
            
            # 确保error_stack是1D数组
            error_stack = error_stack.flatten()
            
            # 检查维度
            if J_stack.shape[0] != len(error_stack):
                print(f"警告：维度不匹配 J: {J_stack.shape}, error: {error_stack.shape}")
                print(f"J_list长度: {len(J_list)}, error_list长度: {len(error_list)}")
                if len(J_list) > 0:
                    print(f"单个J形状: {J_list[0].shape}")
                if len(error_list) > 0:
                    print(f"单个error形状: {error_list[0].shape}")
                break
            
            # 最小二乘求解（添加正则化以提高数值稳定性）
            try:
                # 使用岭回归（ridge regression）提高数值稳定性
                lambda_reg = 1e-6
                JtJ = J_stack.T @ J_stack
                JtJ_reg = JtJ + lambda_reg * np.eye(JtJ.shape[0])
                delta_phi = np.linalg.solve(JtJ_reg, J_stack.T @ error_stack)
            except Exception as e:
                print(f"求解错误: {e}")
                # 如果矩阵奇异，使用伪逆
                delta_phi = np.linalg.pinv(J_stack) @ error_stack
            
            # 限制步长（防止参数变化过大）
            max_step = 0.1  # 最大步长
            step_norm = np.linalg.norm(delta_phi)
            if step_norm > max_step:
                delta_phi = delta_phi * (max_step / step_norm)
            
            # 更新参数
            phi = phi + delta_phi
            
            # 计算误差
            error_norm = np.linalg.norm(error_stack)
            param_error = np.linalg.norm(phi - self.phi_true)
            errors.append(error_norm)
            
            print(f"迭代 {iteration+1}: 位置误差 = {error_norm:.6f}, 参数误差 = {param_error:.6f}, 步长 = {np.linalg.norm(delta_phi):.6f}")
            
            # 检查收敛（基于位置误差和步长）
            if error_norm < tolerance or np.linalg.norm(delta_phi) < tolerance:
                print(f"收敛于第 {iteration+1} 次迭代")
                break
            
            # 如果位置误差不再减小，也停止
            if len(errors) > 5 and abs(errors[-1] - errors[-5]) < tolerance * 10:
                print(f"位置误差已稳定，停止于第 {iteration+1} 次迭代")
                break
        
        self.phi_est = phi
        return phi, errors
    
    def evaluate(self, q_list, measured_positions):
        """评估校准结果"""
        print("\n=== 校准结果评估 ===")
        
        # 真实参数
        dh_params_true, _ = self._vector_to_params(self.phi_true)
        print("\n真实DH参数:")
        for i, (a, d, alpha, theta) in enumerate(dh_params_true):
            print(f"  关节 {i+1}: a={a:.4f}, d={d:.4f}, alpha={alpha:.4f}, theta={theta:.4f}")
        
        # 估计参数
        dh_params_est, camera_params_est = self._vector_to_params(self.phi_est)
        print("\n估计DH参数:")
        for i, (a, d, alpha, theta) in enumerate(dh_params_est):
            print(f"  关节 {i+1}: a={a:.4f}, d={d:.4f}, alpha={alpha:.4f}, theta={theta:.4f}")
        
        # 参数误差
        print("\n参数误差:")
        for i, ((a_t, d_t, alpha_t, theta_t), (a_e, d_e, alpha_e, theta_e)) in enumerate(
            zip(dh_params_true, dh_params_est)):
            print(f"  关节 {i+1}:")
            print(f"    a误差: {abs(a_t - a_e):.6f}")
            print(f"    d误差: {abs(d_t - d_e):.6f}")
            print(f"    alpha误差: {abs(alpha_t - alpha_e):.6f}")
            print(f"    theta误差: {abs(theta_t - theta_e):.6f}")
        
        # 位置误差
        print("\n位置误差:")
        for i, (q, p_measured) in enumerate(zip(q_list, measured_positions)):
            T_theoretical = self.forward_kinematics(q, self.phi_est)
            p_theoretical = T_theoretical.t
            error = np.linalg.norm(p_measured - p_theoretical)
            print(f"  位姿 {i+1}: {error:.6f} mm")


def generate_test_data(robot_true, q_list, noise_level=0.1):
    """
    生成测试数据（模拟3D相机测量）
    
    Args:
        robot_true: 真实机器人模型
        q_list: 关节角度列表
        noise_level: 噪声水平（mm）
    """
    measured_positions = []
    
    for q in q_list:
        # 计算真实末端位置（在基坐标系中）
        T_base_end = robot_true.fkine(q)
        p_base = T_base_end.t
        
        # 模拟相机坐标系（假设相机在基坐标系上方）
        # 这里简化处理，直接使用基坐标系位置
        # 实际应用中需要加上相机到基坐标系的变换
        
        # 添加测量噪声
        noise = np.random.normal(0, noise_level, 3)
        p_measured = p_base + noise
        
        measured_positions.append(p_measured)
    
    return measured_positions


def main():
    """主函数"""
    print("=" * 60)
    print("DH参数校准示例")
    print("=" * 60)
    
    # 定义真实DH参数（2关节机器人示例）
    true_dh_params = [
        (0.3, 0.0, 0.0, 0.0),   # 关节1: a=300mm
        (0.25, 0.0, 0.0, 0.0),  # 关节2: a=250mm
    ]
    
    # 定义初始估计参数（带较小误差，便于收敛）
    initial_dh_params = [
        (0.2, 0.0, 0.0, 0.0),   # 关节1: 误差-10mm
        (0.2, 0.0, 0.0, 0.0),   # 关节2: 误差+10mm
    ]
    
    # 相机参数（真实值）
    camera_params_true = {
        'd0': 0.1,      # 100mm
        'theta0': np.pi/6,  # 30度
        'a1': 0.2,     # 200mm
        'd1': 0.05,    # 50mm
        'alpha1': 0.0,
        'theta1': 0.0
    }
    
    # 相机参数（初始估计，带较小误差）
    camera_params_initial = {
        'd0': 0.098,   # 误差-2mm
        'theta0': np.pi/6 + 0.02,  # 误差+1.15度
        'a1': 0.198,   # 误差-2mm
        'd1': 0.051,   # 误差+1mm
        'alpha1': 0.005,  # 误差+0.29度
        'theta1': -0.005  # 误差-0.29度
    }
    
    # 创建校准对象
    calibrator = DHCalibration(
        true_dh_params=true_dh_params,
        initial_dh_params=initial_dh_params,
        camera_params=camera_params_initial
    )
    
    # 设置真实相机参数（用于生成测量数据）
    calibrator.camera_params = camera_params_true
    
    # 生成测试位姿
    np.random.seed(42)
    n_poses = 20
    q_list = []
    for _ in range(n_poses):
        q = np.random.uniform(-np.pi/2, np.pi/2, 2)
        q_list.append(q)
    
    # 生成测量数据（使用真实参数计算，加上相机变换）
    print("\n生成测量数据...")
    measured_positions = []
    for q in q_list:
        # 使用真实参数和正向运动学计算末端位置（在相机坐标系中）
        # 这里直接使用calibrator的forward_kinematics，但使用真实参数
        T_camera_end = calibrator.forward_kinematics(q, calibrator.phi_true)
        p_camera = np.array(T_camera_end.t).flatten()  # 确保是1D数组
        
        # 添加测量噪声
        noise = np.random.normal(0, 0.001, 3)  # 1mm噪声
        p_measured = p_camera + noise
        
        measured_positions.append(p_measured)
    
    # 恢复初始相机参数估计
    calibrator.camera_params = camera_params_initial
    
    # 执行校准
    print("\n" + "=" * 60)
    phi_final, errors = calibrator.calibrate(
        q_list, 
        measured_positions,
        max_iterations=50,
        tolerance=1e-6
    )
    
    # 评估结果
    calibrator.evaluate(q_list, measured_positions)
    
    # 绘制误差收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(errors, 'b-o', linewidth=2, markersize=4)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('位置误差 (mm)', fontsize=12)
    plt.title('校准误差收敛曲线', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    output_path = '/home/frank/extra_storage/Frank/doc/RL/springer_robotics_converted/calibration_convergence.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n误差收敛曲线已保存到: calibration_convergence.png")
    
    print("\n" + "=" * 60)
    print("校准完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

