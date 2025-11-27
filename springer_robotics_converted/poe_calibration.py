#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于POE（Product of Exponentials）方法的机器人运动学校准
使用螺旋理论（Screw Theory）实现机器人运动学校准
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


class POECalibration:
    """使用POE方法的机器人校准类"""
    
    def __init__(self, true_screw_axes, initial_screw_axes, M0_true, M0_initial, camera_params=None):
        """
        初始化
        
        Args:
            true_screw_axes: 真实螺旋轴列表 [[ω1, v1], [ω2, v2], ...]
            initial_screw_axes: 初始估计螺旋轴列表（带误差）
            M0_true: 真实零位末端位姿 SE3
            M0_initial: 初始估计零位末端位姿 SE3
            camera_params: 相机参数 {'d0': value, 'theta0': value, 'a1': value, ...}
        """
        self.true_screw_axes = true_screw_axes
        self.initial_screw_axes = initial_screw_axes
        self.M0_true = M0_true
        self.M0_initial = M0_initial
        self.camera_params = camera_params or {}
        
        # 参数向量（包括相机参数、螺旋轴参数、零位位姿参数）
        self.phi_true = self._params_to_vector(true_screw_axes, M0_true, camera_params)
        self.phi_est = self._params_to_vector(initial_screw_axes, M0_initial, camera_params)
    
    def _params_to_vector(self, screw_axes, M0, camera_params):
        """将参数转换为向量"""
        phi = []
        # 相机参数
        phi.extend([camera_params.get('d0', 0), camera_params.get('theta0', 0)])
        phi.extend([camera_params.get('a1', 0), camera_params.get('d1', 0), 
                    camera_params.get('alpha1', 0), camera_params.get('theta1', 0)])
        # 螺旋轴参数（每个螺旋轴6个参数：ω(3) + v(3)）
        for omega, v in screw_axes:
            phi.extend(list(omega))
            phi.extend(list(v))
        # 零位位姿参数（位置3个 + 方向3个）
        p0 = M0.t
        R0 = M0.R
        # 将旋转矩阵转换为轴角表示（简化：使用前3个元素）
        # 或者使用欧拉角
        euler = self._rotation_matrix_to_euler(R0)
        phi.extend(list(p0))
        phi.extend(list(euler))
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
        
        # 螺旋轴参数
        screw_axes = []
        n_joints = (len(phi) - idx - 6) // 6  # 减去零位位姿的6个参数
        for i in range(n_joints):
            omega = phi[idx:idx+3]
            v = phi[idx+3:idx+6]
            screw_axes.append((omega, v))
            idx += 6
        
        # 零位位姿参数
        p0 = phi[idx:idx+3]
        euler = phi[idx+3:idx+6]
        R0 = self._euler_to_rotation_matrix(euler)
        M0 = SE3.Rt(R0, p0)
        
        return screw_axes, M0, camera_params
    
    def _rotation_matrix_to_euler(self, R):
        """旋转矩阵转欧拉角（ZYX顺序）"""
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        return np.array([x, y, z])
    
    def _euler_to_rotation_matrix(self, euler):
        """欧拉角转旋转矩阵（ZYX顺序）"""
        x, y, z = euler
        cx, sx = np.cos(x), np.sin(x)
        cy, sy = np.cos(y), np.sin(y)
        cz, sz = np.cos(z), np.sin(z)
        
        R = np.array([
            [cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx],
            [sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx],
            [-sy, cy*sx, cy*cx]
        ])
        return R
    
    def screw_to_se3(self, omega, v, theta):
        """
        将螺旋坐标转换为SE(3)变换（指数映射）
        
        Args:
            omega: 旋转轴方向（3维向量）
            v: 线速度（3维向量）
            theta: 关节角度
        """
        omega_norm = np.linalg.norm(omega)
        
        if omega_norm < 1e-6:
            # 纯平移关节
            return SE3.Tx(v[0]*theta) * SE3.Ty(v[1]*theta) * SE3.Tz(v[2]*theta)
        
        # 归一化
        omega_unit = omega / omega_norm
        v_unit = v / omega_norm
        
        # 计算旋转矩阵（Rodrigues公式）
        omega_hat = np.array([
            [0, -omega_unit[2], omega_unit[1]],
            [omega_unit[2], 0, -omega_unit[0]],
            [-omega_unit[1], omega_unit[0], 0]
        ])
        
        theta_scaled = theta * omega_norm
        R = np.eye(3) + np.sin(theta_scaled) * omega_hat + (1 - np.cos(theta_scaled)) * omega_hat @ omega_hat
        
        # 计算平移向量
        # t = (I - R) * (ω × v) / ||ω||² + ω * ω^T * v * θ / ||ω||
        omega_cross_v = np.cross(omega_unit, v_unit)
        I_minus_R = np.eye(3) - R
        t = I_minus_R @ omega_cross_v / omega_norm + omega_unit * np.dot(omega_unit, v_unit) * theta
        
        return SE3.Rt(R, t)
    
    def _compute_camera_transform(self, camera_params, q):
        """计算相机到基坐标系的变换"""
        d0 = camera_params['d0']
        theta0 = camera_params['theta0']
        T_camera_intermediate = SE3.Tz(d0) * SE3.Rz(theta0)
        
        a1 = camera_params['a1']
        d1 = camera_params['d1']
        alpha1 = camera_params['alpha1']
        theta1 = camera_params['theta1']
        T_intermediate_base = SE3.Rx(alpha1) * SE3.Tx(a1) * SE3.Tz(d1) * SE3.Rz(theta1 + q[0])
        
        return T_camera_intermediate * T_intermediate_base
    
    def forward_kinematics(self, q, phi):
        """
        使用POE方法计算正向运动学
        
        POE公式：T(θ) = exp([ξ₁]θ₁) × exp([ξ₂]θ₂) × ... × exp([ξₙ]θₙ) × M(0)
        
        Args:
            q: 关节角度
            phi: 参数向量
        """
        screw_axes, M0, camera_params = self._vector_to_params(phi)
        
        # 计算相机到基坐标系的变换
        T_camera_base = self._compute_camera_transform(camera_params, q)
        
        # POE公式：T = exp(ξ₁θ₁) × exp(ξ₂θ₂) × ... × exp(ξₙθₙ) × M(0)
        T_base_end = M0
        for i, (omega, v) in enumerate(screw_axes):
            exp_xi_theta = self.screw_to_se3(omega, v, q[i])
            T_base_end = T_base_end * exp_xi_theta
        
        # 完整变换链
        T_camera_end = T_camera_base * T_base_end
        
        return T_camera_end
    
    def _adjoint(self, T):
        """
        计算Adjoint变换
        
        Ad_T = [R    0]
                [p^R R]
        
        其中 p^ 是p的反对称矩阵
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
    
    def compute_jacobian(self, q, phi):
        """
        计算雅可比矩阵（POE方法）
        
        POE方法的雅可比：
        J_i = Ad_{exp(ξ₁θ₁)...exp(ξᵢ₋₁θᵢ₋₁)M(0)} (ξᵢ)
        
        对于位置雅可比，只需要前3行
        """
        screw_axes, M0, camera_params = self._vector_to_params(phi)
        
        # 计算所有中间变换
        transforms = []
        T_accum = M0
        transforms.append(T_accum)
        
        for i, (omega, v) in enumerate(screw_axes):
            exp_xi_theta = self.screw_to_se3(omega, v, q[i])
            T_accum = T_accum * exp_xi_theta
            transforms.append(T_accum)
        
        # 计算雅可比矩阵
        J = []
        
        # 相机参数的雅可比（类似DH方法）
        T_camera_base = self._compute_camera_transform(camera_params, q)
        T_camera_intermediate = SE3.Tz(camera_params['d0']) * SE3.Rz(camera_params['theta0'])
        
        # d0的雅可比
        z0_camera = T_camera_intermediate.R @ np.array([0, 0, 1])
        J.append(z0_camera)
        
        # theta0的雅可比
        z0_theta = T_camera_intermediate.R @ np.array([0, 0, 1])
        p_end = (T_camera_base * transforms[-1]).t
        p_intermediate = T_camera_intermediate.t
        d_0_end = p_end - p_intermediate
        J.append(np.cross(z0_theta, d_0_end))
        
        # a1的雅可比
        x0_camera = T_camera_intermediate.R @ np.array([1, 0, 0])
        J.append(x0_camera)
        
        # d1的雅可比
        T_intermediate_base = SE3.Rx(camera_params['alpha1']) * SE3.Tx(camera_params['a1']) * \
                              SE3.Tz(camera_params['d1']) * SE3.Rz(camera_params['theta1'] + q[0])
        z1_camera = (T_camera_intermediate * T_intermediate_base).R @ np.array([0, 0, 1])
        J.append(z1_camera)
        
        # alpha1的雅可比
        x0_alpha = T_camera_intermediate.R @ np.array([1, 0, 0])
        d_0_end_alpha = p_end - p_intermediate
        J.append(np.cross(x0_alpha, d_0_end_alpha))
        
        # theta1的雅可比
        z1_theta = (T_camera_intermediate * T_intermediate_base).R @ np.array([0, 0, 1])
        p_base = (T_camera_intermediate * T_intermediate_base).t
        d_1_end = p_end - p_base
        J.append(np.cross(z1_theta, d_1_end))
        
        # POE参数的雅可比
        # 对于旋转关节，通常只校准螺旋轴的方向（单位向量）
        # 这里简化：假设ω是单位向量，只校准v（3个参数）
        T_camera_current = T_camera_base
        for i, (omega, v) in enumerate(screw_axes):
            # 计算当前变换
            T_i = T_camera_current * transforms[i]
            
            # 对ω参数的雅可比（假设ω是单位向量，只校准方向）
            # 使用数值微分或解析方法
            # 简化：使用Adjoint变换
            Ad_Ti = self._adjoint(T_i)
            
            # 对v参数的雅可比（3个参数）
            # v的变化直接影响线速度
            for j in range(3):
                v_j = np.zeros(3)
                v_j[j] = 1.0
                # v的变化对位置的影响 = Ad_Ti的下半部分（平移部分）
                # 简化：直接使用单位向量
                J.append(Ad_Ti[3:, 3:][:, j])  # 位置雅可比
            
            # 对ω参数的雅可比（简化：假设ω是单位向量）
            # 只校准ω的方向（2个参数，因为单位向量只有2个自由度）
            omega_norm = np.linalg.norm(omega)
            if omega_norm > 1e-6:
                omega_unit = omega / omega_norm
                # 使用球坐标参数化
                # 简化：直接使用3个参数，但添加约束
                for j in range(3):
                    omega_j = np.zeros(3)
                    omega_j[j] = 1.0
                    xi_j = np.concatenate([omega_j, np.zeros(3)])
                    J_col_full = Ad_Ti @ xi_j
                    J.append(J_col_full[:3])  # 位置雅可比
            else:
                # 纯平移关节
                for j in range(3):
                    J.append(np.zeros(3))
            
            # 更新累积变换
            T_camera_current = T_camera_current * transforms[i+1]
        
        # M0参数的雅可比
        # 位置参数（3个）
        for i in range(3):
            J.append(np.eye(3)[:, i])  # 单位向量
        
        # 方向参数（3个，简化处理）
        for i in range(3):
            # 方向变化对位置的影响（小角度近似）
            # 使用叉积近似
            p_end = (T_camera_base * transforms[-1]).t
            euler_axis = np.zeros(3)
            euler_axis[i] = 1.0
            J.append(np.cross(euler_axis, p_end))
        
        return np.array(J).T
    
    def calibrate(self, q_list, measured_positions, max_iterations=50, tolerance=1e-6):
        """
        校准POE参数
        
        Args:
            q_list: 关节角度列表
            measured_positions: 测量的末端位置列表
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
        """
        phi = self.phi_est.copy()
        errors = []
        
        print("开始校准（POE方法）...")
        print(f"初始参数误差: {np.linalg.norm(phi - self.phi_true)}")
        
        for iteration in range(max_iterations):
            # 计算所有位姿的误差和雅可比
            J_list = []
            error_list = []
            
            for q, p_measured in zip(q_list, measured_positions):
                # 计算理论位置
                T_theoretical = self.forward_kinematics(q, phi)
                p_theoretical = T_theoretical.t
                
                # 计算误差
                error = np.array(p_measured) - np.array(p_theoretical)
                if error.ndim > 1:
                    error = error.flatten()
                error_list.append(error)
                
                # 计算雅可比
                J = self.compute_jacobian(q, phi)
                J_list.append(J)
            
            # 堆叠
            J_stack = np.vstack(J_list)
            error_stack = np.hstack(error_list)
            error_stack = error_stack.flatten()
            
            # 检查维度
            if J_stack.shape[0] != len(error_stack):
                print(f"警告：维度不匹配 J: {J_stack.shape}, error: {error_stack.shape}")
                break
            
            # 最小二乘求解（添加正则化）
            try:
                lambda_reg = 1e-6
                JtJ = J_stack.T @ J_stack
                JtJ_reg = JtJ + lambda_reg * np.eye(JtJ.shape[0])
                delta_phi = np.linalg.solve(JtJ_reg, J_stack.T @ error_stack)
            except Exception as e:
                print(f"求解错误: {e}")
                delta_phi = np.linalg.pinv(J_stack) @ error_stack
            
            # 限制步长（POE方法需要更小的步长）
            max_step = 0.01  # 减小步长
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
            
            # 检查收敛
            if error_norm < tolerance or np.linalg.norm(delta_phi) < tolerance:
                print(f"收敛于第 {iteration+1} 次迭代")
                break
            
            if len(errors) > 5 and abs(errors[-1] - errors[-5]) < tolerance * 10:
                print(f"位置误差已稳定，停止于第 {iteration+1} 次迭代")
                break
        
        self.phi_est = phi
        return phi, errors
    
    def evaluate(self, q_list, measured_positions):
        """评估校准结果"""
        print("\n=== 校准结果评估（POE方法）===")
        
        # 真实参数
        screw_axes_true, M0_true, _ = self._vector_to_params(self.phi_true)
        print("\n真实螺旋轴:")
        for i, (omega, v) in enumerate(screw_axes_true):
            print(f"  关节 {i+1}: ω={omega}, v={v}")
        print(f"\n真实零位位姿:")
        print(f"  位置: {M0_true.t}")
        print(f"  旋转矩阵:\n{M0_true.R}")
        
        # 估计参数
        screw_axes_est, M0_est, camera_params_est = self._vector_to_params(self.phi_est)
        print("\n估计螺旋轴:")
        for i, (omega, v) in enumerate(screw_axes_est):
            print(f"  关节 {i+1}: ω={omega}, v={v}")
        print(f"\n估计零位位姿:")
        print(f"  位置: {M0_est.t}")
        print(f"  旋转矩阵:\n{M0_est.R}")
        
        # 参数误差
        print("\n参数误差:")
        for i, ((omega_t, v_t), (omega_e, v_e)) in enumerate(zip(screw_axes_true, screw_axes_est)):
            print(f"  关节 {i+1}:")
            print(f"    ω误差: {np.linalg.norm(omega_t - omega_e):.6f}")
            print(f"    v误差: {np.linalg.norm(v_t - v_e):.6f}")
        
        print(f"\n零位位姿误差:")
        print(f"  位置误差: {np.linalg.norm(M0_true.t - M0_est.t):.6f}")
        
        # 位置误差
        print("\n位置误差:")
        for i, (q, p_measured) in enumerate(zip(q_list, measured_positions)):
            T_theoretical = self.forward_kinematics(q, self.phi_est)
            p_theoretical = T_theoretical.t
            error = np.linalg.norm(p_measured - p_theoretical)
            print(f"  位姿 {i+1}: {error:.6f} mm")


def dh_to_poe(dh_params):
    """
    将DH参数转换为POE参数（辅助函数）
    
    对于标准旋转关节，DH参数可以转换为POE参数
    """
    screw_axes = []
    M0 = SE3()
    
    # 计算零位时的末端位姿
    for a, d, alpha, theta in dh_params:
        M0 = M0 * (SE3.Rx(alpha) * SE3.Tx(a) * SE3.Tz(d) * SE3.Rz(theta))
    
    # 计算每个关节的螺旋轴
    T_accum = SE3()
    for i, (a, d, alpha, theta) in enumerate(dh_params):
        # 旋转轴（z轴方向）
        z_axis = T_accum.R @ np.array([0, 0, 1])
        
        # 轴上一点
        p_on_axis = T_accum.t
        
        # 线速度 v = -ω × p
        omega = z_axis
        v = -np.cross(omega, p_on_axis)
        
        screw_axes.append((omega, v))
        
        # 更新累积变换
        T_accum = T_accum * (SE3.Rx(alpha) * SE3.Tx(a) * SE3.Tz(d) * SE3.Rz(theta))
    
    return screw_axes, M0


def main():
    """主函数"""
    print("=" * 60)
    print("POE方法机器人运动学校准示例")
    print("=" * 60)
    
    # 定义真实DH参数（用于生成POE参数）
    true_dh_params = [
        (0.3, 0.0, 0.0, 0.0),   # 关节1: a=300mm
        (0.25, 0.0, 0.0, 0.0),  # 关节2: a=250mm
    ]
    
    # 将DH参数转换为POE参数
    true_screw_axes, M0_true = dh_to_poe(true_dh_params)
    
    # 定义初始估计（添加误差）
    initial_dh_params = [
        (0.29, 0.0, 0.0, 0.0),   # 关节1: 误差-10mm
        (0.24, 0.0, 0.0, 0.0),   # 关节2: 误差+10mm
    ]
    initial_screw_axes, M0_initial = dh_to_poe(initial_dh_params)
    
    # 添加零位位姿误差
    M0_initial = M0_initial * SE3.Tx(0.01) * SE3.Ty(0.01)  # 添加10mm位置误差
    
    # 相机参数（真实值）
    camera_params_true = {
        'd0': 0.1,
        'theta0': np.pi/6,
        'a1': 0.2,
        'd1': 0.05,
        'alpha1': 0.0,
        'theta1': 0.0
    }
    
    # 相机参数（初始估计，带误差）
    camera_params_initial = {
        'd0': 0.098,
        'theta0': np.pi/6 + 0.02,
        'a1': 0.198,
        'd1': 0.051,
        'alpha1': 0.005,
        'theta1': -0.005
    }
    
    # 创建校准对象
    calibrator = POECalibration(
        true_screw_axes=true_screw_axes,
        initial_screw_axes=initial_screw_axes,
        M0_true=M0_true,
        M0_initial=M0_initial,
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
    
    # 生成测量数据
    print("\n生成测量数据...")
    measured_positions = []
    for q in q_list:
        T_camera_end = calibrator.forward_kinematics(q, calibrator.phi_true)
        p_camera = np.array(T_camera_end.t).flatten()
        noise = np.random.normal(0, 0.001, 3)
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
    plt.title('POE方法校准误差收敛曲线', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    output_path = '/home/frank/extra_storage/Frank/doc/RL/springer_robotics_converted/poe_calibration_convergence.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n误差收敛曲线已保存到: poe_calibration_convergence.png")
    
    print("\n" + "=" * 60)
    print("POE方法校准完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

