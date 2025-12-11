"""
位姿误差评估模块
计算预测位姿与真实位姿之间的误差指标
"""

import numpy as np
from scipy.spatial.transform import Rotation
from typing import Tuple, Dict, Optional


def rotation_matrix_to_angle(R: np.ndarray) -> float:
    """
    将旋转矩阵转换为旋转角度（弧度），然后转换为度数。
    
    Args:
        R: 3x3 旋转矩阵
        
    Returns:
        旋转角度（度数）
    """
    # 使用 scipy 的 Rotation 类
    r = Rotation.from_matrix(R)
    angle_rad = r.magnitude()  # 返回旋转角度（弧度）
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def compute_rotation_error(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """
    计算两个旋转矩阵之间的角度误差。
    
    Args:
        R_pred: 预测的旋转矩阵 (3, 3)
        R_gt: 真实旋转矩阵 (3, 3)
        
    Returns:
        旋转角度误差（度数）
    """
    # 计算相对旋转: R_error = R_gt^T * R_pred
    R_error = R_gt.T @ R_pred
    return rotation_matrix_to_angle(R_error)


def compute_translation_error(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    """
    计算两个平移向量之间的欧氏距离误差。
    
    Args:
        t_pred: 预测的平移向量 (3,)
        t_gt: 真实平移向量 (3,)
        
    Returns:
        平移误差（单位与输入相同）
    """
    return np.linalg.norm(t_pred - t_gt)


def extract_rotation_translation(pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    从位姿矩阵中提取旋转矩阵和平移向量。
    
    Args:
        pose: 位姿矩阵，可以是 (3, 4) 或 (4, 4) 格式
             格式为 [R|t]，其中 R 是旋转矩阵，t 是平移向量
        
    Returns:
        (R, t): 旋转矩阵 (3, 3) 和平移向量 (3,)
    """
    if pose.shape == (4, 4):
        R = pose[:3, :3]
        t = pose[:3, 3]
    elif pose.shape == (3, 4):
        R = pose[:3, :3]
        t = pose[:3, 3]
    else:
        raise ValueError(f"Unsupported pose shape: {pose.shape}")
    
    return R, t


def convert_poses_to_4x4(poses: np.ndarray) -> np.ndarray:
    """
    将位姿数组转换为统一的 4x4 格式。
    
    Args:
        poses: 位姿数组，可以是 (N, 3, 4) 或 (N, 4, 4) 格式
        
    Returns:
        转换后的位姿数组，形状 (N, 4, 4)
    """
    if not isinstance(poses, np.ndarray):
        poses = np.array(poses)
    
    N = len(poses)
    poses_4x4 = []
    
    for i in range(N):
        pose = poses[i]
        if pose.shape == (3, 4):
            # 转换为 4x4
            pose_4x4 = np.eye(4)
            pose_4x4[:3, :] = pose
        elif pose.shape == (4, 4):
            pose_4x4 = pose.copy()
        else:
            raise ValueError(f"Unsupported pose shape at index {i}: {pose.shape}")
        poses_4x4.append(pose_4x4)
    
    return np.array(poses_4x4)

def compute_pairwise_relative_errors(
    poses_pred: np.ndarray,
    poses_gt: np.ndarray,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """
    计算每对相邻位姿之间的相对旋转角度(RRA)和相对平移量(RTA)。
    
    Args:
        poses_pred: 预测位姿数组，形状 (N, 3, 4) 或 (N, 4, 4)
        poses_gt: 真实位姿数组，形状 (N, 3, 4) 或 (N, 4, 4)
        verbose: 是否打印详细信息
        
    Returns:
        包含每对位姿的RRA和RTA的字典:
            - rra: 相对旋转角度数组 (N-1,)，单位：度
            - rta: 相对平移量数组 (N-1,)，单位：与输入相同
            - rra_pred: 预测的相对旋转角度 (N-1,)
            - rta_pred: 预测的相对平移量 (N-1,)
            - rra_gt: 真实的相对旋转角度 (N-1,)
            - rta_gt: 真实的相对平移量 (N-1,)
    """
    # 确保输入是 numpy 数组
    if not isinstance(poses_pred, np.ndarray):
        poses_pred = np.array(poses_pred)
    if not isinstance(poses_gt, np.ndarray):
        poses_gt = np.array(poses_gt)
    
    # 检查形状
    assert len(poses_pred) == len(poses_gt), f"预测位姿数量 ({len(poses_pred)}) 与真实位姿数量 ({len(poses_gt)}) 不匹配"
    
    N = len(poses_pred)
    rra_errors = []  # 相对旋转角度误差
    rta_errors = []  # 相对平移量误差（余弦相似度）
    
    for i in range(N - 1):
        # 提取当前帧和下一帧的位姿
        pose_pred_i = poses_pred[i]
        pose_pred_j = poses_pred[i + 1]
        pose_gt_i = poses_gt[i]
        pose_gt_j = poses_gt[i + 1]
        
        # 提取旋转矩阵和平移向量
        R_pred_i, t_pred_i = extract_rotation_translation(pose_pred_i)
        R_pred_j, t_pred_j = extract_rotation_translation(pose_pred_j)
        R_gt_i, t_gt_i = extract_rotation_translation(pose_gt_i)
        R_gt_j, t_gt_j = extract_rotation_translation(pose_gt_j)
        
        # 计算相对位姿: pose_ij = pose_j * pose_i^{-1}
        # 对于旋转: R_ij = R_j * R_i^T
        # 对于平移: t_ij = t_j - R_ij * t_i
        R_pred_ij = R_pred_j @ R_pred_i.T
        # t_pred_ij = t_pred_j - R_pred_ij @ t_pred_i
        t_pred_ij = t_pred_j - t_pred_i
        
        R_gt_ij = R_gt_j @ R_gt_i.T
        # t_gt_ij = t_gt_j - R_gt_ij @ t_gt_i
        t_gt_ij = t_gt_j - t_gt_i
        
        # 计算相对旋转角度
        rra_pred = rotation_matrix_to_angle(R_pred_ij)
        rra_gt = rotation_matrix_to_angle(R_gt_ij)
        rra_error = abs(rra_pred - rra_gt)

        rta_error = t_pred_ij.T @ t_gt_ij / (np.linalg.norm(t_pred_ij) * np.linalg.norm(t_gt_ij))

        rra_errors.append(rra_error)
        rta_errors.append(rta_error)
    
    rra_errors = np.array(rra_errors)
    rta_errors = np.array(rta_errors)
    
    if verbose:
        print("\n每对位姿之间的相对误差:")
        print(f"\n相对旋转角度误差 (RRA Error):")
        print(f"  均值: {np.mean(rra_errors):.4f}°")
        print(f"  标准差: {np.std(rra_errors):.4f}°")
        print(f"  中位数: {np.median(rra_errors):.4f}°")
        print(f"  最大值: {np.max(rra_errors):.4f}°")
        print(f"  最小值: {np.min(rra_errors):.4f}°")
        
        print(f"\n相对平移量误差 (RTA Error):")
        print(f"  均值: {np.mean(rta_errors):.6f}")
        print(f"  标准差: {np.std(rta_errors):.6f}")
        print(f"  中位数: {np.median(rta_errors):.6f}")
        print(f"  最大值: {np.max(rta_errors):.6f}")
        print(f"  最小值: {np.min(rta_errors):.6f}")
        
    
    return {
        "rra": rra_errors,  # 相对旋转角度误差
        "rta": rta_errors,  # 相对平移量误差（余弦相似度）
    }