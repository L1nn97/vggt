import os
import sys
import json
import open3d as o3d
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any
from datetime import datetime
import time

import cv2
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt

np.set_printoptions(threshold = np.inf) 
np.set_printoptions(suppress = True)

from data.dtu_loader import DTUScanLoader
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.layers.token_weighter import TokenFusionStrategy, TokenFusionStrategyConfig
from eval.criterion import Regr3D_t_ScaleShiftInv, L21
from vggt.utils.geometry import depth_to_world_coords_points, depth_to_cam_coords_points
from evaluation.depth_estimation import calculate_depth_error, align_pred_to_gt
from evaluation.pose_estimation import compute_pairwise_relative_errors, convert_poses_to_4x4
import json


def calc_aligned_depth(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    gt_mask: np.ndarray,
    use_local_display: bool = True,
    verbose: bool = False,
    max_depth_diff: float = 50.0,
):
    """
    对齐预测深度与GT深度，计算误差统计，并可视化结果。

    Args:
        pred_depth: 预测深度图，形状为 [H, W] 或 [H, W, 1]
        gt_depth: GT深度图，可以是numpy数组或tensor
        gt_mask: GT mask，可以是numpy数组或tensor
        use_local_display: 是否本地显示图像
        verbose: 是否打印详细信息

    Returns:
        tuple: (scale, shift, aligned_pred_depth, stats_dict) 
            - scale: 对齐缩放因子
            - shift: 对齐偏移量
            - aligned_pred_depth: 对齐后的预测深度
            - stats_dict: 深度误差统计字典，包含以下键：
                - mean: 均值
                - std: 标准差
                - median: 中位数
                - min: 最小值
                - max: 最大值
                - q25: 25%分位数
                - q75: 75%分位数
                - q95: 95%分位数
                - q99: 99%分位数
                - valid_pixels: 有效像素数
                - total_pixels: 总像素数
                - valid_ratio: 有效像素比例（百分比）
    """
    # 转换为numpy数组
    if isinstance(gt_depth, torch.Tensor):
        gt_depth = gt_depth.cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()

    # 对齐预测深度到GT
    scale, shift, aligned_pred_depth = align_pred_to_gt(
        pred_depth, gt_depth, gt_mask, 100
    )

    # 打印基本信息（仅在verbose模式）
    if verbose:
        print(f"{'Scale':<15}: {scale:.6f}")
        print(f"{'Shift':<15}: {shift:.6f}")

    # 处理维度，确保都是 [H, W]
    gt_depth_view = gt_depth.copy()
    gt_mask_view = gt_mask.copy()

    if gt_depth_view.ndim > 2:
        gt_depth_view = gt_depth_view.squeeze()
    if aligned_pred_depth.ndim > 2:
        aligned_pred_depth = aligned_pred_depth.squeeze()
    if gt_mask_view.ndim > 2:
        gt_mask_view = gt_mask_view.squeeze()

    valid_mask = gt_mask_view.astype(bool) 

    depth_diff = np.abs(gt_depth_view - aligned_pred_depth)
    
    diff_mask = depth_diff <= max_depth_diff
    valid_mask = valid_mask & diff_mask

    depth_diff_masked = np.where(valid_mask, depth_diff, np.nan)

    # 计算差值的统计数据
    valid_diff = depth_diff[valid_mask] if np.any(valid_mask) else np.array([])
    
    # 构建统计信息字典
    stats_dict = {}
    if len(valid_diff) > 0:
        stats_dict = {
            "mean": float(np.mean(valid_diff)),
            "std": float(np.std(valid_diff)),
            "median": float(np.median(valid_diff)),
            "min": float(np.min(valid_diff)),
            "max": float(np.max(valid_diff)),
            "q25": float(np.percentile(valid_diff, 25)),
            "q75": float(np.percentile(valid_diff, 75)),
            "q95": float(np.percentile(valid_diff, 95)),
            "q99": float(np.percentile(valid_diff, 99)),
            "valid_pixels": int(len(valid_diff)),
            "total_pixels": int(valid_mask.size),
            "valid_ratio": float(len(valid_diff) / valid_mask.size * 100.0),
        }
    else:
        # 如果没有有效像素，返回空值
        stats_dict = {
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "q95": np.nan,
            "q99": np.nan,
            "valid_pixels": 0,
            "total_pixels": int(valid_mask.size),
            "valid_ratio": 0.0,
        }
    
    if verbose:
        if len(valid_diff) > 0:
            print("\n" + "=" * 60)
            print("深度误差统计信息 (仅在有效mask区域):")
            print("=" * 60)
            print(f"  均值 (Mean):           {stats_dict['mean']:.6f}")
            print(f"  标准差 (Std):          {stats_dict['std']:.6f}")
            print(f"  中位数 (Median):       {stats_dict['median']:.6f}")
            print(f"  最小值 (Min):          {stats_dict['min']:.6f}")
            print(f"  最大值 (Max):          {stats_dict['max']:.6f}")
            print(f"  25%分位数 (Q25):       {stats_dict['q25']:.6f}")
            print(f"  75%分位数 (Q75):       {stats_dict['q75']:.6f}")
            print(f"  95%分位数 (Q95):       {stats_dict['q95']:.6f}")
            print(f"  99%分位数 (Q99):       {stats_dict['q99']:.6f}")
            print(f"  有效像素数:            {stats_dict['valid_pixels']}")
            print(f"  总像素数:              {stats_dict['total_pixels']}")
            print(f"  有效像素比例:          {stats_dict['valid_ratio']:.2f}%")
            print("=" * 60 + "\n")
        else:
            print("警告: 没有有效的mask区域用于计算误差统计")

    # 水平拼接三个深度图: GT, 对齐后的预测, 差值
    combined_depth = np.concatenate(
        [gt_depth_view, aligned_pred_depth, depth_diff_masked], axis=1
    )

    # 显示拼接后的深度图（由use_local_display控制）
    if use_local_display:
        plt.figure(figsize=(24, 8))
        im = plt.imshow(combined_depth, cmap="viridis")
        plt.colorbar(im, label="Depth / Error")
        plt.title(
            "Left: GT Depth | Middle: Aligned Predicted Depth | Right: Absolute Error"
        )
        plt.axvline(
            x=gt_depth_view.shape[1],
            color="red",
            linestyle="--",
            linewidth=2,
            label="Split line 1",
        )
        plt.axvline(
            x=gt_depth_view.shape[1] * 2,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Split line 2",
        )
        plt.legend()
        plt.show()
        plt.close()

    # 绘制误差统计直方图（由use_local_display控制）
    if len(valid_diff) > 0 and use_local_display:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 左图: 线性尺度直方图
        axes[0].hist(
            valid_diff, bins=100, edgecolor="black", alpha=0.7, color="skyblue"
        )
        axes[0].axvline(
            np.mean(valid_diff),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(valid_diff):.6f}",
        )
        axes[0].axvline(
            np.median(valid_diff),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {np.median(valid_diff):.6f}",
        )
        axes[0].set_xlabel("Absolute Depth Error", fontsize=12)
        axes[0].set_ylabel("Frequency", fontsize=12)
        axes[0].set_title("Depth Error Histogram (Linear Scale)", fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 右图: 对数尺度直方图
        axes[1].hist(
            valid_diff, bins=100, edgecolor="black", alpha=0.7, color="lightcoral"
        )
        axes[1].axvline(
            np.mean(valid_diff),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(valid_diff):.6f}",
        )
        axes[1].axvline(
            np.median(valid_diff),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {np.median(valid_diff):.6f}",
        )
        axes[1].set_xlabel("Absolute Depth Error", fontsize=12)
        axes[1].set_ylabel("Frequency (Log Scale)", fontsize=12)
        axes[1].set_title("Depth Error Histogram (Log Scale)", fontsize=14)
        axes[1].set_yscale("log")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        plt.close()

    return scale, shift, aligned_pred_depth, stats_dict

def save_point_cloud_to_ply(points: np.ndarray, colors: np.ndarray, ply_path: str) -> None:
    """
    将点云保存为 PLY 格式文件。

    Args:
        points: 点云坐标数组，形状为 (N, 3)，每个点为 (x, y, z)
        colors: 颜色数组，形状为 (N, 3)，每个点的 RGB 值，范围 0-255
        ply_path: 保存路径（包含文件名）
    
    Returns:
        None
    """
    num_vertices = points.shape[0]
    
    # 确保目录存在
    dir_path = os.path.dirname(ply_path)
    if dir_path:  # 如果目录路径不为空
        os.makedirs(dir_path, exist_ok=True)
    
    with open(ply_path, "w") as f:
        # PLY 头
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # 写每个点
        for (x, y, z), (r, g, b) in zip(points, colors):
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")

    print(f"PLY point cloud saved to {ply_path}")


def save_evaluation_metrics(
    weighted_stats: Dict[str, float],
    stats_list: List[Dict[str, Any]],
    total_valid_pixels: int,
    total_pixels: int,
    valid_stats_list: List[Dict[str, Any]],
    pairwise_metrics: Dict[str, np.ndarray],
    save_path: str,
) -> None:
    """
    保存深度估计和位姿误差的评估指标到JSON文件。
    
    Args:
        weighted_stats: 加权平均统计信息
        stats_list: 每个视图的统计信息列表
        total_valid_pixels: 总有效像素数
        total_pixels: 总像素数
        valid_stats_list: 有效视图的统计信息列表
        pairwise_metrics: 位姿误差指标（包含rra和rta）
        save_path: 保存路径
    """
    # 准备深度估计统计数据
    depth_stats_summary = {
        "weighted_average": {
            "mean": float(weighted_stats['mean']),
            "std": float(weighted_stats['std']),
            "median": float(weighted_stats['median']),
            "min": float(weighted_stats['min']),
            "max": float(weighted_stats['max']),
            "q25": float(weighted_stats['q25']),
            "q75": float(weighted_stats['q75']),
            "q95": float(weighted_stats['q95']),
            "q99": float(weighted_stats['q99']),
        },
        "pixel_statistics": {
            "total_valid_pixels": int(total_valid_pixels),
            "total_pixels": int(total_pixels),
            "valid_ratio_percent": float(total_valid_pixels / total_pixels * 100.0),
            "num_views": len(stats_list),
            "num_valid_views": len(valid_stats_list),
        },
        "per_view_statistics": [
            {
                "view_id": i,
                "mean": float(stats_dict['mean']) if not np.isnan(stats_dict['mean']) else None,
                "median": float(stats_dict['median']) if not np.isnan(stats_dict['median']) else None,
                "std": float(stats_dict['std']) if not np.isnan(stats_dict['std']) else None,
                "min": float(stats_dict['min']) if not np.isnan(stats_dict['min']) else None,
                "max": float(stats_dict['max']) if not np.isnan(stats_dict['max']) else None,
                "q25": float(stats_dict['q25']) if not np.isnan(stats_dict['q25']) else None,
                "q75": float(stats_dict['q75']) if not np.isnan(stats_dict['q75']) else None,
                "q95": float(stats_dict['q95']) if not np.isnan(stats_dict['q95']) else None,
                "q99": float(stats_dict['q99']) if not np.isnan(stats_dict['q99']) else None,
                "valid_pixels": int(stats_dict['valid_pixels']),
                "total_pixels": int(stats_dict['total_pixels']),
                "valid_ratio_percent": float(stats_dict['valid_ratio']),
            }
            for i, stats_dict in enumerate(stats_list)
        ],
    }
    
    # 准备位姿误差统计数据
    pose_stats_summary = {
        "relative_rotation_angle_error": {
            "mean": float(np.mean(pairwise_metrics["rra"])),
            "std": float(np.std(pairwise_metrics["rra"])),
            "median": float(np.median(pairwise_metrics["rra"])),
            "min": float(np.min(pairwise_metrics["rra"])),
            "max": float(np.max(pairwise_metrics["rra"])),
        },
        "relative_translation_amount_error": {
            "mean": float(np.mean(pairwise_metrics["rta"])),
            "std": float(np.std(pairwise_metrics["rta"])),
            "median": float(np.median(pairwise_metrics["rta"])),
            "min": float(np.min(pairwise_metrics["rta"])),
            "max": float(np.max(pairwise_metrics["rta"])),
        },
        "pairwise_errors": {
            "rra_errors": pairwise_metrics["rra"].tolist(),
            "rta_errors": pairwise_metrics["rta"].tolist(),
        },
    }
    
    # 合并所有统计数据
    all_metrics = {
        "depth_estimation": depth_stats_summary,
        "pose_estimation": pose_stats_summary,
    }
    
    # 保存到JSON文件
    with open(save_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n所有评估指标已保存到: {save_path}")


def save_config_to_json(cfg: Any, json_path: str, additional_configs: Dict[str, Any] = None) -> None:
    """
    将配置对象保存为 JSON 文件。

    Args:
        cfg: 配置对象（通常是 dataclass 实例）
        json_path: 保存路径（包含文件名）
        additional_configs: 额外的配置字典，会合并到保存的配置中
    
    Returns:
        None
    """
    def convert_to_json_serializable(obj):
        """递归转换对象为 JSON 可序列化的类型"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.dtype):
            return str(obj)
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, (list, set)):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_json_serializable(value) for key, value in obj.items()}
        elif hasattr(obj, '__dict__'):
            # 处理 dataclass 或其他有 __dict__ 的对象
            if hasattr(obj, '__dataclass_fields__'):
                # dataclass
                result = {}
                for field_name in obj.__dataclass_fields__:
                    value = getattr(obj, field_name)
                    result[field_name] = convert_to_json_serializable(value)
                return result
            else:
                # 普通对象
                return {key: convert_to_json_serializable(value) for key, value in obj.__dict__.items()}
        else:
            return obj
    
    # 转换配置对象为字典
    if hasattr(cfg, '__dataclass_fields__'):
        # dataclass
        data = {}
        for field_name in cfg.__dataclass_fields__:
            value = getattr(cfg, field_name)
            data[field_name] = convert_to_json_serializable(value)
    elif isinstance(cfg, dict):
        data = convert_to_json_serializable(cfg)
    else:
        # 尝试转换为字典
        data = convert_to_json_serializable(cfg.__dict__ if hasattr(cfg, '__dict__') else cfg)
    
    # 合并额外配置
    if additional_configs:
        additional_data = convert_to_json_serializable(additional_configs)
        data.update(additional_data)
    
    # 确保目录存在
    dir_path = os.path.dirname(json_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    # 保存为 JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Configuration saved to {json_path}")

def compute_chamfer_distance(points_pred, points_gt, max_dist=1.0, display_outliner = False):
    # Filter out NaN and Inf values from point clouds
    valid_mask_pred = np.isfinite(points_pred).all(axis=1)
    valid_mask_gt = np.isfinite(points_gt).all(axis=1)
    
    nan_count_pred = np.sum(~valid_mask_pred)
    nan_count_gt = np.sum(~valid_mask_gt)
    
    if nan_count_pred > 0:
        print(f"[compute_chamfer_distance] WARNING: Found {nan_count_pred} NaN/Inf points in pred, filtering them out")
        points_pred = points_pred[valid_mask_pred]
    
    if nan_count_gt > 0:
        print(f"[compute_chamfer_distance] WARNING: Found {nan_count_gt} NaN/Inf points in gt, filtering them out")
        points_gt = points_gt[valid_mask_gt]
    
    if len(points_pred) == 0:
        print(f"[compute_chamfer_distance] ERROR: No valid points in pred after filtering!")
        return np.nan
    
    if len(points_gt) == 0:
        print(f"[compute_chamfer_distance] ERROR: No valid points in gt after filtering!")
        return np.nan
    
    # Ensure point cloud size is not too large, which would cause slow computation
    MAX_POINTS = 100000
    if points_pred.shape[0] > MAX_POINTS:
        np.random.seed(33)  # Fix random seed
        indices = np.random.choice(points_pred.shape[0], MAX_POINTS, replace=False)
        points_pred = points_pred[indices]

    if points_gt.shape[0] > MAX_POINTS:
        np.random.seed(33)  # Fix random seed
        indices = np.random.choice(points_gt.shape[0], MAX_POINTS, replace=False)
        points_gt = points_gt[indices]

    # Convert numpy point clouds to open3d point cloud objects
    pcd_pred = o3d.geometry.PointCloud()
    pcd_gt = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(points_pred)
    pcd_gt.points = o3d.utility.Vector3dVector(points_gt)

    # Downsample point clouds to accelerate computation
    voxel_size = 0.05  # 5cm voxel size
    pcd_pred = pcd_pred.voxel_down_sample(voxel_size)
    pcd_gt = pcd_gt.voxel_down_sample(voxel_size)

    # Compute distances from predicted point cloud to GT point cloud
    distances1 = np.asarray(pcd_pred.compute_point_cloud_distance(pcd_gt))
    # Compute distances from GT point cloud to predicted point cloud
    distances2 = np.asarray(pcd_gt.compute_point_cloud_distance(pcd_pred))
    
    # Check for NaN in computed distances
    nan_dist1 = np.sum(~np.isfinite(distances1))
    nan_dist2 = np.sum(~np.isfinite(distances2))
    if nan_dist1 > 0:
        print(f"[compute_chamfer_distance] WARNING: Found {nan_dist1} NaN/Inf in distances1, replacing with max_dist")
        distances1 = np.nan_to_num(distances1, nan=max_dist, posinf=max_dist, neginf=max_dist)
    if nan_dist2 > 0:
        print(f"[compute_chamfer_distance] WARNING: Found {nan_dist2} NaN/Inf in distances2, replacing with max_dist")
        distances2 = np.nan_to_num(distances2, nan=max_dist, posinf=max_dist, neginf=max_dist)
    
    # Statistics for distances1 (pred -> gt)
    mask1 = distances1 > max_dist
    count1 = np.sum(mask1)
    if count1 > 0:
        mean_error1 = np.mean(distances1[mask1])
        print(f"distances1 (pred->gt): {count1} points exceed max_dist={max_dist}, mean error: {mean_error1:.4f}")
    else:
        print(f"distances1 (pred->gt): all {len(distances1)} points are within max_dist={max_dist}")

    # Statistics for distances2 (gt -> pred)
    mask2 = distances2 > max_dist
    count2 = np.sum(mask2)
    if count2 > 0:
        mean_error2 = np.mean(distances2[mask2])
        print(f"distances2 (gt->pred): {count2} points exceed max_dist={max_dist}, mean error: {mean_error2:.4f}")
    else:
        print(f"distances2 (gt->pred): all {len(distances2)} points are within max_dist={max_dist}")

    if display_outliner:
        # 绘制大于误差限的点云（距离大于max_dist的点）
        # 可视化 pred->gt 误差大于max_dist的点（红色）以及全部点（灰色）
        if np.sum(mask1) > 0:
            # 大于max_dist的预测点
            error_points_pred = np.asarray(pcd_pred.points)[mask1]
            pcd_err_pred = o3d.geometry.PointCloud()
            pcd_err_pred.points = o3d.utility.Vector3dVector(error_points_pred)
            pcd_err_pred.paint_uniform_color([1, 0, 0])  # 红色

            # 其余全部预测点（浅灰）
            all_points_pred = np.asarray(pcd_pred.points)
            valid_points_pred = all_points_pred[~mask1]
            pcd_valid_pred = o3d.geometry.PointCloud()
            pcd_valid_pred.points = o3d.utility.Vector3dVector(valid_points_pred)
            pcd_valid_pred.paint_uniform_color([0.6, 0.6, 0.6])  # 灰色

            # GT点云
            pcd_gt.paint_uniform_color([0, 1, 0])  # 绿色

            o3d.visualization.draw_geometries([pcd_err_pred, pcd_valid_pred], window_name="预测点云超限(红)+GT(绿)")

        # 可视化 gt->pred 误差大于max_dist的GT点（蓝色）
        if np.sum(mask2) > 0:
            error_points_gt = np.asarray(pcd_gt.points)[mask2]
            pcd_err_gt = o3d.geometry.PointCloud()
            pcd_err_gt.points = o3d.utility.Vector3dVector(error_points_gt)
            pcd_err_gt.paint_uniform_color([0, 0, 1])  # 蓝色

            # 其余全部GT点（浅灰）
            all_points_gt = np.asarray(pcd_gt.points)
            valid_points_gt = all_points_gt[~mask2]
            pcd_valid_gt = o3d.geometry.PointCloud()
            pcd_valid_gt.points = o3d.utility.Vector3dVector(valid_points_gt)
            pcd_valid_gt.paint_uniform_color([0.6, 0.6, 0.6])  # 灰色

            # 预测点云
            pcd_pred.paint_uniform_color([1, 0.647, 0])  # 橙色

            o3d.visualization.draw_geometries([pcd_err_gt, pcd_valid_gt], window_name="GT点云超限(蓝)+预测点云(橙)")


    # Apply distance clipping
    distances1 = np.clip(distances1, 0, max_dist)
    distances2 = np.clip(distances2, 0, max_dist)

    # Chamfer Distance is the sum of mean distances in both directions
    chamfer_dist = (np.mean(distances1) + np.mean(distances2)) / 2.0

    print(f"Chamfer Distance: {np.mean(distances1)}, {np.mean(distances2)}, {chamfer_dist}")

    return chamfer_dist 

# Import umeyama_alignment for internal use in eval_trajectory
def umeyama_alignment(src_points, dst_points, estimate_scale=True):
    MAX_POINTS = 10000
    if src_points.shape[0] > MAX_POINTS:
        np.random.seed(33)  # Fix random seed
        indices = np.random.choice(src_points.shape[0], MAX_POINTS, replace=False)
        src = src_points[indices]

    if dst_points.shape[0] > MAX_POINTS:
        np.random.seed(33)  # Fix random seed
        indices = np.random.choice(dst_points.shape[0], MAX_POINTS, replace=False)
        dst = dst_points[indices]

    # Compute centroids
    src_mean = src.mean(axis=1, keepdims=True)
    dst_mean = dst.mean(axis=1, keepdims=True)

    # Center the point clouds
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    # Compute covariance matrix
    cov = dst_centered @ src_centered.T

    try:
        # Singular Value Decomposition
        U, D, Vt = svd(cov)
        V = Vt.T

        # Handle reflection case
        det_UV = np.linalg.det(U @ V.T)
        S = np.eye(3)
        if det_UV < 0:
            S[2, 2] = -1

        # Compute rotation matrix
        R = U @ S @ V.T

        if estimate_scale:
            # Compute scale factor - fix dimension issue
            src_var = np.sum(src_centered * src_centered)
            if src_var < 1e-10:
                print(
                    "Warning: Source point cloud variance close to zero, setting scale factor to 1.0"
                )
                scale = 1.0
            else:
                # Fix potential dimension issue with np.diag(S)
                # Use diagonal elements directly
                scale = np.sum(D * np.diag(S)) / src_var
        else:
            scale = 1.0

        # Compute translation vector
        t = dst_mean.ravel() - scale * (R @ src_mean).ravel()

        return scale, R, t

    except Exception as e:
        print(f"Error in umeyama_alignment computation: {e}")
        print(
            "Returning default transformation: scale=1.0, rotation=identity matrix, translation=centroid difference"
        )
        # Return default transformation
        scale = 1.0
        R = np.eye(3)
        t = (dst_mean - src_mean).ravel()
        return scale, R, 

def align_point_clouds_scale(source_pc, target_pc):
    # Compute bounding box sizes of point clouds
    source_min = np.min(source_pc, axis=0)
    source_max = np.max(source_pc, axis=0)
    target_min = np.min(target_pc, axis=0)
    target_max = np.max(target_pc, axis=0)

    source_size = source_max - source_min
    target_size = target_max - target_min

    # Compute point cloud centers
    source_center = (source_max + source_min) / 2
    target_center = (target_max + target_min) / 2

    # Compute overall scale factor (using diagonal length)
    source_diag = np.sqrt(np.sum(source_size**2))
    target_diag = np.sqrt(np.sum(target_size**2))

    if source_diag < 1e-8:
        print("Warning: Source point cloud size close to zero")
        scale_factor = 1.0
    else:
        scale_factor = target_diag / source_diag

    # Apply scaling (with source point cloud center as reference)
    centered_source = source_pc - source_center
    scaled_centered = centered_source * scale_factor
    scaled_aligned_source = scaled_centered + target_center

    return scaled_aligned_source, scale_factor