import os
import sys
import json
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any
from datetime import datetime

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

    # 计算差值（绝对值误差）
    depth_diff = np.abs(gt_depth_view - aligned_pred_depth)
    # 截断diff>max_depth_diff的值（视为异常值）
    diff_mask = depth_diff <= max_depth_diff
    # 只考虑mask有效区域，并且diff<=max_depth_diff
    valid_mask = gt_mask_view.astype(bool) & diff_mask
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


@dataclass
class VGGTReconstructConfig:
    """单场景重建与点云导出的配置。"""

    # 基本路径与设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    checkpoint_path: str = "/home/vision/ws/vggt/checkpoints/model.pt"

    output_dir: str = (
        "/home/vision/ws/vggt/runs/exp_tokens_ot/results/depth_estimation/"
    )

    # 模型相关配置
    patch_size: int = 14
    special_tokens_num: int = 5

    # dtu 数据集相关配置
    dtu_root: str = "/home/vision/ws/datasets/SampleSet/dtu_mvs"
    target_size: Tuple[int, int] = (518, 350)
    scan_id: int = 1
    num_views: int = 4
    view_step: int = 1

    # 置信度过滤配置（对最终导出的点云生效）
    # 是否根据置信度过滤点云， 百分位阈值（0~100），例如 50.0 表示保留大于等于中位数置信度的点
    use_conf_filter: bool = False
    conf_percentile: float = 30.0

    # 深度误差截断配置
    # 深度差值超过此阈值的点将被视为异常值并排除在统计之外
    max_depth_diff: float = 50.0

    # use local display
    use_local_display: bool = False

    # save results
    save_results: bool = False

    # knockout 配置
    knockout_layer_idx: List[int] = field(default_factory=lambda: [])
    knockout_method: str = "top_k"  # "random" or "visible_score" or "top_k"
    knockout_random_ratio: float = 0.1
    knockout_top_k: int = 100

#这一段代码是为了获取示例图像，用于测试###################################################
import glob
from vggt.utils.load_fn import load_and_preprocess_images

def get_example_images(path: str, num_images: int = 3, device: str = "cuda") -> torch.Tensor:
    """
    Loads up to num_images .png images from the given path, sorts them, 
    and returns a float32 torch tensor normalized to [0,1] with shape [S, 3, H, W].
    Accepts both [H,W,3] and [3,H,W] image files.
    """
    image_paths = glob.glob(os.path.join(path, "*.png"))
    if not image_paths:
        raise ValueError(f"No .png images found in directory: {path}")
    image_paths.sort()
    image_paths = image_paths[:num_images]

    return load_and_preprocess_images(image_paths).to(device)

# imgs = get_example_images("/home/vision/ws/vggt/examples/kitchen/images", cfg.num_images, device=cfg.device)

###################################################################################

if __name__ == "__main__":

    cfg = VGGTReconstructConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16,
        checkpoint_path="/home/vision/ws/vggt/checkpoints/model.pt",
        output_dir="/home/vision/ws/vggt/runs/exp_tokens_ot/results/",
        patch_size=14,
        special_tokens_num=5,
        dtu_root="/home/vision/ws/datasets/SampleSet/dtu_mvs",
        target_size=(518, 350),
        scan_id=1,
        num_views=4,
        view_step=1,
        use_conf_filter=False,
        conf_percentile=30.0,
        max_depth_diff=50.0,
        use_local_display=False,
        save_results=True,
        knockout_layer_idx=[-1],
        knockout_method="top_k",
        knockout_random_ratio=0.5,
        knockout_top_k=100
    )

    token_fusion_strategy_cfg = TokenFusionStrategyConfig(
        dtu_mvs_root=cfg.dtu_root,
        scan_id=cfg.scan_id,
        num_views=cfg.num_views,
        step=cfg.view_step,
        device=cfg.device,
        target_size=cfg.target_size,
        patch_size=cfg.patch_size,
        special_tokens_num=cfg.special_tokens_num,
        knockout_layer_idx=cfg.knockout_layer_idx,
        knockout_method=cfg.knockout_method,
        knockout_random_ratio=cfg.knockout_random_ratio,
        knockout_top_k=cfg.knockout_top_k,
    )

    token_weighter = TokenFusionStrategy(token_fusion_strategy_cfg)

    # 添加时间戳到输出目录（用于 attn_map_save_dir）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_root = os.path.join(cfg.output_dir, timestamp)

    if cfg.save_results:
        os.makedirs(save_root, exist_ok=True)
        print(f"Create output directory: {save_root}")
        save_config_to_json(
            cfg, 
            os.path.join(save_root, "cfg.json"),
            additional_configs={"token_fusion_strategy_cfg": token_fusion_strategy_cfg}
        )

    model = VGGT(token_weighter=token_weighter)
    model.load_state_dict(torch.load(cfg.checkpoint_path))
    model.to(cfg.device)
    model.eval()

    gts = []  # 初始化 gts 列表
    loader = DTUScanLoader(
        cfg.dtu_root,
        scan_id=cfg.scan_id,
        num_views=cfg.num_views,
        step=cfg.view_step,
        device=cfg.device,
        target_size=cfg.target_size,
    )
    gt_images = loader.load_images()
    gt_depths = loader.load_depths()  # List[np.ndarray] 或类似格式
    gt_masks = loader.load_masks()  # List[np.ndarray] 或类似格式
    gt_intrinsics, gt_extrinsics = loader.load_cameras()  # (S, 3, 3), (S, 4, 4)

    # print("gt_images.shape: ", gt_images.shape)
    # print("gt_depths.shape: ", gt_depths.shape)
    # print("gt_masks.shape: ", gt_masks.shape)
    # print("gt_intrinsics.shape: ", gt_intrinsics.shape)
    # print("gt_extrinsics.shape: ", gt_extrinsics.shape)

    with torch.no_grad():
        with torch.amp.autocast('cuda',dtype=cfg.dtype):
            predictions = model(gt_images)

    if len(token_weighter.tokens_erank_kernel_norm)> 0:
        plt.figure()
        plt.plot(token_weighter.tokens_erank_kernel_norm, marker='o')
        plt.title(f"tokens erank kernel norm Over Layers")
        plt.xlabel("Layer Index")
        plt.ylabel("tokens erank kernel norm")
        plt.grid(True)
        if cfg.save_results:
            plt.savefig(os.path.join(save_root, "tokens_erank_kernel_norm.png"))
        if cfg.use_local_display:
            plt.show()
        else:
            plt.close()
        
    if len(token_weighter.tokens_erank_fro_norm) > 0:
        plt.figure()
        plt.plot(token_weighter.tokens_erank_fro_norm, marker='x')
        plt.title(f"tokens erank fro norm Over Layers")
        plt.xlabel("Layer Index")
        plt.ylabel("tokens erank fro norm")
        plt.grid(True)
        if cfg.save_results:
            plt.savefig(os.path.join(save_root, "tokens_erank_fro_norm.png"))
        if cfg.use_local_display:
            plt.show()
        else:
            plt.close()
    
    if len(token_weighter.x_cos_similarity) > 0:
        plt.figure()
        plt.plot(token_weighter.x_cos_similarity, marker='o')
        plt.title(f"token cos similarity Over Layers")
        plt.xlabel("Layer Index")
        plt.ylabel("token cos similarity")
        plt.grid(True)
        if cfg.save_results:
            plt.savefig(os.path.join(save_root, "x_cos_similarity.png"))
        if cfg.use_local_display:
            plt.show()
        else:
            plt.close()

    if len(token_weighter.q_rope_gain) > 0 or len(token_weighter.top_k_dominance) > 0:
        plt.figure()
        if len(token_weighter.q_rope_gain) > 0:
            plt.plot(token_weighter.q_rope_gain, marker='o')
        if len(token_weighter.top_k_dominance) > 0:
            plt.plot(token_weighter.top_k_dominance, marker='x')
        plt.title(f"rope gain and top-k dominance Over Layers")
        plt.xlabel("Layer Index")
        plt.ylabel("ratio(rope gain / q_original & top-k num / num_keys)")
        plt.grid(True)
        if cfg.save_results:
            plt.savefig(os.path.join(save_root, "rope_gain_top_k_dominance.png"))
        if cfg.use_local_display:
            plt.show()
        else:
            plt.close()
    
    if len(token_weighter.top_k_dominance) > 0:
        plt.figure()
        layer_indices = range(len(token_weighter.top_k_dominance))
        plt.bar(layer_indices, token_weighter.top_k_dominance)
        plt.title(f"top-k dominance Over Layers")
        plt.xlabel("Layer Index")
        plt.ylabel("top-k dominance")
        plt.grid(True, axis='y')
        if cfg.save_results:
            plt.savefig(os.path.join(save_root, "top_k_dominance.png"))
        if cfg.use_local_display:
            plt.show()
        else:
            plt.close()


    # gt_images: [S, 3, H, W]，VGGT 内部会加 batch 维度 => pose_enc: [B, S, 9]
    pred_extrinsic, pred_intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], gt_images.shape[-2:]
    )
    
    depth = predictions["depth"].detach().cpu().numpy().reshape(gt_depths.shape)
    depth_conf = predictions["depth_conf"].detach().cpu().numpy().reshape(gt_depths.shape)
    pred_extrinsic = pred_extrinsic.detach().squeeze(0).cpu().numpy()  
    pred_intrinsic = pred_intrinsic.detach().squeeze(0).cpu().numpy()  

    # print("depth.shape: ", depth.shape)
    # print("depth_conf.shape: ", depth_conf.shape)
    # print("pred_extrinsic.shape: ", pred_extrinsic.shape)
    # print("pred_intrinsic.shape: ", pred_intrinsic.shape)

    aligned_pred_depths = []
    shifts = []
    scales = []
    stats_list = []
    for i in range(depth.shape[0]):
        scale, shift, aligned_pred_depth, stats_dict = calc_aligned_depth(
            pred_depth=depth[i],
            gt_depth=gt_depths[i],
            gt_mask=gt_masks[i],
            use_local_display=False,
            verbose=False,
            max_depth_diff=cfg.max_depth_diff,
        )
        aligned_pred_depths.append(aligned_pred_depth)
        shifts.append(shift)
        scales.append(scale)
        stats_list.append(stats_dict)
        # print(f"mean depth error: {stats_dict['mean']:.6f}")
        # rescale extrinsic and intrinsic
        pred_extrinsic[i][:, 3] = pred_extrinsic[i][:, 3] * scale 
    
    # 计算各视图的平均统计值
    total_valid_pixels = sum([s['valid_pixels'] for s in stats_list])
    total_pixels = sum([s['total_pixels'] for s in stats_list])
    
    # 计算加权平均（按有效像素数加权）
    valid_stats_list = [s for s in stats_list if not np.isnan(s['mean']) and s['valid_pixels'] > 0]
    
    # 加权平均（按有效像素数）
    weights = np.array([s['valid_pixels'] for s in valid_stats_list])
    weights = weights / weights.sum()
    
    weighted_stats = {
        "mean": np.average([s['mean'] for s in valid_stats_list], weights=weights),
        "std": np.average([s['std'] for s in valid_stats_list], weights=weights),
        "median": np.average([s['median'] for s in valid_stats_list], weights=weights),
        "min": np.min([s['min'] for s in valid_stats_list]),
        "max": np.max([s['max'] for s in valid_stats_list]),
        "q25": np.average([s['q25'] for s in valid_stats_list], weights=weights),
        "q75": np.average([s['q75'] for s in valid_stats_list], weights=weights),
        "q95": np.average([s['q95'] for s in valid_stats_list], weights=weights),
        "q99": np.average([s['q99'] for s in valid_stats_list], weights=weights),
    }   
    
    # 打印加权平均统计（按有效像素数加权）
    print("\n加权平均统计 (按有效像素数加权):")
    print(f"  均值 (Mean):           {weighted_stats['mean']:.6f}")
    print(f"  标准差 (Std):          {weighted_stats['std']:.6f}")
    print(f"  中位数 (Median):       {weighted_stats['median']:.6f}")
    print(f"  最小值 (Min):          {weighted_stats['min']:.6f}")
    print(f"  最大值 (Max):          {weighted_stats['max']:.6f}")
    print(f"  25%分位数 (Q25):       {weighted_stats['q25']:.6f}")
    print(f"  75%分位数 (Q75):       {weighted_stats['q75']:.6f}")
    print(f"  95%分位数 (Q95):       {weighted_stats['q95']:.6f}")
    print(f"  99%分位数 (Q99):       {weighted_stats['q99']:.6f}")
    print(f"  有效像素比例:          {total_valid_pixels / total_pixels * 100.0:.2f}%")
    
    # 打印每个视图的简要信息
    print("\n各视图详细统计:")
    for i, stats_dict in enumerate(stats_list):
        if not np.isnan(stats_dict['mean']):
            print(f"  视图 {i}: 均值={stats_dict['mean']:.6f}, "
                  f"中位数={stats_dict['median']:.6f}, "
                  f"有效像素={stats_dict['valid_pixels']} ({stats_dict['valid_ratio']:.2f}%)")
        else:
            print(f"  视图 {i}: 无有效数据")
    
    
    # 确保位姿格式一致（转换为 4x4 格式）
    pred_poses_4x4 = convert_poses_to_4x4(pred_extrinsic)
    gt_poses_4x4 = convert_poses_to_4x4(gt_extrinsics)
    
    pairwise_metrics = compute_pairwise_relative_errors(
        poses_pred=pred_poses_4x4,
        poses_gt=gt_poses_4x4,
        verbose=True,
    )
    
    # 保存所有统计结果到JSON文件
    if cfg.save_results:
        metrics_path = os.path.join(save_root, "evaluation_metrics.json")
        save_evaluation_metrics(
            weighted_stats=weighted_stats,
            stats_list=stats_list,
            total_valid_pixels=total_valid_pixels,
            total_pixels=total_pixels,
            valid_stats_list=valid_stats_list,
            pairwise_metrics=pairwise_metrics,
            save_path=metrics_path,
        )

    # if cfg.save_results:
    #     save_depth_estimation_results(aligned_pred_depths, pred_extrinsic, pred_intrinsic, gt_depths, gt_masks, stats_list, save_root)

    aligned_pred_depths = np.array(aligned_pred_depths)

    pred_points_from_depth = unproject_depth_map_to_point_map(aligned_pred_depths, pred_extrinsic, pred_intrinsic)
    pred_conf = depth_conf  # [S, H, W]

    pred_points_from_depth_flat = pred_points_from_depth.reshape(-1, 3)  
    pred_conf_flat = pred_conf.reshape(-1) 

    pred_points_from_depth_flat = pred_points_from_depth_flat @ pred_extrinsic[0][:3, :3].T + pred_extrinsic[0][:3, 3] 

    gt_pts_from_depth = unproject_depth_map_to_point_map(gt_depths.cpu().numpy(), gt_extrinsics, gt_intrinsics)
    gt_pts_from_depth_flat = gt_pts_from_depth.reshape(-1, 3)  
    gt_pts_from_depth_flat = gt_pts_from_depth_flat @ gt_extrinsics[0][:3, :3].T + gt_extrinsics[0][:3, 3] 

    colors = gt_images.detach().cpu().permute(0, 2, 3, 1).numpy()
    colors = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
    colors_flat = colors.reshape(-1, 3)  

    # 按置信度过滤点云
    pts_flat_filtered = pred_points_from_depth_flat
    colors_flat_filtered = colors_flat
    if cfg.use_conf_filter:
        # 计算分位数阈值，例如 50% 分位数（中位数）
        conf_threshold = np.percentile(pred_conf_flat, cfg.conf_percentile)
        print(f"Conf threshold: {conf_threshold}")
        mask = pred_conf_flat >= conf_threshold
        print(f"Max conf: {pred_conf_flat.max()}, Min conf: {pred_conf_flat.min()}")

        pts_flat_filtered = pred_points_from_depth_flat[mask]
        colors_flat_filtered = colors_flat[mask]

    if cfg.save_results:
        # save_point_cloud_to_ply(pred_points_from_depth_flat, colors_flat, os.path.join(save_root, "pred_points.ply"))
        # save_point_cloud_to_ply(gt_pts_from_depth_flat, colors_flat, os.path.join(save_root, "gt_points.ply"))
        save_config_to_json(
            cfg, 
            os.path.join(save_root, "cfg.json"),
            additional_configs={"token_fusion_strategy_cfg": token_fusion_strategy_cfg}
        )


    if cfg.use_local_display:

        import open3d as o3d

        # pcd_gt = o3d.geometry.PointCloud()
        # pcd_gt.points = o3d.utility.Vector3dVector(gt_pts_from_depth_flat)
        # pcd_gt.colors = o3d.utility.Vector3dVector(colors_flat.astype(np.float32) / 255.0)

        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(pts_flat_filtered)
        pcd_pred.colors = o3d.utility.Vector3dVector(colors_flat_filtered.astype(np.float32) / 255.0)

        geometries = [pcd_pred]
        o3d.visualization.draw_geometries(geometries)