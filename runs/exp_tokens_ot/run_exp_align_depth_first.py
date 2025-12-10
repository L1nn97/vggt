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

from evaluation.utils import save_config_to_json, save_evaluation_metrics, calc_aligned_depth

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
        target_size=(518, 336),
        # target_size=(252, 168),
        scan_id=4,
        num_views=4,
        view_step=1,
        use_conf_filter=False,
        conf_percentile=30.0,
        max_depth_diff=50.0,
        knockout_layer_idx=[],
        knockout_method="corres_mask",
        knockout_random_ratio=0.5,
        knockout_top_k=100,
        use_local_display=True,
        save_results=False,
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
        with torch.amp.autocast(cfg.device,dtype=cfg.dtype):
            current_memory = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            predictions = model(gt_images)
            additional_memory = torch.cuda.memory_allocated() - (current_memory + 1e9)
            peak_memory = torch.cuda.max_memory_allocated()
            additional_peak_memory = peak_memory - (current_memory + 1e9)

            print(f"Additional memory used: {additional_memory / (1024 ** 3)} GB")
            print(f"Additional peak memory used: {additional_peak_memory / (1024 ** 3)} GB")

            if cfg.save_results:
                with open(os.path.join(save_root, "memory_usage.txt"), "a") as f:
                    f.write(f"Additional memory used: {additional_memory / (1024 ** 3)} GB\n")
                    f.write(f"Additional peak memory used: {additional_peak_memory / (1024 ** 3)} GB\n")
                    f.write(f"Current memory: {current_memory / (1024 ** 3)} GB\n")
                    f.write(f"Peak memory: {peak_memory / (1024 ** 3)} GB\n")
    
    if len(token_weighter.token_cosine_similarity) > 0:
        plt.figure()
        token_similarity_image = np.concatenate(token_weighter.token_cosine_similarity, axis=0)
        plt.imshow(token_similarity_image)
        plt.title(f"token cosine similarity")
        plt.xlabel("Layer Index")
        plt.ylabel("token cosine similarity")
        plt.grid(False)
        if cfg.save_results:
            # 保存原图
            if token_similarity_image.dtype != np.uint8:
                if token_similarity_image.max() <= 1.0:
                    token_similarity_image = (token_similarity_image * 255).astype(np.uint8)
                else:
                    token_similarity_image = token_similarity_image.astype(np.uint8)
            # 确保是BGR格式（cv2.imwrite需要）
            if len(token_similarity_image.shape) == 3 and token_similarity_image.shape[2] == 3:
                token_similarity_image_bgr = cv2.cvtColor(token_similarity_image, cv2.COLOR_RGB2BGR)
            else:
                token_similarity_image_bgr = token_similarity_image
            cv2.imwrite(os.path.join(save_root, "token_cosine_similarity.png"), token_similarity_image_bgr)
        if cfg.use_local_display:
            plt.show()
        else:
            plt.close()

    if len(token_weighter.attention_of_all_heads) > 0:
        all_heads_attn_map = np.concatenate(token_weighter.attention_of_all_heads, axis=0)
        plt.figure()
        plt.imshow(all_heads_attn_map)
        plt.colorbar()
        plt.title(f"all heads attention map")
        plt.xlabel("Layer Index")
        plt.ylabel("all heads attention map")
        plt.grid(False)
        if cfg.save_results:
            # 保存原始注意力图数组
            if all_heads_attn_map.dtype != np.uint8:
                if all_heads_attn_map.max() <= 1.0:
                    all_heads_attn_map = (all_heads_attn_map * 255).astype(np.uint8)
                else:
                    all_heads_attn_map = all_heads_attn_map.astype(np.uint8)
            # 若为RGB则转BGR以符合cv2.imwrite
            if (
                len(all_heads_attn_map.shape) == 3
                and all_heads_attn_map.shape[2] == 3
            ):
                attn_map_bgr = cv2.cvtColor(all_heads_attn_map, cv2.COLOR_RGB2BGR)
            else:
                attn_map_bgr = all_heads_attn_map
            cv2.imwrite(
                os.path.join(save_root, "all_heads_attention_map.png"),
                attn_map_bgr,
            )
        if cfg.use_local_display:
            plt.show()
        else:
            plt.close()

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
    
    pairwise_metrics = compute_pairwise_relative_errors(
        poses_pred=convert_poses_to_4x4(pred_extrinsic),
        poses_gt=convert_poses_to_4x4(gt_extrinsics),
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

    aligned_pred_depths = np.array(aligned_pred_depths)
    # aligned_pred_depth = gt_depths.cpu().numpy()

    # pred_points_from_depth = unproject_depth_map_to_point_map(aligned_pred_depths, pred_extrinsic, pred_intrinsic)
    pred_points_from_depth = unproject_depth_map_to_point_map(aligned_pred_depths, gt_extrinsics, gt_intrinsics)
    pred_conf = depth_conf  # [S, H, W]

    pred_points_from_depth_flat = pred_points_from_depth.reshape(-1, 3)  
    pred_conf_flat = pred_conf.reshape(-1) 

    pred_points_from_depth_flat = pred_points_from_depth_flat @ pred_extrinsic[0][:3, :3].T + pred_extrinsic[0][:3, 3] 

    gt_pts_from_depth = unproject_depth_map_to_point_map(gt_depths.cpu().numpy(), gt_extrinsics, gt_intrinsics)
    gt_pts_from_depth_flat = gt_pts_from_depth.reshape(-1, 3)  
    # gt_pts_from_depth_flat = gt_pts_from_depth_flat @ gt_extrinsics[0][:3, :3].T + gt_extrinsics[0][:3, 3] 

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

        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(gt_pts_from_depth_flat)
        # pcd_gt.colors = o3d.utility.Vector3dVector(colors_flat.astype(np.float32) / 255.0)

        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(pts_flat_filtered)
        pcd_pred.colors = o3d.utility.Vector3dVector(colors_flat_filtered.astype(np.float32) / 255.0)

        geometries = [pcd_pred, pcd_gt]
        o3d.visualization.draw_geometries(geometries)