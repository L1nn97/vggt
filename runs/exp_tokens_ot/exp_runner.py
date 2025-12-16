import os
import sys
import json
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any
from datetime import datetime

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

from data.dtu_loader import DTUScanLoader
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.layers.token_weighter import TokenFusionStrategy, TokenFusionStrategyConfig
from evaluation.depth_estimation import align_pred_gt_per_depth, align_pred_to_gt_without_shift

from evaluation.pose_estimation import (
    compute_pairwise_relative_errors,
    convert_poses_to_4x4,
)

from evaluation.pointmap_estimation import umeyama_alignment
import json

from evaluation.utils import *


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

    # use local display
    use_local_display: bool = False

    # save results
    save_results: bool = False

    # knockout 配置
    knockout_layer_idx: List[int] = field(default_factory=lambda: [])
    knockout_method: str = "top_k"  # "random" or "visible_score" or "top_k"
    knockout_random_ratio: float = 0.1
    knockout_top_k: int = 100

    # token merge 配置
    enable_token_merge: bool = False
    token_merge_ratio: float = 0.2
    sx: int = 5
    sy: int = 5
    no_rand: bool = False
    enable_protection: bool = True

    # debug options
    calculate_rope_gain_ratio: bool = False
    calculate_token_cos_similarity: bool = False
    display_attn_map_after_softmax: bool = False
    calculate_top_k_dominance: bool = False

    # evaluation options
    use_stat_filter: bool = False
    use_icp_alignment: bool = False

# 这一段代码是为了获取示例图像，用于测试###################################################
import glob
from vggt.utils.load_fn import load_and_preprocess_images


def get_example_images(
    path: str, num_images: int = 3, device: str = "cuda"
) -> torch.Tensor:
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
    
    import argparse
    import sys

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        
    parser = argparse.ArgumentParser(description="VGGT Reconstruction Experimental Runner")

    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--dtype', type=str, default="torch.float16", help="torch.float16, torch.bfloat16, etc")
    
    parser.add_argument('--checkpoint_path', type=str, default="/home/vision/ws/vggt/checkpoints/model.pt")
    parser.add_argument('--output_dir', type=str, default="/home/vision/ws/vggt/runs/exp_tokens_ot/results/")
    
    parser.add_argument('--patch_size', type=int, default=14)
    parser.add_argument('--special_tokens_num', type=int, default=5)
    parser.add_argument('--target_size', type=int, nargs=2, default=[518,336])
    # parser.add_argument('--target_size', type=int, nargs=2, default=[252,168])
    
    parser.add_argument('--dtu_root', type=str, default="/home/vision/ws/datasets/SampleSet/dtu_mvs")
    parser.add_argument('--scan_id', type=int, default=4)
    parser.add_argument('--num_views', type=int, default=4)
    parser.add_argument('--view_step', type=int, default=1)
    
    parser.add_argument('--use_conf_filter', type=str2bool, default=False)
    parser.add_argument('--conf_percentile', type=float, default=30.0)

    parser.add_argument('--knockout_layer_idx', type=int, nargs='*', default=[], 
                        help='List of layer indices for knockout (e.g., --knockout_layer_idx 4 11 17 23)')
    parser.add_argument('--knockout_method', type=str, default="corres_mask")
    parser.add_argument('--knockout_random_ratio', type=float, default=0.5)
    parser.add_argument('--knockout_top_k', type=int, default=100)

    parser.add_argument('--enable_token_merge', type=str2bool, default=False)
    parser.add_argument('--token_merge_ratio', type=float, default=0.5)
    parser.add_argument('--sx', type=int, default=5)
    parser.add_argument('--sy', type=int, default=5)
    parser.add_argument('--no_rand', type=str2bool, default=False)
    parser.add_argument('--enable_protection', type=str2bool, default=True)

    parser.add_argument('--calculate_rope_gain_ratio', type=str2bool, default=False)
    parser.add_argument('--calculate_token_cos_similarity', type=str2bool, default=False)
    parser.add_argument('--display_attn_map_after_softmax', type=str2bool, default=False)
    parser.add_argument('--calculate_top_k_dominance', type=str2bool, default=False)
    parser.add_argument('--use_local_display', type=str2bool, default=False)
    parser.add_argument('--save_results', type=str2bool, default=False)

    parser.add_argument('--use_stat_filter', type=str2bool, default=False)
    parser.add_argument('--use_icp_alignment', type=str2bool, default=False)

    args = parser.parse_args()

    # Map args to config, handling type conversions if necessary
    dtype = getattr(torch, args.dtype.replace("torch.", "")) if hasattr(torch, args.dtype.replace("torch.", "")) else torch.float16

    # Compose configuration with parsed arguments (those that are set)
    cfg = VGGTReconstructConfig(
        device=args.device,
        dtype=dtype,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        special_tokens_num=args.special_tokens_num,
        dtu_root=args.dtu_root,
        target_size=tuple(args.target_size),
        scan_id=args.scan_id,
        num_views=args.num_views,
        view_step=args.view_step,
        use_conf_filter=args.use_conf_filter,
        conf_percentile=args.conf_percentile,
        knockout_layer_idx=args.knockout_layer_idx or [],
        knockout_method=args.knockout_method,
        knockout_random_ratio=args.knockout_random_ratio,
        knockout_top_k=args.knockout_top_k,
        enable_token_merge=args.enable_token_merge,
        token_merge_ratio=args.token_merge_ratio,
        sx=args.sx,
        sy=args.sy,
        no_rand=args.no_rand,
        enable_protection=args.enable_protection,
        calculate_rope_gain_ratio=args.calculate_rope_gain_ratio,
        calculate_token_cos_similarity=args.calculate_token_cos_similarity,
        display_attn_map_after_softmax=args.display_attn_map_after_softmax,
        calculate_top_k_dominance=args.calculate_top_k_dominance,
        use_local_display=args.use_local_display,
        save_results=args.save_results,
        use_stat_filter=args.use_stat_filter,
        use_icp_alignment=args.use_icp_alignment,
    )
    
    from pprint import pprint
    print("Configuration (cfg):")
    pprint(vars(cfg))
    
    loader = DTUScanLoader(
        cfg.dtu_root,
        scan_id=cfg.scan_id,
        num_views=cfg.num_views,
        step=cfg.view_step,
        device=cfg.device,
        target_size=cfg.target_size,
    )

    token_fusion_strategy_cfg = TokenFusionStrategyConfig(
        loader=loader,
        device=cfg.device,
        target_size=cfg.target_size,
        patch_size=cfg.patch_size,
        special_tokens_num=cfg.special_tokens_num,
        knockout_layer_idx=cfg.knockout_layer_idx,
        knockout_method=cfg.knockout_method,
        knockout_random_ratio=cfg.knockout_random_ratio,
        knockout_top_k=cfg.knockout_top_k,

        # FastVGGT token merge options
        enable_token_merge=cfg.enable_token_merge,
        token_merge_ratio=cfg.token_merge_ratio,
        sx=cfg.sx,
        sy=cfg.sy,
        no_rand=cfg.no_rand,
        enable_protection=cfg.enable_protection,

        # debug options
        calculate_rope_gain_ratio=cfg.calculate_rope_gain_ratio,
        calculate_token_cos_similarity=cfg.calculate_token_cos_similarity,
        display_attn_map_after_softmax=cfg.display_attn_map_after_softmax,
        calculate_top_k_dominance=cfg.calculate_top_k_dominance,
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
            additional_configs={"token_fusion_strategy_cfg": token_fusion_strategy_cfg},
        )

    model = VGGT(
        enable_point = False,
        enable_track = False,
        token_weighter=token_weighter
    )
    model.load_state_dict(torch.load(cfg.checkpoint_path), strict=False)
    model.to(cfg.device)
    model.eval()

    gt_images = loader.load_images()
    gt_depths = loader.load_depths()  # List[np.ndarray] 或类似格式
    gt_points, gt_points_colors = loader.load_points()  # [N, 3]
    gt_points_np = gt_points.cpu().numpy()
    gt_points_colors_np = gt_points_colors.cpu().numpy()

    gt_masks = loader.load_masks()  # List[np.ndarray] 或类似格式
    gt_intrinsics, gt_extrinsics = loader.load_cameras()  # (S, 3, 3), (S, 4, 4)

    R = gt_extrinsics[0][:3, :3]   # (3, 3)
    t = gt_extrinsics[0][:3, 3]    # (3,)
    gt_points_world = gt_points_np @ R.T + t

    import GPUtil
    if torch.cuda.is_available():
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU {gpu.id}: {gpu.name}, total {gpu.memoryTotal}MB, used {gpu.memoryUsed}MB, free {gpu.memoryFree}MB")
    else:
        print(f"RAM usage: {psutil.virtual_memory().used/1024/1024:.2f} MB (of {psutil.virtual_memory().total/1024/1024:.2f} MB)")

    
    @torch.no_grad()
    @torch.amp.autocast("cuda", dtype=cfg.dtype)
    def run_model_with_memory_stats(model, images):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        mem_before = torch.cuda.memory_allocated()

        predictions = model(images)

        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated()
        peak_mem = torch.cuda.max_memory_allocated()

        additional_peak_mem = peak_mem - mem_before

        return predictions, mem_before, peak_mem, additional_peak_mem


    predictions, mem_before, peak_memory, additional_peak_memory = run_model_with_memory_stats(model, gt_images)

    print(f"Memory before inference: {mem_before / (1024 ** 3)} GB")
    print(f"Peak memory used: {peak_memory / (1024 ** 3)} GB")
    print(f"Additional peak memory used: {additional_peak_memory / (1024 ** 3)} GB")

    if len(token_weighter.token_cosine_similarity) > 0:
        plt.figure()
        token_similarity_image = np.concatenate(
            token_weighter.token_cosine_similarity, axis=0
        )
        plt.imshow(token_similarity_image)
        plt.title(f"token cosine similarity")
        plt.xlabel("Layer Index")
        plt.ylabel("token cosine similarity")
        plt.grid(False)
        if cfg.save_results:
            # 保存原图
            if token_similarity_image.dtype != np.uint8:
                if token_similarity_image.max() <= 1.0:
                    token_similarity_image = (token_similarity_image * 255).astype(
                        np.uint8
                    )
                else:
                    token_similarity_image = token_similarity_image.astype(np.uint8)
            # 确保是BGR格式（cv2.imwrite需要）
            if (
                len(token_similarity_image.shape) == 3
                and token_similarity_image.shape[2] == 3
            ):
                token_similarity_image_bgr = cv2.cvtColor(
                    token_similarity_image, cv2.COLOR_RGB2BGR
                )
            else:
                token_similarity_image_bgr = token_similarity_image
            cv2.imwrite(
                os.path.join(save_root, "token_cosine_similarity.png"),
                token_similarity_image_bgr,
            )
        if cfg.use_local_display:
            plt.show()
        else:
            plt.close()

    if len(token_weighter.attention_of_all_heads) > 0:
        all_heads_attn_map = np.concatenate(
            token_weighter.attention_of_all_heads, axis=0
        )
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
            if len(all_heads_attn_map.shape) == 3 and all_heads_attn_map.shape[2] == 3:
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

    if len(token_weighter.tokens_erank_kernel_norm) > 0:
        plt.figure()
        plt.plot(token_weighter.tokens_erank_kernel_norm, marker="o")
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
        plt.plot(token_weighter.tokens_erank_fro_norm, marker="x")
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
        plt.plot(token_weighter.x_cos_similarity, marker="o")
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
            plt.plot(token_weighter.q_rope_gain, marker="o")
        if len(token_weighter.top_k_dominance) > 0:
            plt.plot(token_weighter.top_k_dominance, marker="x")
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
        plt.grid(True, axis="y")
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

    depth = predictions["depth"].cpu().squeeze()
    depth_conf = predictions["depth_conf"].cpu().squeeze()

    # depth_conf_display = np.concatenate([i.numpy() for i in depth_conf])
    # plt.imshow(depth_conf_display)
    # plt.show()
    
    pred_extrinsic = pred_extrinsic.detach().squeeze(0).cpu().numpy()
    pred_intrinsic = pred_intrinsic.detach().squeeze(0).cpu().numpy()

    print(
        "########################calculate pairwise pose estimation error#########################"
    )
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
    print(
        "########################calculate pairwise pose estimation error done#########################"
    )

    from evaluation.point_cloud_fusion import fusion_point_cloud
    from evaluation.filter import filter_depth_by_conf, stat_filter, remove_Nan_Zero_Inf

    depth, conf_mask = filter_depth_by_conf(depth,
                                            depth_conf, 
                                            cfg.conf_percentile, 
                                            3.0,
                                            verbose=True)

    valid_mask = torch.logical_and(
        gt_depths.squeeze().cpu() > 1e-3,
        conf_mask,
    )

    # valid_mask_display = np.concatenate([i.squeeze().numpy() for i in valid_mask])
    # plt.imshow(valid_mask_display)
    # plt.show()

    from evaluation.depth_estimation import calculate_depth_scales
    scales_by_depth = calculate_depth_scales(depth.squeeze().cpu().numpy(), 
                                             gt_depths.squeeze().cpu().numpy(), 
                                             valid_mask.squeeze().cpu().numpy())
    mean_scale = np.mean(scales_by_depth)
    print(f"scales_by_depth: {scales_by_depth}")
    print(f"mean_scale: {mean_scale}")

    N = depth.shape[0]
    for i in range(N):
        depth[i] = depth[i] * scales_by_depth[i]
        pred_extrinsic[i][:3, 3] = pred_extrinsic[i][:3, 3] * scales_by_depth[i]

    fused_points, fused_colors = fusion_point_cloud(depth, 
                                                    pred_extrinsic, 
                                                    pred_intrinsic, 
                                                    gt_images, 
                                                    dist_thresh=7.0, 
                                                    num_consist=2, 
                                                    scale_factor=1.0)

    fused_points, fused_colors = remove_Nan_Zero_Inf(fused_points, fused_colors)
    gt_points_world, gt_points_colors_np = remove_Nan_Zero_Inf(gt_points_world, gt_points_colors_np)

    from evaluation.display import display_point_clouds
    display_point_clouds([fused_points], [fused_colors], title="transformed pred points")

    if cfg.use_stat_filter:
        fused_points, fused_colors, fused_ind = stat_filter(fused_points, fused_colors, nb_neighbors=20, std_ratio=2.0)
        gt_points_world, gt_points_colors_np, gt_ind = stat_filter(gt_points_world, gt_points_colors_np, nb_neighbors=20, std_ratio=2.0)

    from evaluation.registration import register_point_clouds_open3d_icp
    if cfg.use_icp_alignment:
        transformed_pred_points, transformation = register_point_clouds_open3d_icp(fused_points, gt_points_world)
    else:
        transformed_pred_points, transformation = fused_points, np.eye(4)

    search_best_scale = False
    if search_best_scale:
        all_chamfer_dists = []
        for i in range(-50, 10, 1):
            print(f"--------------------{i}--------------------")
            chamfer_dist = compute_chamfer_distance(
                np.asarray(pred_cloud.points) * (mean_scale + i*0.1), np.asarray(gt_cloud.points), 1.0,
            )
            all_chamfer_dists.append(chamfer_dist)

        plt.plot(all_chamfer_dists)
        plt.show()

        min_chamfer_dist = np.min(all_chamfer_dists)
        min_chamfer_dist_idx = np.argmin(all_chamfer_dists)
        print(f"min_chamfer_dist: {min_chamfer_dist}")
        print(f"min_chamfer_dist_idx: {min_chamfer_dist_idx}")

        pred_cloud.points = o3d.utility.Vector3dVector(np.asarray(pred_cloud.points) * (mean_scale + min_chamfer_dist_idx*0.1))

    chamfer_dist = compute_chamfer_distance(
        transformed_pred_points, gt_points_world, 2.0, True
    )
    print(f"chamfer_dist: {chamfer_dist}")


    display_point_clouds([transformed_pred_points, gt_points_world], [fused_colors, gt_points_colors_np], title="transformed pred points")

    sys.exit()
    """

    print(f"After valid_mask filter - gt_pts_filtered shape: {gt_pts_filtered.shape}")
    print(f"After valid_mask filter - pred_pts_filtered shape: {pred_pts_filtered.shape}")

    # 统计滤波去除离群点
    pcd_gt_filter, ind_gt = stat_filter(gt_pts_filtered, nb_neighbors=20, std_ratio=2.0)
    print(f"GT statistical filter: {len(gt_pts_filtered)} -> {len(ind_gt)} points")

    pcd_pred_filter, ind_pred = stat_filter(pred_pts_filtered, nb_neighbors=20, std_ratio=2.0)
    print(f"Pred statistical filter: {len(pred_pts_filtered)} -> {len(ind_pred)} points")

    # 取两个滤波结果的交集，确保点云对应关系
    ind_gt_set = set(ind_gt)
    ind_pred_set = set(ind_pred)
    ind_common = list(ind_gt_set & ind_pred_set)
    print(f"Common indices after both filters: {len(ind_common)} points")

    # 使用共同索引过滤点云
    gt_pts_masked = gt_pts_filtered[ind_common].T  # (K, 3) -> (3, K)
    pred_pts_masked = pred_pts_filtered[ind_common].T  # (K, 3) -> (3, K)
    colors_flat_aligned = colors_flat_filtered[ind_common]  # (K, 3)

    print(f"Final gt_pts_masked shape: {gt_pts_masked.shape}")
    print(f"Final pred_pts_masked shape: {pred_pts_masked.shape}")

    from evaluation.pointmap_estimation import umeyama_alignment

    scale, R, t = umeyama_alignment(pred_pts_masked, gt_pts_masked)

    pred_pts_masked_aligned = R @ (scale * pred_pts_masked) + t[:, np.newaxis]

    print(f"scale: {scale}")
    print(f"R: {R}")
    print(f"t: {t}")

    chamfer_dist = compute_chamfer_distance(
        pred_pts_masked_aligned.T, gt_pts_masked.T, 1.0
    )

    import open3d as o3d
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(gt_pts_masked.T)
    cl, ind = pcd_gt.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_gt = pcd_gt.select_by_index(ind)

    
    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(pred_pts_masked_aligned.T)
    pcd_pred.colors = o3d.utility.Vector3dVector(colors_flat_aligned.astype(np.float32) / 255.0)
    cl, ind = pcd_pred.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_pred = pcd_pred.select_by_index(ind)
    o3d.visualization.draw_geometries([pcd_gt, pcd_pred])

    # sys.exit()
    """

    pred_points_from_depth = unproject_depth_map_to_point_map(
        depth, pred_extrinsic, pred_intrinsic
    )
    pred_points_from_depth_flat = pred_points_from_depth.reshape(-1, 3)
    pred_points_from_depth_flat = (
        pred_points_from_depth_flat @ pred_extrinsic[0][:3, :3].T
        + pred_extrinsic[0][:3, 3]
    )

    gt_points_from_depth = unproject_depth_map_to_point_map(
        gt_depths.cpu().numpy(), gt_extrinsics, gt_intrinsics
    )
    gt_pts_from_depth_flat = gt_points_from_depth.reshape(-1, 3)
    gt_pts_from_depth_flat = (
        gt_pts_from_depth_flat @ gt_extrinsics[0][:3, :3].T + gt_extrinsics[0][:3, 3]
    )

    aligned_pred_points_flat, depth_scales = align_pred_gt_per_depth(
        pred_depth=depth,
        gt_depth=gt_depths.cpu().numpy(),
        valid_mask=gt_masks.cpu().numpy(),
        gt_extrinsics=gt_extrinsics,
        gt_intrinsics=gt_intrinsics,
        pred_extrinsic=pred_extrinsic,
        pred_intrinsic=pred_intrinsic,
        use_gt_extrinsic=False,
    )

    # aligned_pred_points_flat, mean_scale = align_pred_gt_by_pred_extrinsic(
    #     pred_points_from_depth_flat=pred_points_from_depth_flat,
    #     gt_extrinsics=gt_extrinsics,
    #     pred_extrinsic=pred_extrinsic,
    # )

    pred_points_from_depth_flat = aligned_pred_points_flat

    pred_conf = depth_conf  # [S, H, W]
    pred_conf_flat = pred_conf.reshape(-1)

    colors = gt_images.detach().cpu().permute(0, 2, 3, 1).numpy()
    colors = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
    colors_flat = colors.reshape(-1, 3)

    # 按置信度过滤点云
    if cfg.use_conf_filter:
        # 计算分位数阈值，例如 50% 分位数（中位数）
        conf_threshold = np.percentile(pred_conf_flat, cfg.conf_percentile)
        print(f"Conf threshold: {conf_threshold}")
        mask = pred_conf_flat >= conf_threshold
        print(f"Max conf: {pred_conf_flat.max()}, Min conf: {pred_conf_flat.min()}")

        pred_points_from_depth_flat = pred_points_from_depth_flat[mask]
        colors_flat = colors_flat[mask]

    # gt_pts_from_depth_flat = gt_pts_from_depth_flat / scale

    import open3d as o3d

    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(gt_pts_from_depth_flat)
    cl, ind = pcd_gt.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_gt = pcd_gt.select_by_index(ind)

    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(pred_points_from_depth_flat)
    pcd_pred.colors = o3d.utility.Vector3dVector(colors_flat.astype(np.float32) / 255.0)
    cl, ind = pcd_pred.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_pred = pcd_pred.select_by_index(ind)

    # icp
    # icp_result = o3d.pipelines.registration.registration_icp(
    #     pcd_pred,
    #     pcd_gt,
    #     10.0,
    #     np.eye(4),
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    # )
    # print(icp_result.transformation)
    # pcd_pred.transform(icp_result.transformation)

    pred_points_from_depth_flat = np.asarray(pcd_pred.points)
    gt_pts_from_depth_flat = np.asarray(pcd_gt.points)

    chamfer_dist = compute_chamfer_distance(
        pred_points_from_depth_flat, gt_pts_from_depth_flat, 1.0
    )

    if cfg.save_results:
        save_config_to_json(
            cfg,
            os.path.join(save_root, "cfg.json"),
            additional_configs={"token_fusion_strategy_cfg": token_fusion_strategy_cfg},
        )

    if cfg.use_local_display:

        import open3d as o3d

        geometries = [pcd_gt, pcd_pred]
        o3d.visualization.draw_geometries(geometries)


