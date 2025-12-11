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

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

from data.dtu_loader import DTUScanLoader
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.layers.token_weighter import TokenFusionStrategy, TokenFusionStrategyConfig
from eval.criterion import Regr3D_t_ScaleShiftInv, L21
from vggt.utils.geometry import depth_to_world_coords_points, depth_to_cam_coords_points
from evaluation.depth_estimation import (
    calculate_depth_error,
    align_pred_to_gt,
    align_pred_gt_per_depth,
    align_pred_gt_by_pred_extrinsic,
)
from evaluation.pose_estimation import (
    compute_pairwise_relative_errors,
    convert_poses_to_4x4,
)
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

    cfg = VGGTReconstructConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=(
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        ),
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
        knockout_layer_idx=range(0, 5, 1),
        # knockout_layer_idx=[],
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
            additional_configs={"token_fusion_strategy_cfg": token_fusion_strategy_cfg},
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
    gt_points, gt_points_colors = loader.load_points()  # [N, 3]
    gt_masks = loader.load_masks()  # List[np.ndarray] 或类似格式
    gt_intrinsics, gt_extrinsics = loader.load_cameras()  # (S, 3, 3), (S, 4, 4)

    with torch.no_grad():
        with torch.amp.autocast(cfg.device, dtype=cfg.dtype):
            current_memory = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            predictions = model(gt_images)
            additional_memory = torch.cuda.memory_allocated() - (current_memory + 1e9)
            peak_memory = torch.cuda.max_memory_allocated()
            additional_peak_memory = peak_memory - (current_memory + 1e9)

            print(f"Additional memory used: {additional_memory / (1024 ** 3)} GB")
            print(
                f"Additional peak memory used: {additional_peak_memory / (1024 ** 3)} GB"
            )

            if cfg.save_results:
                with open(os.path.join(save_root, "memory_usage.txt"), "a") as f:
                    f.write(
                        f"Additional memory used: {additional_memory / (1024 ** 3)} GB\n"
                    )
                    f.write(
                        f"Additional peak memory used: {additional_peak_memory / (1024 ** 3)} GB\n"
                    )
                    f.write(f"Current memory: {current_memory / (1024 ** 3)} GB\n")
                    f.write(f"Peak memory: {peak_memory / (1024 ** 3)} GB\n")

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

    depth = predictions["depth"].detach().cpu().numpy().reshape(gt_depths.shape)
    depth_conf = (
        predictions["depth_conf"].detach().cpu().numpy().reshape(gt_depths.shape)
    )
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

    chamfer_dist = compute_chamfer_distance(
        pred_points_from_depth_flat, gt_pts_from_depth_flat, 5.0
    )
