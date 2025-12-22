import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any
from datetime import datetime

import cv2
import json
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
from evaluation.depth_estimation import calculate_depth_scales, calculate_depth_scales_shift, process_depth_and_fuse_point_clouds
from evaluation.pointmap_estimation import align_point_clouds
from evaluation.filter import filter_depth_by_conf

from evaluation.pose_estimation import (
    compute_pairwise_relative_errors,
    convert_poses_to_4x4,
)


from evaluation.utils import *
from evaluation.display import display_point_clouds, render_point_cloud_from_view, display_depth_comparison, display_rgb_depth_overlay

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
    knockout_method: str = "top_k"  # "random" or "visible_score" or "top_k" or "corres_mask"
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


def evaluate_scan(scan_id: int, base_cfg: VGGTReconstructConfig, save_root: str = None) -> Dict[str, Any]:
    """
    对单个scan进行评估

    Args:
        scan_id: 要评估的scan ID
        base_cfg: 基础配置（不包含scan_id，由函数参数指定）
        save_root: 保存结果的根目录，如果为None则不保存

    Returns:
        包含评估结果的字典
    """
    # 创建针对当前scan的配置
    cfg = VGGTReconstructConfig(**base_cfg.__dict__)

    print(f"Evaluating scan {scan_id}...")

    loader = DTUScanLoader(
        cfg.dtu_root,
        scan_id=scan_id,
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

    model = VGGT(
        enable_point = False,
        enable_track = False,
        token_weighter=token_weighter
    )
    model.load_state_dict(torch.load(cfg.checkpoint_path), strict=False)
    model.to(cfg.device)
    model.eval()

    gt_images = loader.load_images()
    gt_depths = loader.load_depths()
    gt_points, gt_points_colors = loader.load_points()
    gt_points_np = gt_points.cpu().numpy()
    gt_points_colors_np = gt_points_colors.cpu().numpy()

    gt_masks = loader.load_masks()
    gt_intrinsics, gt_extrinsics = loader.load_cameras()

    @torch.no_grad()
    @torch.amp.autocast("cuda", dtype=cfg.dtype)
    def run_model_with_memory_stats(model, images):
        # 同步并清零峰值显存统计
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        mem_before = torch.cuda.memory_allocated()

        # 统计推理时间（毫秒）
        start = time.time()
        predictions = model(images)
        torch.cuda.synchronize()
        end = time.time()
        inference_time_ms = (end - start) * 1000.0

        mem_after = torch.cuda.memory_allocated()
        peak_mem = torch.cuda.max_memory_allocated()

        additional_peak_mem = peak_mem - mem_before

        return predictions, mem_before, peak_mem, additional_peak_mem, inference_time_ms

    predictions, mem_before, peak_memory, additional_peak_mem, inference_time_ms = run_model_with_memory_stats(model, gt_images)

    # 处理预测结果
    pred_extrinsic, pred_intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], gt_images.shape[-2:]
    )

    depth = predictions["depth"].cpu().squeeze()
    depth_conf = predictions["depth_conf"].cpu().squeeze()

    pred_extrinsic = pred_extrinsic.detach().squeeze(0).cpu().numpy()
    pred_intrinsic = pred_intrinsic.detach().squeeze(0).cpu().numpy()

    depth, conf_mask = filter_depth_by_conf(depth,
                                           depth_conf,
                                           cfg.conf_percentile,
                                           3.0,
                                           verbose=True)

    valid_mask = torch.logical_and(
        gt_depths.squeeze().cpu() > 1e-3,
        conf_mask,
    )

    transformed_pred_points, fused_colors, gt_pnts, gt_pnts_colors = align_point_clouds(
        depth, pred_extrinsic, pred_intrinsic, gt_depths, gt_extrinsics, gt_intrinsics, gt_images, valid_mask
    )

    chamfer_dist = compute_chamfer_distance(
        transformed_pred_points, gt_pnts, 2.0, True if cfg.use_local_display else False
    )

    # 可视化（如果启用）
    if cfg.use_local_display:
        display_point_clouds(
            [transformed_pred_points],
            [fused_colors],
            title=f"transformed pred points - scan {scan_id}",
        )

    # 收集所有调试信息
    debug_info = {}

    if len(token_weighter.token_cosine_similarity) > 0:
        token_similarity_image = np.concatenate(
            token_weighter.token_cosine_similarity, axis=0
        )
        debug_info['token_cosine_similarity'] = token_similarity_image

        if cfg.use_local_display:
            plt.figure()
            plt.imshow(token_similarity_image)
            plt.title(f"token cosine similarity - scan {scan_id}")
            plt.xlabel("Layer Index")
            plt.ylabel("token cosine similarity")
            plt.grid(False)
            plt.show()

    if len(token_weighter.attention_of_all_heads) > 0:
        all_heads_attn_map = np.concatenate(
            token_weighter.attention_of_all_heads, axis=0
        )
        debug_info['attention_of_all_heads'] = all_heads_attn_map

        if cfg.use_local_display:
            plt.figure()
            plt.imshow(all_heads_attn_map)
            plt.colorbar()
            plt.title(f"all heads attention map - scan {scan_id}")
            plt.xlabel("Layer Index")
            plt.ylabel("all heads attention map")
            plt.grid(False)
            plt.show()

    if len(token_weighter.tokens_erank_kernel_norm) > 0:
        debug_info['tokens_erank_kernel_norm'] = token_weighter.tokens_erank_kernel_norm
        if cfg.use_local_display:
            plt.figure()
            plt.plot(token_weighter.tokens_erank_kernel_norm, marker="o")
            plt.title(f"tokens erank kernel norm Over Layers - scan {scan_id}")
            plt.xlabel("Layer Index")
            plt.ylabel("tokens erank kernel norm")
            plt.grid(True)
            plt.show()

    if len(token_weighter.tokens_erank_fro_norm) > 0:
        debug_info['tokens_erank_fro_norm'] = token_weighter.tokens_erank_fro_norm
        if cfg.use_local_display:
            plt.figure()
            plt.plot(token_weighter.tokens_erank_fro_norm, marker="x")
            plt.title(f"tokens erank fro norm Over Layers - scan {scan_id}")
            plt.xlabel("Layer Index")
            plt.ylabel("tokens erank fro norm")
            plt.grid(True)
            plt.show()

    if len(token_weighter.x_cos_similarity) > 0:
        debug_info['x_cos_similarity'] = token_weighter.x_cos_similarity
        if cfg.use_local_display:
            plt.figure()
            plt.plot(token_weighter.x_cos_similarity, marker="o")
            plt.title(f"token cos similarity Over Layers - scan {scan_id}")
            plt.xlabel("Layer Index")
            plt.ylabel("token cos similarity")
            plt.grid(True)
            plt.show()

    if len(token_weighter.q_rope_gain) > 0:
        debug_info['q_rope_gain'] = token_weighter.q_rope_gain
    if len(token_weighter.top_k_dominance) > 0:
        debug_info['top_k_dominance'] = token_weighter.top_k_dominance

        if cfg.use_local_display:
            plt.figure()
            if len(token_weighter.q_rope_gain) > 0:
                plt.plot(token_weighter.q_rope_gain, marker="o")
            plt.plot(token_weighter.top_k_dominance, marker="x")
            plt.title(f"rope gain and top-k dominance Over Layers - scan {scan_id}")
            plt.xlabel("Layer Index")
            plt.ylabel("ratio(rope gain / q_original & top-k num / num_keys)")
            plt.grid(True)
            plt.show()

            plt.figure()
            layer_indices = range(len(token_weighter.top_k_dominance))
            plt.bar(layer_indices, token_weighter.top_k_dominance)
            plt.title(f"top-k dominance Over Layers - scan {scan_id}")
            plt.xlabel("Layer Index")
            plt.ylabel("top-k dominance")
            plt.grid(True, axis="y")
            plt.show()

    # 保存结果
    if save_root is not None:
        scan_save_dir = os.path.join(save_root, f"scan_{scan_id}")
        os.makedirs(scan_save_dir, exist_ok=True)

        # 保存配置
        save_config_to_json(
            cfg,
            os.path.join(scan_save_dir, "cfg.json"),
            additional_configs={"token_fusion_strategy_cfg": token_fusion_strategy_cfg},
        )

        # 保存指标
        metrics_path = os.path.join(scan_save_dir, "metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(
                f"scan_id={scan_id}, num_views={cfg.num_views}, "
                f"use_icp_alignment={cfg.use_icp_alignment}, "
                f"use_stat_filter={cfg.use_stat_filter}, "
                f"chamfer_dist={chamfer_dist}, "
                f"inference_time_ms={inference_time_ms:.3f}, "
                f"mem_before_GB={mem_before / (1024 ** 3):.6f}, "
                f"peak_memory_GB={peak_memory / (1024 ** 3):.6f}, "
                f"additional_peak_memory_GB={additional_peak_mem / (1024 ** 3):.6f}\n"
            )

        # 保存点云
        np.save(
            os.path.join(scan_save_dir, "transformed_pred_points.npy"),
            transformed_pred_points.astype(np.float32),
        )

        # 保存调试图像
        for key, image in debug_info.items():
            if key in ['token_cosine_similarity', 'attention_of_all_heads']:
                # 保存图像
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                # 确保是BGR格式（cv2.imwrite需要）
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                else:
                    image_bgr = image
                cv2.imwrite(os.path.join(scan_save_dir, f"{key}.png"), image_bgr)
            elif key in ['tokens_erank_kernel_norm', 'tokens_erank_fro_norm', 'x_cos_similarity']:
                plt.figure()
                plt.plot(image, marker="o" if "kernel" in key else "x")
                plt.title(f"{key.replace('_', ' ')} Over Layers - scan {scan_id}")
                plt.xlabel("Layer Index")
                plt.ylabel(key.replace('_', ' '))
                plt.grid(True)
                plt.savefig(os.path.join(scan_save_dir, f"{key}.png"))
                plt.close()
            elif key == 'top_k_dominance':
                plt.figure()
                layer_indices = range(len(image))
                plt.bar(layer_indices, image)
                plt.title(f"top-k dominance Over Layers - scan {scan_id}")
                plt.xlabel("Layer Index")
                plt.ylabel("top-k dominance")
                plt.grid(True, axis="y")
                plt.savefig(os.path.join(scan_save_dir, f"{key}.png"))
                plt.close()

        # 保存渲染图像
        try:
            H, W = gt_images.shape[-2], gt_images.shape[-1]
            view_idx = 1

            render_path = os.path.join(scan_save_dir, f"transformed_pred_vs_gt_cam{view_idx}.png")
            render_point_cloud_from_view(
                pred_points_world=transformed_pred_points,
                pred_colors=fused_colors,
                gt_points_world=gt_points_np,
                gt_colors=gt_points_colors_np,
                intrinsic=pred_intrinsic[view_idx],
                extrinsic=pred_extrinsic[view_idx],
                image_height=H,
                image_width=W,
                view_idx=view_idx,
                save_path=render_path,
            )
        except Exception as e:
            print(f"[save_results] WARNING: failed to render and save Open3D image for scan {scan_id}: {e}")

    # 返回评估结果
    return {
        'scan_id': scan_id,
        'chamfer_dist': chamfer_dist,
        'inference_time_ms': inference_time_ms,
        'mem_before_GB': mem_before / (1024 ** 3),
        'peak_memory_GB': peak_memory / (1024 ** 3),
        'additional_peak_memory_GB': additional_peak_mem / (1024 ** 3),
        'transformed_pred_points': transformed_pred_points,
        'fused_colors': fused_colors,
        'gt_points': gt_pnts,
        'gt_colors': gt_pnts_colors,
        'debug_info': debug_info
    }

# 使用示例
"""
# 批量评估多个scan
from exp_runner import evaluate_multiple_scans, VGGTReconstructConfig

cfg = VGGTReconstructConfig()
scan_ids = [1, 2, 3, 4, 5]
results = evaluate_multiple_scans(scan_ids, cfg, save_root="/path/to/save")

# 或者逐个评估（使用evaluate_scan函数）
from exp_runner import evaluate_scan

results = []
for scan_id in [1, 2, 3, 4, 5]:
    result = evaluate_scan(scan_id, cfg, save_root="/path/to/save")
    results.append(result)

# 结果包含：
# - chamfer_dist: Chamfer距离
# - inference_time_ms: 推理时间
# - mem_before_GB/peak_memory_GB/additional_peak_memory_GB: 内存使用情况
# - transformed_pred_points/fused_colors: 预测点云
# - gt_points/gt_colors: 真值点云
# - debug_info: 调试信息（如果启用）
"""

def evaluate_multiple_scans(scan_ids: List[int], base_cfg: VGGTReconstructConfig, save_root: str = None) -> List[Dict[str, Any]]:
    """
    对多个scan进行批量评估

    Args:
        scan_ids: 要评估的scan ID列表
        base_cfg: 基础配置
        save_root: 保存结果的根目录，如果为None则不保存

    Returns:
        包含所有scan评估结果的列表
    """
    results = []

    # 显示GPU信息
    import GPUtil
    if torch.cuda.is_available():
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU {gpu.id}: {gpu.name}, total {gpu.memoryTotal}MB, used {gpu.memoryUsed}MB, free {gpu.memoryFree}MB")
    else:
        print(f"RAM usage: {psutil.virtual_memory().used/1024/1024:.2f} MB (of {psutil.virtual_memory().total/1024/1024:.2f} MB)")

    for scan_id in scan_ids:
        try:
            result = evaluate_scan(scan_id, base_cfg, save_root)
            results.append(result)
            print(f"Scan {scan_id} evaluation completed. Chamfer distance: {result['chamfer_dist']:.6f}")
        except Exception as e:
            print(f"Error evaluating scan {scan_id}: {e}")
            results.append({
                'scan_id': scan_id,
                'error': str(e)
            })

    return results


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

    parser.add_argument('--dtu_root', type=str, default="/home/vision/ws/datasets/SampleSet/dtu_mvs")
    parser.add_argument('--scan_ids', type=int, nargs='+', required=True, help='scan IDs，用于批量评估 (e.g., --scan_ids 1 2 3 4)')
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

    parser.add_argument('--use_stat_filter', type=str2bool, default=False)
    parser.add_argument('--use_icp_alignment', type=str2bool, default=False)
    
    parser.add_argument('--use_local_display', type=str2bool, default=False)
    parser.add_argument('--save_results', type=str2bool, default=False)

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

    # 获取要评估的scan IDs
    scan_ids = args.scan_ids
    print(f"Evaluating scans: {scan_ids}")

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_root = None
    if cfg.save_results:
        save_root = os.path.join(cfg.output_dir, timestamp)
        os.makedirs(save_root, exist_ok=True)
        print(f"Create output directory: {save_root}")

        # 保存全局配置
        save_config_to_json(
            cfg,
            os.path.join(save_root, "global_cfg.json"),
            additional_configs={},
        )

    # 执行评估
    results = evaluate_multiple_scans(scan_ids, cfg, save_root)

    # 输出汇总结果
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)

    successful_results = [r for r in results if 'error' not in r]
    failed_results = [r for r in results if 'error' in r]

    if successful_results:
        chamfer_distances = [r['chamfer_dist'] for r in successful_results]
        inference_times = [r['inference_time_ms'] for r in successful_results]

        print(f"Successfully evaluated {len(successful_results)} scans")
        avg_chamfer = np.mean(chamfer_distances)
        avg_infer_time = np.mean(inference_times)
        print(f"Average Chamfer Distance: {avg_chamfer:.6f}")
        print(f"Average Inference Time (ms): {avg_infer_time:.3f}")

        for result in successful_results:
            print(f"  Scan {result['scan_id']}: Chamfer={result['chamfer_dist']:.6f}, Time={result['inference_time_ms']:.3f}ms")

    if failed_results:
        print(f"\nFailed to evaluate {len(failed_results)} scans:")
        for result in failed_results:
            print(f"  Scan {result['scan_id']}: {result['error']}")

    # 保存汇总结果到JSON
    if cfg.save_results and save_root:
        summary_path = os.path.join(save_root, "evaluation_summary.json")
        with open(summary_path, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'scan_ids': scan_ids,
                'successful_results': successful_results,
                'failed_results': failed_results,
                'summary': {
                    'total_scans': len(results),
                    'successful_scans': len(successful_results),
                    'failed_scans': len(failed_results),
                    'avg_chamfer_dist': float(np.mean(chamfer_distances)) if successful_results else None,
                    'std_chamfer_dist': float(np.std(chamfer_distances)) if successful_results else None,
                    'avg_inference_time_ms': float(np.mean(inference_times)) if successful_results else None,
                    'std_inference_time_ms': float(np.std(inference_times)) if successful_results else None,
                }
            }, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")
