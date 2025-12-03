import os
import sys
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
import torch
import glob
import imageio
import matplotlib.pyplot as plt

# current_work_dir = os.getcwd()
# sys.path += [
#     os.path.join(current_work_dir, "vggt"),
#     os.path.join(current_work_dir, "spann3r")
# ]

from data.dtu_loader import DTUScanLoader
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.layers.token_weighter import TokenFusionStrategy
from eval.criterion import Regr3D_t_ScaleShiftInv, L21
from vggt.utils.geometry import depth_to_world_coords_points, depth_to_cam_coords_points


def plot_conf_histogram(conf_flat: np.ndarray, save_path: str = None):
    """
    绘制置信度的直方图，并在图上显示统计信息。
    
    Args:
        conf_flat: 一维置信度数组
        save_path: 保存路径（可选），如果为None则不保存
    """
    # 计算统计量
    conf_max = conf_flat.max()
    conf_min = conf_flat.min()
    conf_mean = conf_flat.mean()
    conf_median = np.median(conf_flat)
    conf_std = conf_flat.std()
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    conf_percentiles = {p: np.percentile(conf_flat, p) for p in percentiles}
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制直方图
    n, bins, patches = ax.hist(conf_flat, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
    
    # 在图上添加统计信息文本
    stats_text = (
        f"Max: {conf_max:.4f}\n"
        f"Min: {conf_min:.4f}\n"
        f"Mean: {conf_mean:.4f}\n"
        f"Median: {conf_median:.4f}\n"
        f"Std: {conf_std:.4f}\n"
        f"10th: {conf_percentiles[10]:.4f} | 50th: {conf_percentiles[50]:.4f} | 90th: {conf_percentiles[90]:.4f}\n"
        f"95th: {conf_percentiles[95]:.4f} | 99th: {conf_percentiles[99]:.4f}"
    )
    
    # 将统计信息放在图的右上角
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 添加垂直线标记中位数和均值
    ax.axvline(conf_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {conf_mean:.4f}')
    ax.axvline(conf_median, color='green', linestyle='--', linewidth=2, label=f'Median: {conf_median:.4f}')
    
    # 设置标签和标题
    ax.set_xlabel('Confidence Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Confidence Distribution Histogram', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confidence histogram saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


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

    dtu_root: str = "/home/vision/ws/datasets/SampleSet/dtu_mvs"
    example_images_path: str = "/home/vision/ws/vggt/examples/kitchen/images"
    target_size: Tuple[int, int] = (518, 350)

    output_dir: str = "/home/vision/ws/vggt/runs/exp_tokens_ot/results"
    attn_map_save_dir: str = os.path.join(output_dir, "attention_maps")

    # test images source
    test_images_source: str = "dtu" # or "dtu"
    scan_id: int = 1
    num_images: int = 4

    # 点云来源模式：
    # - "depth": 使用 depth + 相机（unproject_depth_map_to_point_map）
    # - "world": 使用模型直接输出的 predictions["world_points"]
    point_source: str = "depth"

    # 置信度过滤配置（对最终导出的点云生效）
    # 是否根据置信度过滤点云
    use_conf_filter: bool = False

    # 百分位阈值（0~100），例如 50.0 表示保留大于等于中位数置信度的点
    conf_percentile: float = 30.0

    # use local display
    use_local_display: bool = True

    # save attn map
    save_attn_map: bool = False

    # save results
    save_results: bool = False

    # 是否使用 ICP 配准
    use_icp: bool = True


if __name__ == "__main__":
    cfg = VGGTReconstructConfig()

    token_weighter_args = {
        "dtu_mvs_root": cfg.dtu_root,
        "scan_id": cfg.scan_id,
        "num_views": cfg.num_images,
        "step": 1,
        "device": 'cpu',
        "target_size": (518, 350),
        "patch_size": 14,
        "special_tokens_num": 5,


        # for attention knockout
        "knockout_layer_idx": [],
        "knockout_method": "random",
        # for random knockout
        "knockout_random_ratio": 0.5,
        # for visible score knockout nothing to do
        # for top-k preserved knockout      
        "knockout_top_k": 100,
    }

    token_weighter = TokenFusionStrategy(token_weighter_args)

    model = VGGT(attn_map_save_dir=cfg.attn_map_save_dir if cfg.save_attn_map else None, token_weighter=token_weighter)
    model.load_state_dict(torch.load(cfg.checkpoint_path))
    model.to(cfg.device)
    model.eval()

    gts = []  # 初始化 gts 列表
    if cfg.test_images_source == "example":
        gt_images = get_example_images(cfg.example_images_path, cfg.num_images, device=cfg.device)
        print("gt_images.shape: ", gt_images.shape)
    elif cfg.test_images_source == "dtu":
        loader = DTUScanLoader(
            cfg.dtu_root,
            scan_id=cfg.scan_id,
            num_views=cfg.num_images,
            step=1,
            device=cfg.device,
            target_size=cfg.target_size,
        )
        gt_images = loader.load_images()
        gt_depths = loader.load_depths()  # List[np.ndarray] 或类似格式
        gt_masks = loader.load_masks()    # List[np.ndarray] 或类似格式
        gt_intrinsics, gt_extrinsics = loader.load_cameras()  # (S, 3, 3), (S, 4, 4)

        print("gt_images.shape: ", gt_images.shape)
        print("gt_depths.shape: ", gt_depths.shape)
        print("gt_masks.shape: ", gt_masks.shape)
        print("gt_intrinsics.shape: ", gt_intrinsics.shape)
        print("gt_extrinsics.shape: ", gt_extrinsics.shape)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=cfg.dtype):
            predictions = model(gt_images)

    # ----------------------------------------------------------------------
    # 仿照 demo_gradio.py 的方式：
    # 1) 用 pose_enc 转 extrinsic / intrinsic
    # 2) 用 depth + 相机参数通过 unproject_depth_map_to_point_map 得到点云
    # 3) 全部转换为 numpy 并保存
    # ----------------------------------------------------------------------
    save_root = cfg.output_dir
    os.makedirs(save_root, exist_ok=True)

    # 1) extrinsic / intrinsic
    # gt_images: [S, 3, H, W]，VGGT 内部会加 batch 维度 => pose_enc: [B, S, 9]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], gt_images.shape[-2:]
    )

    # 2) 转 numpy & 去掉 batch 维
    depth = predictions["depth"].detach().cpu().numpy().reshape(gt_depths.shape)          # [B, S, H, W, 1]
    depth_conf = predictions["depth_conf"].detach().cpu().numpy().reshape(gt_depths.shape) # [B, S, H, W]
    extrinsic = extrinsic.detach().squeeze(0).cpu().numpy()             # [B, S, 3, 4] 或类似
    intrinsic = intrinsic.detach().squeeze(0).cpu().numpy()                 # [B, S, 3, 3] 或类似

    def calculate_depth_error(
        pred_depth: np.ndarray,
        gt_depth: np.ndarray,
        mask: np.ndarray,
    ):
        """
        对每一张深度图，估计一个缩放因子 scale，使得缩放后的预测深度与 GT 在 mask 内
        的 L2 误差最小，然后返回每张图的绝对误差图和平均绝对误差。

        pred_depth: 预测深度，形状 [S, H, W]
        gt_depth: GT 深度，形状 [S, H, W]
        mask: 可用像素的 mask，形状 [S, H, W]，非零为有效
        """
        S = pred_depth.shape[0]
        per_view_scale = []

        for s in range(S):
            pred = pred_depth[s]   # [H, W]
            gt = gt_depth[s]       # [H, W]
            m = mask[s]            # [H, W]

            valid = m != 0
            if not np.any(valid):
                per_view_scale.append(1.0)
                continue

            pred_valid = pred[valid].astype(np.float64)
            gt_valid = gt[valid].astype(np.float64)

            # 最小二乘意义下的最佳缩放：argmin_scale || scale * pred - gt ||_2^2
            num = np.sum(pred_valid * gt_valid)
            den = np.sum(pred_valid * pred_valid) + 1e-8
            scale = num / den
            per_view_scale.append(scale)

        return per_view_scale

    # 计算深度误差（可选调试用）
    if cfg.test_images_source == "dtu":
        pred_scale_list = calculate_depth_error(
            depth,                      # [B, S, H, W, 1]
            gt_depths.cpu().numpy(),    # [S, H, W]
            gt_masks.cpu().numpy(),     # [S, H, W]
        )

        depth_display = []
        pred_display = []
        error_display = []
        for s in range(depth.shape[0]):
            depth_display.append(depth[s] * pred_scale_list[s]) 
            pred_display.append(gt_depths[s].cpu().numpy())
            # error_display.append(np.abs(depth[s] * pred_scale_list[s] - gt_depths[s].cpu().numpy()))
            error_display.append(depth[s] * pred_scale_list[s] - gt_depths[s].cpu().numpy())

        depth_display = np.concatenate(depth_display, axis=-1)
        pred_display = np.concatenate(pred_display, axis=-1)
        error_display = np.concatenate(error_display, axis=-1)

        display = np.concatenate([depth_display, pred_display, error_display], axis=0)
        # show
        plt.imshow(display)
        plt.show()


        
    
    # 3) 得到点云：
    # 根据配置 cfg.point_source 决定：
    # - "depth": 使用 depth + 相机参数反投影，对应的置信度使用 depth_conf
    # - "world": 直接使用模型输出的 world_points，对应的置信度使用 world_points_conf
    if cfg.point_source == "depth":
        world_points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic) 
        # 置信度：使用 depth_conf
        world_conf = depth_conf  # [S, H, W]
    elif cfg.point_source == "world":
        # 直接使用模型输出的 world_points
        world_points = predictions["world_points"].detach().cpu().numpy()  # [B, S, H, W, 3]
        world_points = np.squeeze(world_points, axis=0)  # [S, H, W, 3]
        world_conf = predictions["world_points_conf"].detach().cpu().numpy()
        world_conf = np.squeeze(world_conf, axis=0)  # [S, H, W]
    else:
        raise ValueError(f"Unknown point_source: {cfg.point_source}")
    
    if cfg.save_results:
        # 保存 numpy
        depth_path = os.path.join(save_root, "depth.npy")
        depth_conf_path = os.path.join(save_root, "depth_conf.npy")
        extrinsic_path = os.path.join(save_root, "extrinsic.npy")
        intrinsic_path = os.path.join(save_root, "intrinsic.npy")
        points_path = os.path.join(save_root, "world_points.npy")

        np.save(depth_path, depth)
        np.save(depth_conf_path, depth_conf)
        np.save(extrinsic_path, extrinsic)
        np.save(intrinsic_path, intrinsic)
        np.save(points_path, world_points)

        print(f"Depth saved to {depth_path}")
        print(f"Depth conf saved to {depth_conf_path}")
        print(f"Extrinsic saved to {extrinsic_path}")
        print(f"Intrinsic saved to {intrinsic_path}")
        print(f"World points saved to {points_path} (source = {cfg.point_source})")

        # ------------------------------------------------------------------
        # 保存 depth 为 PNG（归一化到 0-255）
        # depth: [S, H, W, 1] 或 [S, H, W]
        # ------------------------------------------------------------------
        depth_squeezed = depth.squeeze(-1) if depth.ndim == 4 else depth  # [S, H, W]
        S_depth = depth_squeezed.shape[0]
        
        for view_idx in range(S_depth):
            depth_view = depth_squeezed[view_idx]  # [H, W]
            
            # 归一化到 0-255
            depth_min = depth_view.min()
            depth_max = depth_view.max()
            if depth_max > depth_min:
                depth_normalized = ((depth_view - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            else:
                depth_normalized = np.zeros_like(depth_view, dtype=np.uint8)
            
            depth_png_path = os.path.join(save_root, f"depth_view{view_idx:03d}.png")
            cv2.imwrite(depth_png_path, depth_normalized)
        
        print(f"Depth PNG images saved ({S_depth} views)")

        # ------------------------------------------------------------------
        # 保存 RGB 图像为 PNG（归一化到 0-255）
        # gt_images: [S, 3, H, W]，值在 [0, 1] 范围
        # ------------------------------------------------------------------
        imgs_cpu = gt_images.detach().cpu()  # [S, 3, H, W]
        imgs_rgb = imgs_cpu.permute(0, 2, 3, 1).numpy()  # [S, H, W, 3]
        imgs_rgb = (np.clip(imgs_rgb, 0.0, 1.0) * 255).astype(np.uint8)  # [S, H, W, 3], 0-255
        
        S_rgb = imgs_rgb.shape[0]
        for view_idx in range(S_rgb):
            rgb_view = imgs_rgb[view_idx]  # [H, W, 3], RGB
            # cv2.imwrite 需要 BGR 格式
            rgb_view_bgr = cv2.cvtColor(rgb_view, cv2.COLOR_RGB2BGR)
            rgb_png_path = os.path.join(save_root, f"rgb_view{view_idx:03d}.png")
            cv2.imwrite(rgb_png_path, rgb_view_bgr)
        
        print(f"RGB PNG images saved ({S_rgb} views)")

        # ------------------------------------------------------------------
        # 导出 PLY
        # 使用第一个视角的点云和对应图像颜色
        # gt_images: [S, 3, H, W]，world_points: [S, H, W, 3]
        #
        # 新增：根据 depth_conf 对点云进行过滤，保留高置信度像素
        # ------------------------------------------------------------------
        ply_path = os.path.join(save_root, f"point_cloud_all_{cfg.point_source}.ply")

    # 将所有视角的点云和置信度、颜色全部拼接在一起
    # world_points: [S, H, W, 3]
    # world_conf:   [S, H, W]
    # gt_images:         [S, 3, H, W]
    S, H, W, _ = world_points.shape

    # 点和置信度
    pts_flat = world_points.reshape(-1, 3)          # [S*H*W, 3]
    print("pts_flat.shape: ", pts_flat.shape)
    conf_flat = world_conf.reshape(-1)              # [S*H*W]

    if cfg.save_results:
        conf_histogram_path = os.path.join(save_root, "conf_histogram.png")
        plot_conf_histogram(conf_flat, save_path=conf_histogram_path)

    # 颜色：先变为 (S, H, W, 3)，再 flatten
    imgs_cpu = gt_images.detach().cpu()                  # [S, 3, H, W]
    colors = imgs_cpu.permute(0, 2, 3, 1).numpy()   # [S, H, W, 3]
    colors = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
    colors_flat = colors.reshape(-1, 3)             # [S*H*W, 3]

    # 按置信度过滤点云
    if cfg.use_conf_filter:
        # 计算分位数阈值，例如 50% 分位数（中位数）
        conf_threshold = np.percentile(conf_flat, cfg.conf_percentile)
        print(f"Conf threshold: {conf_threshold}")
        mask = conf_flat >= conf_threshold
        print(f"Max conf: {conf_flat.max()}, Min conf: {conf_flat.min()}")

        pts_flat = pts_flat[mask]
        colors_flat = colors_flat[mask]

    if cfg.save_results:
        num_vertices = pts_flat.shape[0]

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
            for (x, y, z), (r, g, b) in zip(pts_flat, colors_flat):
                f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")

        print(f"PLY point cloud (all views, source = {cfg.point_source}) saved to {ply_path}")

    if cfg.test_images_source == "dtu":
        gts = []
        S = gt_images.shape[0]
        for i in range(S):
            world_coords, cam_coords, valid_mask = depth_to_world_coords_points(
                gt_depths[i].cpu().numpy(), gt_extrinsics[i][:3, :], gt_intrinsics[i]
            )

            gt_dict = {
                "camera_pose": torch.from_numpy(gt_extrinsics[i]).float(),  # [1, 4, 4]
                "images": torch.from_numpy(gt_images[i].cpu().numpy()).float(),  # [1, 3, H, W]
                "pts3d": torch.from_numpy(world_coords).float(),  # [1, H, W, 3]
                "valid_mask": torch.from_numpy(valid_mask).bool(),  # [1, H, W]
            }
            gts.append(gt_dict)
        
        pts_gt_flat = []
        colors_gt_flat = []
        mask_gt_flat = []

        for s in range(S):
            mask = gts[s]["valid_mask"].cpu().numpy().reshape(-1)
            mask_gt_flat.append(mask)
            pts = gts[s]["pts3d"].cpu().numpy().reshape(-1, 3)
            colors = gts[s]["images"].cpu().permute(1, 2, 0).numpy().reshape(-1, 3)
            pts_gt_flat.append(pts)
            colors_gt_flat.append(colors)

        mask_gt_flat = np.concatenate(mask_gt_flat)  # 使用 concatenate 连接一维数组
        pts_gt_flat = np.vstack(pts_gt_flat)
        colors_gt_flat = np.vstack(colors_gt_flat)
        pts_gt_flat = pts_gt_flat[mask_gt_flat]
        colors_gt_flat = colors_gt_flat[mask_gt_flat]
        colors_gt_flat = (np.clip(colors_gt_flat, 0.0, 1.0) * 255).astype(np.uint8)

        # 对预测点云应用相同的 mask（确保点数匹配）
        # 注意：只有在预测点云还没有被置信度过滤时才能使用相同的 mask
        if len(pts_flat) == len(mask_gt_flat):
            pts_flat = pts_flat[mask_gt_flat]
            colors_flat = colors_flat[mask_gt_flat]
        else:
            print(f"Warning: pts_flat length ({len(pts_flat)}) != mask_gt_flat length ({len(mask_gt_flat)}), skipping mask application")

        
        from eval.dataset_utils.corr import inv, geotrf
        
        camera1_pose_inv = inv(gts[0]["camera_pose"].numpy())
        pts_gt_flat = geotrf(camera1_pose_inv, pts_gt_flat)
        
        # 根据平均距离，将预测点云的尺度对齐到 GT 点云的尺度
        gt_dist = np.linalg.norm(pts_gt_flat, axis=1)
        gt_avg_scale = gt_dist.sum() / (gt_dist.shape[0] + 1e-3)

        pred_dist = np.linalg.norm(pts_flat, axis=1)
        pred_avg_scale = pred_dist.sum() / (pred_dist.shape[0] + 1e-3)

        # 预测点云按比例缩放，使其平均距离与 GT 一致
        if pred_avg_scale > 0:
            scale_factor = pred_avg_scale / gt_avg_scale
            pts_flat = pts_flat / scale_factor

        if cfg.use_icp:
            # ICP配准：改进初始对齐和参数
            import open3d as o3d
            from open3d.pipelines import registration
            
            # 创建点云对象
            source = o3d.geometry.PointCloud()
            source.points = o3d.utility.Vector3dVector(pts_gt_flat)
            target = o3d.geometry.PointCloud()
            target.points = o3d.utility.Vector3dVector(pts_flat)
            
            # 估计法向量（PointToPlane需要）
            source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            
            # 计算初始对齐：基于质心对齐
            source_center = np.mean(pts_gt_flat, axis=0)
            target_center = np.mean(pts_flat, axis=0)
            initial_translation = target_center - source_center
            initial_transform = np.eye(4)
            initial_transform[:3, 3] = initial_translation
            
            # 多尺度ICP：先粗配准再精配准
            # 粗配准：使用较大的阈值和PointToPoint方法
            print("Running coarse ICP...")
            icp_coarse = registration.registration_icp(
                source, target,
                max_correspondence_distance=0.1,  # 较大的初始阈值
                init=initial_transform,
                estimation_method=registration.TransformationEstimationPointToPoint(),
                criteria=registration.ICPConvergenceCriteria(max_iteration=50)
            )
            print(f"Coarse ICP fitness: {icp_coarse.fitness}, inlier_rmse: {icp_coarse.inlier_rmse}")
            
            # 精配准：使用较小的阈值和PointToPlane方法
            print("Running fine ICP...")
            icp_fine = registration.registration_icp(
                source, target,
                max_correspondence_distance=0.05,  # 较小的阈值用于精配准
                init=icp_coarse.transformation,  # 使用粗配准的结果作为初始值
                estimation_method=registration.TransformationEstimationPointToPlane(),
                criteria=registration.ICPConvergenceCriteria(max_iteration=100)
            )
            print(f"Fine ICP fitness: {icp_fine.fitness}, inlier_rmse: {icp_fine.inlier_rmse}")
            
            # 应用最终变换到GT点云
            icp_result = icp_fine
            pts_gt_flat = (pts_gt_flat @ icp_result.transformation[:3, :3].T + icp_result.transformation[:3, 3])

            # calculate chamfer distance
            # Chamfer Distance 定义：双向最近点匹配的平均距离
            # CD(A,B) = (1/|A|) * Σ_{a∈A} min_{b∈B} ||a-b|| + (1/|B|) * Σ_{b∈B} min_{a∈A} ||b-a||
            source_transformed = o3d.geometry.PointCloud()
            source_transformed.points = o3d.utility.Vector3dVector(pts_gt_flat)
            target_for_chamfer = o3d.geometry.PointCloud()
            target_for_chamfer.points = o3d.utility.Vector3dVector(pts_flat)
            
            # 计算双向最近点距离
            # dists_source_to_target[i] = source 中第 i 个点到 target 的最近距离
            dists_source_to_target = source_transformed.compute_point_cloud_distance(target_for_chamfer)
            # dists_target_to_source[i] = target 中第 i 个点到 source 的最近距离
            dists_target_to_source = target_for_chamfer.compute_point_cloud_distance(source_transformed)
            
            # Chamfer Distance = 两个方向距离的平均值（加权平均，按点数）
            # 注意：这里已经是按点数加权的，因为 mean 就是除以点数
            mean_dist_source_to_target = np.mean(dists_source_to_target)  # (1/|source|) * Σ
            mean_dist_target_to_source = np.mean(dists_target_to_source)  # (1/|target|) * Σ
            chamfer_distance = (mean_dist_source_to_target + mean_dist_target_to_source) / 2.0
            
            print(f"Chamfer distance: {chamfer_distance}")
            print(f"  Source->Target mean distance: {mean_dist_source_to_target}")
            print(f"  Target->Source mean distance: {mean_dist_target_to_source}")

    if cfg.use_local_display:
        import open3d as o3d
        
        # 创建预测点云（使用经过置信度过滤的点）
        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(pts_flat)
        pcd_pred.colors = o3d.utility.Vector3dVector(colors_flat.astype(np.float32) / 255.0)
        # pcd_pred.paint_uniform_color([1, 0, 0])
        geometries = []
        geometries.append(pcd_pred)

        if cfg.test_images_source == "dtu":
            pcd_gt = o3d.geometry.PointCloud()
            pcd_gt.points = o3d.utility.Vector3dVector(pts_gt_flat)
            pcd_gt.colors = o3d.utility.Vector3dVector(colors_gt_flat.astype(np.float32) / 255.0)
            # pcd_gt.paint_uniform_color([0, 1, 0])
            geometries.append(pcd_gt)
        
        o3d.visualization.draw_geometries(geometries)
    sys.exit()