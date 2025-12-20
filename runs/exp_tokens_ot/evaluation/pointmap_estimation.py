import numpy as np
import torch
import open3d as o3d

from evaluation.utils import umeyama_alignment

def align_point_clouds(depth, pred_extrinsic, pred_intrinsic, gt_depths, gt_extrinsics, gt_intrinsics, gt_images, valid_mask):
    """
    对预测点云和真实点云进行对齐处理

    Args:
        depth: 预测深度图张量，形状 (B, H, W) 或 (B, 1, H, W)
        pred_extrinsic: 预测相机外参，形状 (B, 3, 4)
        pred_intrinsic: 预测相机内参，形状 (B, 3, 3)
        gt_depths: 真实深度图张量，形状 (B, H, W) 或 (B, 1, H, W)
        gt_extrinsics: 真实相机外参，形状 (B, 3, 4)
        gt_intrinsics: 真实相机内参，形状 (B, 3, 3)
        gt_images: 真实图像张量，形状 (B, 3, H, W)
        valid_mask: 有效性掩码，形状 (B, H, W)

    Returns:
        tuple: (aligned_pred_points, aligned_pred_colors, gt_points, gt_colors)
            - aligned_pred_points: 对齐后的预测点云，形状 (N, 3)
            - aligned_pred_colors: 对齐后的预测点云颜色，形状 (N, 3)
            - gt_points: 合并的真实点云，形状 (M, 3)
            - gt_colors: 合并的真实点云颜色，形状 (M, 3)
    """
    from evaluation.point_cloud_fusion import prepare_proj_mats, generate_points_from_depth

    # 生成预测点云
    pred_proj_mats = prepare_proj_mats(pred_extrinsic, pred_intrinsic)
    pred_pnts = generate_points_from_depth(depth.cpu().squeeze().unsqueeze(1), pred_proj_mats).permute(0, 2, 3, 1)

    # 生成真实点云
    gt_proj_mats = prepare_proj_mats(gt_extrinsics, gt_intrinsics)
    gt_pnts = generate_points_from_depth(gt_depths.cpu().squeeze().unsqueeze(1), gt_proj_mats).permute(0, 2, 3, 1)

    # 扩展有效性掩码用于点云过滤
    valid_mask_for_pnts = valid_mask.unsqueeze(-1).expand(-1, -1, -1, 3).cpu().numpy()

    # 过滤有效点云
    masked_pred_pnts = [pred_pnts[i][valid_mask_for_pnts[i]].reshape(-1, 3).cpu().numpy() for i in range(pred_pnts.shape[0])]
    masked_gt_pnts = [gt_pnts[i][valid_mask_for_pnts[i]].reshape(-1, 3).cpu().numpy() for i in range(gt_pnts.shape[0])]
    masked_colors = [gt_images[i].permute(1, 2, 0)[valid_mask_for_pnts[i]].reshape(-1, 3).cpu().numpy() for i in range(gt_images.shape[0])]

    # 对每个视角的点云进行Umeyama对齐
    scales, Rs, ts = [], [], []
    aligned_pred_pnts = []

    for i in range(len(masked_pred_pnts)):
        scale, R, t = umeyama_alignment(masked_pred_pnts[i], masked_gt_pnts[i])
        scales.append(scale)
        Rs.append(R)
        ts.append(t)
        aligned_pred_pnts.append(masked_pred_pnts[i] * scale @ R.T + t)

        print(f"scale: {scale}, R: {R}, t: {t}")

    # 合并对齐后的点云
    aligned_pred_pnts_merged = np.concatenate(aligned_pred_pnts, axis=0)
    aligned_pred_pnts_colors_merged = np.concatenate(masked_colors, axis=0)
    gt_pnts_merged = np.concatenate(masked_gt_pnts, axis=0)
    gt_pnts_colors_merged = np.concatenate(masked_colors, axis=0)

    return aligned_pred_pnts_merged, aligned_pred_pnts_colors_merged, gt_pnts_merged, gt_pnts_colors_merged