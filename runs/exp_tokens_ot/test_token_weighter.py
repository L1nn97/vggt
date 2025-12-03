from __future__ import annotations

import cv2
import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

from data.dtu_loader import DTUScanLoader
from vggt.utils.geometry import depth_to_world_coords_points, depth_to_cam_coords_points
from vggt.dependency.projection import project_3D_points_np


def process_dtu_view(loader: DTUScanLoader, view_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a DTU view and return the processed image, depth, world coordinates, and intrinsic matrix.
    
    Args:
        loader: DTUScanLoader instance
        view_idx: The index of the view to process

    Returns:
        Tuple containing:
        - image: Processed image as (H, W, 3) numpy array with uint8 values [0, 255]
        - depth: Processed depth map as (H, W) numpy array, resized to match image
        - mask: Processed mask as (H, W) numpy array, resized to match image
        - world_coords: World coordinates as (H, W, 3) numpy array
        - intrinsic: Adjusted intrinsic matrix (3, 3) numpy array
        - extrinsic: Extrinsic matrix (4, 4) numpy array
    """
    # 加载数据
    images = loader.load_images()
    depths = loader.load_depths()
    masks = loader.load_masks()
    intrinsics, extrinsics = loader.load_cameras()

    # 获取指定视图的数据
    image = images[view_idx]
    depth = depths[view_idx]
    mask = masks[view_idx]
    intrinsic = intrinsics[view_idx]
    extrinsic = extrinsics[view_idx]
    
    # 处理深度图格式
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()
    if len(depth.shape) == 3:
        depth = depth.squeeze(-1)
    
    # 处理图像格式
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if len(image.shape) == 3 and image.shape[0] == 3:
        # (3, H, W) -> (H, W, 3)
        image = image.transpose(1, 2, 0)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # 获取图像尺寸
    img_h, img_w = image.shape[:2]
    depth_h, depth_w = depth.shape[:2]
    mask_h, mask_w = mask.shape[:2] if mask is not None else (None, None)
    
    # 调整深度图以匹配图像尺寸
    if (depth_h, depth_w) != (img_h, img_w):
        depth = cv2.resize(depth, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        
        # 调整内参矩阵以匹配新的深度图尺寸
        scale_x = img_w / depth_w
        scale_y = img_h / depth_h
        intrinsic = intrinsic.copy()
        intrinsic[0, 0] *= scale_x  # fx
        intrinsic[1, 1] *= scale_y  # fy
        intrinsic[0, 2] *= scale_x  # cx
        intrinsic[1, 2] *= scale_y  # cy
    if (mask_h, mask_w) != (img_h, img_w):
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.uint8)
        mask = mask > 0
    
    # 确保外参是 (3, 4) 格式
    extrinsic_3x4 = extrinsic[:3, :] if extrinsic.shape[0] == 4 else extrinsic
    
    # 反投影到世界坐标
    world_coords, cam_coords, valid_mask = depth_to_world_coords_points(
        depth, extrinsic_3x4, intrinsic
    )
    
    return image, depth, mask, world_coords, intrinsic, extrinsic
    
def reproj_world_coords_to_image(world_coords_src: np.ndarray, intrinsic_tgt: np.ndarray, extrinsic_tgt: np.ndarray) -> np.ndarray:
    # 确保 extrinsic_tgt 是 (3, 4) 格式
    extrinsic_tgt_3x4 = extrinsic_tgt[:3, :] if extrinsic_tgt.shape[0] == 4 else extrinsic_tgt
    
    # 准备参数：project_3D_points_np 需要 (B, 3, 4) 和 (B, 3, 3) 格式
    points3D = world_coords_src.reshape(-1, 3)  # (N, 3)
    extrinsics_batch = extrinsic_tgt_3x4[None]  # (1, 3, 4)
    intrinsics_batch = intrinsic_tgt[None]  # (1, 3, 3)
    
    # 投影到 image_1
    reproj_uv, reproj_cam = project_3D_points_np(
        points3D, extrinsics_batch, intrinsics_batch
    )
    # reproj_uv shape: (1, N, 2) -> (N, 2)
    reproj_uv = reproj_uv[0].reshape(world_coords_src.shape[0], world_coords_src.shape[1], 2)  # (H, W, 2)
    return reproj_uv

def display_correspondence(src_image: np.ndarray, tgt_image: np.ndarray, reproj_uv: np.ndarray, src_idx: np.ndarray):
    """
    Display correspondence between source and target images.

    Args:
        src_image: Source image as (H, W, 3) numpy array with uint8 values [0, 255]
        tgt_image: Target image as (H, W, 3) numpy array with uint8 values [0, 255]
        reproj_uv: Reprojected UV coordinates as (H, W, 2) numpy array
        src_idx: Source index as (N, 2) numpy array
    """
    assert src_idx.shape[1] == 2 and src_image.shape == tgt_image.shape

    h, w = src_image.shape[:2]
    image_concat = np.concatenate([src_image, tgt_image], axis=1)
    plt.imshow(image_concat)

    for n in range(src_idx.shape[0]):
        (i, j) = src_idx[n]
        u = reproj_uv[i, j, 0]
        v = reproj_uv[i, j, 1]
        if not (u < 0 or u >= w or v < 0 or v >= h):
            plt.plot([j, w + u], [i, v], 'r-')
    plt.show()

class TokenWeightCalculator_DTU:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.loader = DTUScanLoader(
            dtu_mvs_root=args.dtu_mvs_root,
            scan_id=args.scan_id,
            num_views=args.num_views,
            step=args.step,
            device=args.device, 
            target_size=args.target_size,
        )

    def calculate_token_weight(self, src_view_idx: int, tgt_view_idx: int) -> np.ndarray:
        image_0, depth_0, mask_0, world_coords_0, intrinsic_0, extrinsic_0 = process_dtu_view(loader, args.src_view_idx)
        image_1, depth_1, mask_1, world_coords_1, intrinsic_1, extrinsic_1 = process_dtu_view(loader, args.tgt_view_idx)

        reproj_uv_0 = reproj_world_coords_to_image(world_coords_0, intrinsic_1, extrinsic_1)
        reproj_uv_int = reproj_uv_0.astype(np.int32)
        
        valid_mask = (
            (reproj_uv_0[..., 0] >= 0) & (reproj_uv_0[..., 0] < image_1.shape[1]) &
            (reproj_uv_0[..., 1] >= 0) & (reproj_uv_0[..., 1] < image_1.shape[0])
        ).astype(np.float32)

        
        points_1_on_image_0_coordinates = world_coords_1 @ extrinsic_0[:3, :3].T + extrinsic_0[:3, 3]
        points_0_on_image_0_coordinates = world_coords_0 @ extrinsic_0[:3, :3].T + extrinsic_0[:3, 3]

        depth_1_on_image_0_coordinates = np.zeros_like(depth_0)
        depth_0_on_image_0_coordinates = np.zeros_like(depth_0)

        u = reproj_uv_int[..., 0]
        v = reproj_uv_int[..., 1]
        mask_valid = valid_mask > 0

        depth_0_on_image_0_coordinates[mask_valid] = points_0_on_image_0_coordinates[mask_valid, 2]
        depth_1_on_image_0_coordinates[mask_valid] = points_1_on_image_0_coordinates[v[mask_valid], u[mask_valid], 2]

        both_seen = np.zeros_like(depth_0, dtype=bool)
        both_seen[mask_valid] = (depth_0[mask_valid] - depth_1_on_image_0_coordinates[mask_valid]) > -5.0
        both_seen[mask_valid] = (depth_0[mask_valid] - depth_1_on_image_0_coordinates[mask_valid]) < 5.0
        
        src_patch_match_score = np.zeros([target_size[1] // args.patch_size, target_size[0] // args.patch_size])

        for i in range(target_size[1] // args.patch_size):
            for j in range(target_size[0] // args.patch_size):
                patch_both_seen = both_seen[i*args.patch_size:(i+1)*args.patch_size, j*args.patch_size:(j+1)*args.patch_size]
                src_patch_match_score[i, j] = patch_both_seen.sum() / patch_both_seen.size
        
        return src_patch_match_score.reshape(-1)

def main() -> None:
    import argparse
    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description=(
                "Generate a (N+1, M+1) correspondence matrix for two DTU images after "
                "splitting them into a grid of patches."
            )
        )
        parser.add_argument("--dtu_mvs_root", type=str, required=True, help="Path to DTU MVS dataset root")
        parser.add_argument("--scan_id", type=int, required=True, help="DTU scan ID")
        parser.add_argument("--src_view_idx", type=int, required=True, help="Source view index")
        parser.add_argument("--tgt_view_idx", type=int, required=True, help="Target view index")
        parser.add_argument("--patch_size", type=int, default=14, help="Number of patches per dimension")
        parser.add_argument("--target_size", type=tuple, default=(518, 350), help="Optional target image size (width, height).")
        return parser.parse_args()


    args = parse_args()

    # 使用 DTUScanLoader 加载图像
    target_size = tuple(args.target_size) if args.target_size else None
    loader = DTUScanLoader(
        dtu_mvs_root=args.dtu_mvs_root,
        scan_id=args.scan_id,
        num_views=None,  # 加载所有视图
        step=1,
        device=None,  # 返回 numpy 数组
        target_size=target_size,
    )


    image_0, depth_0, mask_0, world_coords_0, intrinsic_0, extrinsic_0 = process_dtu_view(loader, args.src_view_idx)
    image_1, depth_1, mask_1, world_coords_1, intrinsic_1, extrinsic_1 = process_dtu_view(loader, args.tgt_view_idx)

    reproj_uv_0 = reproj_world_coords_to_image(world_coords_0, intrinsic_1, extrinsic_1)
    reproj_uv_int = reproj_uv_0.astype(np.int32)
    
    valid_mask = (
        (reproj_uv_0[..., 0] >= 0) & (reproj_uv_0[..., 0] < image_1.shape[1]) &
        (reproj_uv_0[..., 1] >= 0) & (reproj_uv_0[..., 1] < image_1.shape[0])
    ).astype(np.float32)

    
    points_1_on_image_0_coordinates = world_coords_1 @ extrinsic_0[:3, :3].T + extrinsic_0[:3, 3]
    points_0_on_image_0_coordinates = world_coords_0 @ extrinsic_0[:3, :3].T + extrinsic_0[:3, 3]

    depth_1_on_image_0_coordinates = np.zeros_like(depth_0)
    depth_0_on_image_0_coordinates = np.zeros_like(depth_0)

    u = reproj_uv_int[..., 0]
    v = reproj_uv_int[..., 1]
    mask_valid = valid_mask > 0

    depth_0_on_image_0_coordinates[mask_valid] = points_0_on_image_0_coordinates[mask_valid, 2]
    depth_1_on_image_0_coordinates[mask_valid] = points_1_on_image_0_coordinates[v[mask_valid], u[mask_valid], 2]

    # aaa = np.concatenate([depth_0_on_image_0_coordinates, depth_1_on_image_0_coordinates], axis=1)
    # plt.imshow(depth_0_on_image_0_coordinates - depth_1_on_image_0_coordinates)
    # plt.title("Depth 0 on image 0 coordinates and Depth 1 on image 0 coordinates")
    # plt.show()


    # seen0_not_seen1 = np.zeros_like(depth_0, dtype=bool)
    # seen0_not_seen1 = ((depth_0 - depth_1_on_image_0_coordinates ) < -5.0)
    # seen0_not_seen1 |= ((depth_0 > 0) & (depth_1_on_image_0_coordinates < 1e-6))

    both_seen = np.zeros_like(depth_0, dtype=bool)
    both_seen[mask_valid] = (depth_0[mask_valid] - depth_1_on_image_0_coordinates[mask_valid]) > -5.0
    both_seen[mask_valid] = (depth_0[mask_valid] - depth_1_on_image_0_coordinates[mask_valid]) < 5.0
    
    both_seen_image = np.concatenate([image_0, image_1, image_0 * (both_seen[..., None])], axis=1)
    plt.imshow(both_seen_image)
    plt.title("Both seen image")
    plt.show()

    src_patch_match_score = np.zeros([target_size[1] // args.patch_size, target_size[0] // args.patch_size])
    print(src_patch_match_score.shape)

    for i in range(target_size[1] // args.patch_size):
        for j in range(target_size[0] // args.patch_size):
            patch_both_seen = both_seen[i*args.patch_size:(i+1)*args.patch_size, j*args.patch_size:(j+1)*args.patch_size]
            src_patch_match_score[i, j] = patch_both_seen.sum() / patch_both_seen.size
    
    plt.imshow(src_patch_match_score)
    plt.title("Src patch match score")
    plt.show()

    # import open3d as o3d
    
    # pcd_0 = o3d.geometry.PointCloud()
    # pcd_0.points = o3d.utility.Vector3dVector(points_1_on_image_0_coordinates.reshape(-1, 3))
    # color = np.array([1, 0, 0]).reshape(-1, 3).repeat(points_1_on_image_0_coordinates.reshape(-1, 3).shape[0], axis=0)
    # pcd_0.colors = o3d.utility.Vector3dVector(color)

    # pcd_1 = o3d.geometry.PointCloud()
    # pcd_1.points = o3d.utility.Vector3dVector(points_0_on_image_0_coordinates.reshape(-1, 3))
    # color = np.array([0, 0, 1]).reshape(-1, 3).repeat(points_0_on_image_0_coordinates.reshape(-1, 3).shape[0], axis=0)
    # pcd_1.colors = o3d.utility.Vector3dVector(color)

    # o3d.visualization.draw_geometries([pcd_0, pcd_1])
    # sys.exit()

if __name__ == "__main__":
    main()