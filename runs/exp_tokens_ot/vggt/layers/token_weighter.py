from __future__ import annotations

import cv2
from typing import Any, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

from data.dtu_loader import DTUScanLoader
from vggt.utils.geometry import depth_to_world_coords_points
from vggt.dependency.projection import project_3D_points_np

from vggt.layers.attention_knockout import *
from merging.merge import token_merge_bipartite2d

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

    return reproj_uv[0]

def apply_heatmap(image, heatmap, alpha=0.4):
    """
    :param image: 原始图像，shape (H, W, 3)
    :param heatmap: 注意力图，shape (H, W)，值范围 [0, 1]
    :param alpha: 热图透明度
    """
    # 确保图像数据类型一致
    image = (image - image.min()) / (image.max() - image.min())
    image = 255 * image
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = 255 * heatmap
    if heatmap.dtype != np.uint8:
        heatmap = heatmap.astype(np.uint8)
    
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB) if len(heatmap.shape)==2 else heatmap
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = heatmap_colored[..., [2, 1, 0]]
    
    # 如果图像不是 uint8 类型，进行转换
    if heatmap_colored.dtype != np.uint8:
        heatmap_colored = heatmap_colored.astype(np.uint8)
    
    output = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return output
        
@dataclass
class TokenFusionStrategyConfig:
    dtu_mvs_root: str = "placeholder"
    scan_id: int = 1
    num_views: int = 1
    step: int = 1
    device: str = "cpu"
    target_size: Tuple[int, int] = (518, 350)
    patch_size: int = 14
    special_tokens_num: int = 5
    # for attention knockout
    knockout_layer_idx: list[int] = field(default_factory=lambda: [-1])
    knockout_method: str = "random" # "random" or "visible_score" or "top_k"
    knockout_random_ratio: float = 0.5
    knockout_top_k: int = 100
    # for token merge
    enable_token_merge: bool = False
    token_merge_ratio: float = 0.5
    sx: int = 5
    sy: int = 5
    no_rand: bool = False
    enable_protection: bool = True
    # debug options
    calculate_rope_gain_ratio: bool = False
    calculate_token_cos_similarity: bool = False
    display_attn_map_after_softmax: bool = False
    calculate_top_k_dominance: bool = False

class TokenFusionStrategy:
    def __init__(self, args):
        self.args = args
        # 支持字典和对象两种参数类型
        def get_arg(key, default=None):
            if isinstance(args, dict):
                return args.get(key, default)
            else:
                return getattr(args, key, default)
        
        self.loader = DTUScanLoader(
            dtu_mvs_root=get_arg('dtu_mvs_root'),
            scan_id=get_arg('scan_id'),
            num_views=get_arg('num_views'),
            step=get_arg('step'),
            device=get_arg('device'), 
            target_size=get_arg('target_size'),
        )
        # 保存常用参数
        self.num_views = get_arg('num_views')
        self.target_size = get_arg('target_size')
        self.width = self.target_size[0]
        self.height = self.target_size[1]
        self.patch_size = get_arg('patch_size')
        self.special_tokens_num = get_arg('special_tokens_num')

        self.num_tokens_per_image_on_width = self.width // self.patch_size
        self.num_tokens_per_image_on_height = self.height // self.patch_size
        self.num_tokens_per_image = self.num_tokens_per_image_on_width * self.num_tokens_per_image_on_height

        # 加载数据
        self.images = []
        self.depths = []    
        self.masks = []
        self.world_coords = []
        self.intrinsics = []
        self.extrinsics = []
        for i in range(get_arg('num_views')):
            image_i, depth_i, mask_i, world_coords_i, intrinsic_i, extrinsic_i = process_dtu_view(self.loader, i)
            self.images.append(image_i)
            self.depths.append(depth_i)
            self.masks.append(mask_i)
            self.world_coords.append(world_coords_i)
            self.intrinsics.append(intrinsic_i)
            self.extrinsics.append(extrinsic_i)
        
        # for attention knockout
        self.knockout_layer_idx = get_arg('knockout_layer_idx')
        self.knockout_method = get_arg('knockout_method')
        # for random knockout
        self.knockout_random_ratio = get_arg('knockout_random_ratio')
        # visible score mask matrix
        self.visible_scores = torch.tensor(np.array(self.calculate_visible_score_all()))
        # for top-k preserved knockout      
        self.knockout_top_k = get_arg('knockout_top_k')

        self.enable_token_merge = get_arg('enable_token_merge')
        self.token_merge_ratio = get_arg('token_merge_ratio')
        self.sx = get_arg('sx')
        self.sy = get_arg('sy')
        self.no_rand = get_arg('no_rand')
        self.enable_protection = get_arg('enable_protection')
        self.merge_fn, self.unmerge_fn = None, None
        self.unm_idx, self.src_idx, self.dst_idx, self.a_idx, self.b_idx, self.protected_idx = None, None, None, None, None, None

        # for observe token sparsity
        self.calculate_token_cos_similarity = get_arg('calculate_token_cos_similarity')
        self.x_cos_similarity = []
        self.tokens_erank_kernel_norm = []
        self.tokens_erank_fro_norm = []

        # for rope gain calculation 
        self.calculate_rope_gain_ratio = get_arg('calculate_rope_gain_ratio')
        self.q_original = []
        self.q_rope_gain = []

        # for top-k dominance of global attention after softmax calculation
        self.calc_top_k_dominance = get_arg('calculate_top_k_dominance')
        self.top_k_dominance = []

        # for attention rollout calculation
        self.attention_map_rollout = None

        # for percise corresponding attention knockout
        self.corres_masks = self.calculate_corresponding_attention_mask()

        # for display attention map after softmax
        self.display_attn_map_after_softmax = get_arg('display_attn_map_after_softmax')
        self.attention_of_all_heads = []

        self.token_cosine_similarity = []


    def rope(self, rope, q, k, pos):
        """
        给rope的计算套了一层壳， 为了把计算rope gain的逻辑封装起来
        """
        def calculate_rope_gain(q_original: Tensor, q_rope: Tensor):
            norm_q_diff = torch.norm(q_rope - q_original, p=2, dim=-1).mean()
            norm_q_original = torch.norm(q_original, p=2, dim=-1).mean()
            norm_q_diff_ratio = norm_q_diff / norm_q_original

            self.q_original.append(norm_q_original.item())
            self.q_rope_gain.append(norm_q_diff_ratio.item())

        if self.calculate_rope_gain_ratio:
            q_original = q.clone()

        q = rope(q, pos)
        k = rope(k, pos)

        if self.calculate_rope_gain_ratio:
            calculate_rope_gain(q_original, q)
            del q_original
        return q, k

    def inspect_token_similarity(self, x: Tensor):
        if not self.calculate_token_cos_similarity:
            return
        def calculate_token_cos_similarity(x: Tensor):
            def matrix_norms(tokens):
                U, S, Vh = torch.linalg.svd(tokens, full_matrices=False)  # S shape: (B, min(N,C))
                spectral = S[..., 0]
                knl = S.sum(dim=-1)
                fro = torch.linalg.norm(tokens)  # 更快，不必用奇异值
                return spectral, knl, fro

            D = x.shape[-1]
            x_normed = x.reshape(-1, D).clone()
            N = x_normed.shape[0]

            spectral, knl, fro = matrix_norms(x_normed)
            x_normed /= x_normed.norm(dim=-1, keepdim=True)
            # cos_sim = torch.abs(torch.mm(x_normed, x_normed.t())) # 这里取不取abs有一些影响但是不大
            cos_sim = torch.mm(x_normed, x_normed.t())
            # plt.imshow(cos_sim.cpu().numpy())
            # plt.colorbar()
            # plt.show()
            cos_sim_mean = cos_sim.mean().item()

            self.tokens_erank_kernel_norm.append(knl.item() / (N *spectral.item()))
            self.tokens_erank_fro_norm.append(fro.item() / (N * spectral.item()))
            self.x_cos_similarity.append(cos_sim_mean)
            del x_normed, cos_sim

        def visualize_token_similarity_heatmap(x:Tensor, token_idx: int, image_idx: int):
            D = x.shape[-1]
            x_normed = x.reshape(-1, D).clone() 
            x_normed /= x_normed.norm(dim=-1, keepdim=True)
            cos_sim = torch.mm(x_normed, x_normed.t())
            cos_sim_fp32 = cos_sim.float()

            random_token_idx = np.random.uniform(0, self.num_tokens_per_image + self.special_tokens_num, size=(50,)).astype(np.int32)

            mean_similarity = np.zeros(len(image_idx), dtype=np.float32)
            for token_idx_i in random_token_idx:
                for image_idx_i in image_idx:
                    similarity_heatmap = cos_sim_fp32[token_idx_i, image_idx_i * (self.num_tokens_per_image + 5) : (image_idx_i + 1) * (self.num_tokens_per_image + 5)][5:].cpu().numpy().reshape(self.num_tokens_per_image_on_height, self.num_tokens_per_image_on_width)
                    mean_similarity[image_idx_i] = mean_similarity[image_idx_i] + similarity_heatmap.max().item()

            print(f"mean similarity: {mean_similarity}")

            similarity_heatmaps = []
            for token_idx_i in token_idx:
                for image_idx_i in image_idx:
                    similarity_heatmap = cos_sim_fp32[token_idx_i, image_idx_i * (self.num_tokens_per_image + 5) : (image_idx_i + 1) * (self.num_tokens_per_image + 5)][5:].cpu().numpy().reshape(self.num_tokens_per_image_on_height, self.num_tokens_per_image_on_width)
                    similarity_heatmaps.append(similarity_heatmap)

            # plot by grid concat
            grid_img = np.concatenate([np.concatenate(similarity_heatmaps[i*len(image_idx):(i+1)*len(image_idx)], axis=1) for i in range(len(token_idx))], axis=1)
            self.token_cosine_similarity.append(grid_img)

        calculate_token_cos_similarity(x)
        visualize_token_similarity_heatmap(x, [491+5, 794+5], [0, 1, 2, 3])

    def calculate_top_k_dominance(self, attn_map: Tensor, threshold: float = 0.9):
        if not self.calc_top_k_dominance:
            return
        def compute_topk_dominance(attention_map, threshold=0.9):
            """
            attention_map: [1, H, N, K] 或 [H, N, K]
            只在 CPU 上做统计，避免占用 GPU 显存
            """
            # 全程不需要梯度
            with torch.no_grad():
                attn = attention_map.squeeze(0)      # [H, N, K]
                # 注释掉这一行以分开计算每个head的top-k dominance #########
                # attn = attn.sum(dim=0).unsqueeze(0)  # [1, N, K]
                # attn = attn.mean(dim=0).unsqueeze(0)  # [1, N, K]
                ######################################################
                H, N, K = attn.shape

                attn = attn.detach().to("cpu", dtype=torch.float32)  # 如果 K 很大，可以改成 float16

                k_per_query = torch.zeros(H, N, dtype=torch.float32)

                for h in range(H):
                    attn_h = attn[h]  # [N, K]
                    sorted_scores, _ = torch.sort(attn_h, dim=-1, descending=True)  # [N, K]
                    cumsum = torch.cumsum(sorted_scores, dim=-1)                    # [N, K]
                    mask = cumsum >= threshold                                      # [N, K] bool

                    has_true = mask.any(dim=-1)                                     # [N]
                    first_idx = mask.float().argmax(dim=-1)                         # [N]

                    k_per_query[h] = torch.where(
                        has_true,
                        first_idx + 1,
                        torch.full_like(first_idx, fill_value=K)
                    ).float()

                k_mean = (k_per_query.mean() / K).item()
                return k_mean, k_per_query
                
        k_mean, k_per_query = compute_topk_dominance(attn_map, threshold)
        self.top_k_dominance.append(k_mean)
        return k_mean, k_per_query

    def visualize_attn_map(self, attn_map: Tensor, token_idx: list[int], image_idx: list[int], head=None):
        """
            这是一个实验函数，用来绘制 token_idx 对于 image_idx 的注意力图

            Usage:
                # FastVGGT 实验中的图是softmax之前的， 用来观察attention collapse现象
                # print(f"attn map before softmax of layer {idx}")
                # self.token_weighter.visualize_attn_map(attn_before_softmax, [0, 500, 930, 1430], [0, 1, 2])

                # 我的图是softmax之后的，更容易观察到注意力中的匹配现象
                # print(f"attn map after softmax of layer {idx}")
                # self.token_weighter.visualize_attn_map(attn_after_softmax, [0, 500, 930, 1430], [0, 1, 2])

        """
        def get_attn_map(attn_map: Tensor, token_idx: int, image_idx: int):
            if token_idx > self.num_views * (self.num_tokens_per_image + self.special_tokens_num) or image_idx > self.num_views - 1:
                print(f"token_idx: {token_idx} or image_idx: {image_idx} is out of range")
                return np.zeros((self.num_tokens_per_image_on_height, self.num_tokens_per_image_on_width ))
            if head is None:
                target_attn_map_original = attn_map.squeeze(0).mean(0)
            else:
                target_attn_map_original = attn_map.squeeze(0)[head]

            # target_attn_map_original = attn_map.squeeze(0).sum(dim=0)
            target_attn = target_attn_map_original[token_idx, image_idx * 930 : (image_idx + 1) * 930]
            target_attn_map = target_attn[5:].cpu().numpy().reshape(25, 37)
            del target_attn_map_original, target_attn
            return target_attn_map.astype(np.float32)

        target_attn_maps = []
        for image_idx_i in image_idx:
            for token_idx_i in token_idx:
                target_attn_map = get_attn_map(attn_map, token_idx_i, image_idx_i)
                target_attn_map = cv2.resize(target_attn_map, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                blended_image = apply_heatmap(self.images[image_idx_i], target_attn_map, alpha = 0.8)
                target_attn_maps.append(blended_image)

        # plot by grid concat
        grid_img = np.concatenate([np.concatenate(target_attn_maps[i*len(token_idx):(i+1)*len(token_idx)], axis=0) for i in range(len(image_idx))], axis=1)
        plt.imshow(grid_img)
        plt.colorbar()
        plt.show()
        return target_attn_maps


    def visualize_attn_map_all_heads(self, attn_map: Tensor, token_idx: int, image_idx: int):
        """
            这是一个实验函数，用来绘制 token_idx 对于 image_idx 的注意力图

            Usage:
                # FastVGGT 实验中的图是softmax之前的， 用来观察attention collapse现象
                # print(f"attn map before softmax of layer {idx}")
                # self.token_weighter.visualize_attn_map(attn_before_softmax, [0, 500, 930, 1430], [0, 1, 2])

                # 我的图是softmax之后的，更容易观察到注意力中的匹配现象
                # print(f"attn map after softmax of layer {idx}")
                # self.token_weighter.visualize_attn_map(attn_after_softmax, [0, 500, 930, 1430], [0, 1, 2])

        """
        if self.enable_token_merge or not self.display_attn_map_after_softmax:
            return
        def get_attn_map(attn_map: Tensor, token_idx: int, image_idx: int, head: int):
            if token_idx > self.num_views * (self.num_tokens_per_image + self.special_tokens_num) or image_idx > self.num_views - 1:
                print(f"token_idx: {token_idx} or image_idx: {image_idx} is out of range")
                return np.zeros((self.num_tokens_per_image_on_height, self.num_tokens_per_image_on_width ))
            if head is not None:
                target_attn_map_original = attn_map.squeeze(0)[head]
            else:
                target_attn_map_original = attn_map.squeeze(0).mean(0)

            # target_attn_map_original = attn_map.squeeze(0).sum(dim=0)
            target_attn = target_attn_map_original[token_idx, image_idx * (self.num_tokens_per_image + 5) : (image_idx + 1) * (self.num_tokens_per_image + 5)]
            target_attn_map = target_attn[5:].cpu().numpy().reshape(self.num_tokens_per_image_on_height, self.num_tokens_per_image_on_width)
            del target_attn_map_original, target_attn
            return target_attn_map.astype(np.float32)

        target_attn_maps = []
        for head in range(16):
            target_attn_map = get_attn_map(attn_map, token_idx, image_idx, head)
            target_attn_map = cv2.resize(target_attn_map, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            blended_image = apply_heatmap(self.images[image_idx], target_attn_map, alpha = 0.8)
            target_attn_maps.append(blended_image)
        
        target_attn_map = get_attn_map(attn_map, token_idx, image_idx, None)
        target_attn_map = cv2.resize(target_attn_map, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        blended_image = apply_heatmap(self.images[image_idx], target_attn_map, alpha = 0.8)
        target_attn_maps.append(blended_image)

        # plot by grid concat
        grid_img = np.concatenate(target_attn_maps, axis=1)
        self.attention_of_all_heads.append(grid_img)

    def calculate_visible_mask(self, attn_map: Tensor, src_patch_idx: Tuple[int, int], src_view_idx: int, tgt_view_idx: int) -> np.ndarray:
        image_0, depth_0, mask_0, world_coords_0, intrinsic_0, extrinsic_0 = self.images[src_view_idx], self.depths[src_view_idx], self.masks[src_view_idx], self.world_coords[src_view_idx], self.intrinsics[src_view_idx], self.extrinsics[src_view_idx]
        image_1, depth_1, mask_1, world_coords_1, intrinsic_1, extrinsic_1 = self.images[tgt_view_idx], self.depths[tgt_view_idx], self.masks[tgt_view_idx], self.world_coords[tgt_view_idx], self.intrinsics[tgt_view_idx], self.extrinsics[tgt_view_idx]

        patch_row, patch_col = src_patch_idx

        num_row_patch, num_col_patch = 25, 37

        row_idx = np.arange(patch_row * 14, (patch_row + num_row_patch) * 14)
        col_idx = np.arange(patch_col * 14, (patch_col + num_col_patch) * 14)

        xx, yy = np.meshgrid(col_idx, row_idx)
        src_pnts_uv = np.stack([xx, yy], axis=-1)

        src_image_crop = image_0[src_pnts_uv[..., 1], src_pnts_uv[..., 0]]
        world_coords_crop = world_coords_0[src_pnts_uv[..., 1], src_pnts_uv[..., 0]]
        reproj_uv_crop = reproj_world_coords_to_image(world_coords_crop, intrinsic_1, extrinsic_1).reshape(world_coords_crop.shape[0], world_coords_crop.shape[1], 2)
        reproj_uv_crop_int = reproj_uv_crop.astype(np.int32)

        valid_mask = (
            (reproj_uv_crop_int[..., 0] >= 0) & (reproj_uv_crop_int[..., 0] < image_1.shape[1]) &
            (reproj_uv_crop_int[..., 1] >= 0) & (reproj_uv_crop_int[..., 1] < image_1.shape[0])
        ).astype(np.float32)

        # print(f"reproj_uv_crop: {reproj_uv_crop_int}")

        # print(f"valid_mask.shape: {valid_mask.shape}")

        # points_1_on_image_0_coordinates = world_coords_1 @ extrinsic_0[:3, :3].T + extrinsic_0[:3, 3]
        # depth_1_on_image_0_coordinates = np.zeros_like(depth_0)

        u = reproj_uv_crop_int[..., 0]
        v = reproj_uv_crop_int[..., 1]
        mask_valid = valid_mask > 0

        patch_idx = (v[mask_valid] // self.patch_size, u[mask_valid] // self.patch_size)

        attn_map_col_idx = tgt_view_idx * 930 + 5 + patch_idx[0] * 14 + patch_idx[1]
        attn_map_row_idx = np.ones_like(attn_map_col_idx) * (src_view_idx * 930 + 5 + patch_row * 14 + patch_col)

        attn_map_row_idx = attn_map_row_idx.astype(np.int32).reshape(-1)
        attn_map_col_idx = attn_map_col_idx.astype(np.int32).reshape(-1)

        points = world_coords_0[src_pnts_uv[..., 1][mask_valid], src_pnts_uv[..., 0][mask_valid]]
        # points 形状可能是 (H, W, 3)，这里统一展平为 (N, 3)
        points = points.reshape(-1, 3)

        # gray ∈ [0, 1]，根据 gray 的值使用 jet colormap 映射到 RGB：
        # 0 → 蓝色，1 → 红色，中间值为绿色/黄色等
        gray = attn_map[:, :, attn_map_row_idx, attn_map_col_idx].mean(dim=0).cpu().numpy()
        gray = gray.reshape(-1)                      # (N,)
        gray = np.clip(gray, 0.0, 1.0)               # 保证在 [0, 1]

        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        cmap = plt.cm.get_cmap('jet')
        colors = cmap(gray)[:, :3]                   # (N, 4) → (N, 3)，丢弃 alpha
        colors = colors.astype(np.float32)

        # 保证颜色和点的数量一致（一般应当相同，只是这里做个安全裁剪）
        n = min(points.shape[0], colors.shape[0])
        colors = colors[:n]
        points = points[:n]
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([pcd])
        plt.show()



        # reproj_mask = np.zeros_like(image_1)
        # reproj_mask[v[mask_valid], u[mask_valid]] = 255
        # plt.imshow(reproj_mask)
        # plt.show()

    def calculate_corresponding_attention_mask(self) -> list[list[int]]:
        stride = 5
        def calc_corres_attn_mask_token_i(img_idx: int, row_idx: int, col_idx: int):
            patch_center_uv = (col_idx * 14 + 7, row_idx * 14 + 7)
            patch_center_world_coords = self.world_coords[img_idx][patch_center_uv[1], patch_center_uv[0]]
            valid_u = range(max(0, col_idx - stride), min(self.num_tokens_per_image_on_width - 1, col_idx + stride))
            valid_v = range(max(0, row_idx - stride), min(self.num_tokens_per_image_on_height - 1, row_idx + stride))
            valid_token_idx = []
            for u in valid_u:
                for v in valid_v:
                    valid_token_idx.append(img_idx * (self.num_tokens_per_image + self.special_tokens_num) + self.special_tokens_num + v * self.num_tokens_per_image_on_width + u)

            for i in range(self.num_views):
                if i == img_idx:
                    continue
                patch_center_reproj_uv = reproj_world_coords_to_image(patch_center_world_coords, self.intrinsics[i], self.extrinsics[i]).astype(np.int32)
                u, v = patch_center_reproj_uv[0]
                if(u < 0 or u >= self.width or v < 0 or v >= self.height):
                    continue
                if(np.linalg.norm(patch_center_world_coords - self.world_coords[i][v, u]) > 20):
                    continue
                patch_u = u // self.patch_size
                patch_v = v // self.patch_size
                valid_u = range(max(0, patch_u - stride), min(self.num_tokens_per_image_on_width - 1, patch_u + stride))
                valid_v = range(max(0, patch_v - stride), min(self.num_tokens_per_image_on_height - 1, patch_v + stride))
                valid_token_idx.append(i * (self.num_tokens_per_image + self.special_tokens_num) + self.special_tokens_num + patch_v * self.num_tokens_per_image_on_width + patch_u)
                for u in valid_u:
                    for v in valid_v:
                        valid_token_idx.append(i * (self.num_tokens_per_image + self.special_tokens_num) + self.special_tokens_num + v * self.num_tokens_per_image_on_width + u)

            return valid_token_idx


        corres_masks = []
        for i in range(self.num_views):
            for r in range(self.num_tokens_per_image_on_height):
                for c in range(self.num_tokens_per_image_on_width):
                    corres_masks.append(calc_corres_attn_mask_token_i(i, r, c))
        return corres_masks
       

    def calculate_visible_score(self, src_view_idx: int, tgt_view_idx: int) -> np.ndarray:
        image_0, depth_0, mask_0, world_coords_0, intrinsic_0, extrinsic_0 = self.images[src_view_idx], self.depths[src_view_idx], self.masks[src_view_idx], self.world_coords[src_view_idx], self.intrinsics[src_view_idx], self.extrinsics[src_view_idx]
        image_1, depth_1, mask_1, world_coords_1, intrinsic_1, extrinsic_1 = self.images[tgt_view_idx], self.depths[tgt_view_idx], self.masks[tgt_view_idx], self.world_coords[tgt_view_idx], self.intrinsics[tgt_view_idx], self.extrinsics[tgt_view_idx]

        reproj_uv_0 = reproj_world_coords_to_image(world_coords_0, intrinsic_1, extrinsic_1).reshape(world_coords_0.shape[0], world_coords_0.shape[1], 2)
        reproj_uv_int = reproj_uv_0.astype(np.int32)
        
        valid_mask = (
            (reproj_uv_0[..., 0] >= 0) & (reproj_uv_0[..., 0] < image_1.shape[1]) &
            (reproj_uv_0[..., 1] >= 0) & (reproj_uv_0[..., 1] < image_1.shape[0])
        ).astype(np.float32)

        
        points_1_on_image_0_coordinates = world_coords_1 @ extrinsic_0[:3, :3].T + extrinsic_0[:3, 3]

        depth_1_on_image_0_coordinates = np.zeros_like(depth_0)

        u = reproj_uv_int[..., 0]
        v = reproj_uv_int[..., 1]
        mask_valid = valid_mask > 0
        depth_1_on_image_0_coordinates[mask_valid] = points_1_on_image_0_coordinates[v[mask_valid], u[mask_valid], 2]

        both_seen = np.zeros_like(depth_0, dtype=bool)
        both_seen[mask_valid] = (depth_0[mask_valid] - depth_1_on_image_0_coordinates[mask_valid]) > -5.0
        both_seen[mask_valid] = (depth_0[mask_valid] - depth_1_on_image_0_coordinates[mask_valid]) < 5.0
        
        # 从图像尺寸或 self.target_size 获取目标尺寸
        if self.target_size is not None:
            target_size = tuple(self.target_size) if isinstance(self.target_size, (list, tuple)) else self.target_size
            target_w, target_h = target_size[0], target_size[1]
        else:
            # 从图像尺寸推断
            target_h, target_w = image_0.shape[:2]
        
        src_patch_match_score = np.zeros([target_h // self.patch_size, target_w // self.patch_size])

        for i in range(target_h // self.patch_size):
            for j in range(target_w // self.patch_size):
                patch_both_seen = both_seen[i*self.patch_size:(i+1)*self.patch_size, j*self.patch_size:(j+1)*self.patch_size]
                src_patch_match_score[i, j] = patch_both_seen.sum() / patch_both_seen.size
        return src_patch_match_score.reshape(-1)
    
    def calculate_visible_score_all(self) -> np.ndarray:
        visible_scores = []
        for i in range(self.num_views):
            for j in range(self.num_views):
                visible_scores.append(self.calculate_visible_score(i, j))
        return visible_scores

    def attention_knockout(self, attn_map: Tensor, curr_layer_idx: int) -> Tensor:
        if self.enable_token_merge:
            return attn_map
        if curr_layer_idx not in self.knockout_layer_idx and -1 not in self.knockout_layer_idx:
            return attn_map

        # fill_value = float('-inf')
        fill_value = 0.0
        if self.knockout_method == "random":
            print("random knockout for layer: ", curr_layer_idx)
            return random_knockout(attn_map, self.num_views, self.num_tokens_per_image, self.width // self.patch_size, self.height // self.patch_size, self.knockout_random_ratio, fill_value)
        elif self.knockout_method == "visible_score":
            print("visible score knockout for layer: ", curr_layer_idx)
            return visible_score_knockout(attn_map, self.num_views, self.num_tokens_per_image, self.width // self.patch_size, self.height // self.patch_size, self.visible_scores, fill_value)
        elif self.knockout_method == "top_k":
            print("top-k preserved knockout for layer: ", curr_layer_idx)
            return top_k_preserved_knockout(attn_map, self.num_views, self.num_tokens_per_image, self.width // self.patch_size, self.height // self.patch_size, self.knockout_top_k, fill_value)
        elif self.knockout_method == "corres_mask":
            print("corres mask knockout for layer: ", curr_layer_idx)
            return corres_mask_knockout(attn_map, self.num_views, self.num_tokens_per_image, self.width // self.patch_size, self.height // self.patch_size, self.corres_masks, fill_value)
        else:
            raise ValueError(f"Invalid knockout method: {self.knockout_method}")

    def token_merge_FastVGGT(self, x, q, k, v):
        if not self.enable_token_merge:
            return q, k, v
        generator = torch.Generator(device=x.device)
        generator.manual_seed(33)

        r = int(x.shape[1] * self.token_merge_ratio)

        (self.merge_fn, self.unmerge_fn, self.unm_idx, self.src_idx, self.dst_idx, self.a_idx, self.b_idx, self.protected_idx) = token_merge_bipartite2d(
            x,
            self.num_tokens_per_image_on_width,
            self.num_tokens_per_image_on_height,
            self.sx,
            self.sy,
            r,
            no_rand=self.no_rand,
            generator=generator,
            enable_protection=self.enable_protection,
        )

        B_q, H_q, N_q, D_q = q.shape

        q_merge_in = q.permute(0, 2, 1, 3).reshape(B_q, N_q, H_q * D_q) # (B, N, H * D)
        k_merge_in = k.permute(0, 2, 1, 3).reshape(B_q, N_q, H_q * D_q)
        v_merge_in = v.permute(0, 2, 1, 3).reshape(B_q, N_q, H_q * D_q)

        q_out, k_out, v_out = self.merge_fn(
            q_merge_in,
            mode="mean",
            extra_tensors=k_merge_in,
            extra_tensors_2=v_merge_in,
        )

        del q_merge_in, k_merge_in, v_merge_in

        N_m = q_out.shape[1]
        q = q_out.reshape(B_q, N_m, H_q, D_q).permute(0, 2, 1, 3)
        k = k_out.reshape(B_q, N_m, H_q, D_q).permute(0, 2, 1, 3)
        v = v_out.reshape(B_q, N_m, H_q, D_q).permute(0, 2, 1, 3)

        del q_out, k_out, v_out

        return q, k, v

    def token_unmerge_FastVGGT(self, x):
        if self.unmerge_fn is None:
            return x
        return self.unmerge_fn(x)

    def who_are_merged(self):
        if self.src_idx is None:
            pass
        N = (self.num_tokens_per_image + self.special_tokens_num) * self.num_views
        merged_map = torch.zeros(N)

        merged_idx = torch.gather(self.a_idx, dim=1, index=self.src_idx)[0, :, 0]
        # merged_idx = torch.gather(self.b_idx, dim=1, index=self.dst_idx)[0, :, 0]
        merged_map[merged_idx.cpu()] = 1
        merge_map_display = []
        for i in range(self.num_views):
            merge_map_display.append(merged_map[self.special_tokens_num + (self.special_tokens_num + self.num_tokens_per_image) * i: (self.special_tokens_num + self.num_tokens_per_image) * (i + 1)].reshape(self.num_tokens_per_image_on_height, self.num_tokens_per_image_on_width).numpy())
        merge_map_display = np.concatenate(merge_map_display, axis=1)
        plt.imshow(merge_map_display)
        plt.title("Merged map")
        plt.show()
        return merge_map_display


if __name__ == "__main__":
    def run() -> None:
        import argparse
        
        def parse_args() -> argparse.Namespace:
            parser = argparse.ArgumentParser(
                description=(
                    "Generate a (N+1, M+1) correspondence matrix for two DTU images after "
                    "splitting them into a grid of patches."
                )
            )
            parser.add_argument("--dtu_mvs_root", type=str, default="/home/vision/ws/datasets/SampleSet/dtu_mvs", help="Path to DTU MVS dataset root")
            parser.add_argument("--scan_id", type=int, default=1, help="DTU scan ID")
            parser.add_argument("--src_view_idx", type=int, default=0, help="Source view index")
            parser.add_argument("--tgt_view_idx", type=int, default=4, help="Target view index")
            parser.add_argument("--patch_size", type=int, default=14, help="Number of patches per dimension")
            parser.add_argument("--target_size", type=tuple, default=(518, 350), help="Optional target image size (width, height).")
            parser.add_argument("--num_views", type=int, default=10, help="Number of views")
            parser.add_argument("--step", type=int, default=1, help="Step size")
            parser.add_argument("--device", type=str, default="cpu", help="Device")
            return parser.parse_args()


        args = parse_args()

        token_weight_calculator = TokenFusionStrategy(args)
        token_weight = token_weight_calculator.calculate_visible_score(args.src_view_idx, args.tgt_view_idx)

        token_weight_np = token_weight.reshape(args.target_size[1] // args.patch_size, args.target_size[0] // args.patch_size)
        plt.imshow(token_weight_np)
        plt.title("Token weight")
        plt.show()

    run()