from __future__ import annotations

import torch
import numpy as np
from typing import List
from torch import Tensor

def get_attention_map(x: Tensor, i: int, j: int, patch_row: int, patch_col: int, num_tokens_per_image_total: int, special_token_num: int, width: int, height: int):
    patch_idx = patch_row * width + patch_col
    selected_attention_map = x[:, :, num_tokens_per_image_total * i + special_token_num + patch_idx, num_tokens_per_image_total * j + special_token_num: num_tokens_per_image_total * (j+1)]
    selected_attention_map = selected_attention_map.reshape(16, height, width).cpu().numpy()
    grid_rows, grid_cols = 4, 4
    selected_attention_map_grid = selected_attention_map.reshape(grid_rows, grid_cols, height, width)
    grid_image = np.concatenate([
        np.concatenate([selected_attention_map_grid[i, j] for j in range(grid_cols)], axis=1)
        for i in range(grid_rows)
    ], axis=0)
    return grid_image

def random_knockout(x: Tensor, num_images: int, num_patches_per_image: int, width: int, height: int, ratio: float, fill_value: float = float('-inf')) -> Tensor:
    """
    Randomly knock out the attention values in the input tensor x.

    Args:
        x: Tensor, the input tensor to be knocked out.
        ratio: float, the ratio of the attention values to be knocked out (0.0 to 1.0).

    Returns:
        Tensor, the knocked out tensor.
    """
    debug_mode = False
    B, H, N, _ = x.shape 
    special_token_num = 5
    num_tokens_per_image_total = num_patches_per_image + special_token_num

    attention_map_before_knockout, attention_map_after_knockout = None, None
    if debug_mode:
        i, j = 1, 3
        patch_row, patch_col = 12, 18
        attention_map_before_knockout = get_attention_map(x, i, j, patch_row, patch_col, num_tokens_per_image_total, special_token_num, width, height)

    preserve_mask = torch.zeros(x.shape[-2:], device="cpu")
    special_token_start_idx = torch.arange(0, num_images * num_tokens_per_image_total, num_tokens_per_image_total)
    special_token_indices = (
        special_token_start_idx[:, None] + torch.arange(special_token_num)
    ).reshape(-1)
    preserve_mask[special_token_indices, :] = 1
    preserve_mask[:, special_token_indices] = 1

    can_knockout_mask = (preserve_mask == 0)
    print("attention map mean before knockout: ", x.mean().item())
    print("attention map max before knockout: ", x.max().item())
    print("attention map min before knockout: ", x.min().item())
    print("attention map sum before knockout: ", x.sum().item())

    num_can_knockout = can_knockout_mask.sum().item()
    if num_can_knockout == 0:
        print("No can knockout positions, return original tensor")
        return x
    
    num_to_knockout = int(num_can_knockout * ratio)
    if num_to_knockout == 0:
        print("No need to knockout, return original tensor")
        return x

    random_values = torch.rand_like(x, dtype=torch.float32, device="cpu")
    knockout_positions = can_knockout_mask & (random_values < ratio)
    x.masked_fill_(knockout_positions.to(x.device), fill_value)

    if debug_mode:
        attention_map_after_knockout = get_attention_map(x, i, j, patch_row, patch_col, num_tokens_per_image_total, special_token_num, width, height)
        from matplotlib import pyplot as plt
        plt.imshow(np.concatenate([attention_map_before_knockout, attention_map_after_knockout], axis=0))
        plt.colorbar()
        plt.show()
    
    print("attention map mean after knockout: ", x.mean().item())
    print("attention map sum after knockout: ", x.sum().item())
    print("knockout ratio: ", knockout_positions.sum().item() / knockout_positions.numel())
    del can_knockout_mask, random_values, knockout_positions
    return x

def visible_score_knockout(x: Tensor, num_images: int, num_patches_per_image: int, width: int, height: int,
                           visible_scores: List[Tensor], fill_value: float = float('-inf')) -> Tensor:
    """
    Knock out the attention values in the input tensor x based on the visible scores.

    Args:
        x: Tensor, the input tensor to be knocked out.
        num_images: int, the number of images.
        num_patches_per_image: int, the number of patches per image.
        width: int, the width of the input tensor.
        height: int, the height of the input tensor.
        visible_scores: List[Tensor], the visible scores of the input tensor.
        fill_value: float, the value to fill the knocked out attention values.
        
    Returns:
        Tensor, the knocked out tensor.
    """
    print("attention map mean before knockout: ", x.mean().item())
    print("attention map max before knockout: ", x.max().item())
    print("attention map min before knockout: ", x.min().item())
    print("attention map sum before knockout: ", x.sum().item())

    debug_mode = False
    # if debug_mode:
    #     visible_mask_grid = visible_scores.cpu().numpy().reshape(4, 4, height, width)
    #     visible_mask = np.concatenate([
    #         np.concatenate([visible_mask_grid[i, j] for j in range(4)], axis=1)
    #         for i in range(4)
    #     ], axis=0)
    #     from matplotlib import pyplot as plt
    #     plt.imshow(visible_mask)
    #     plt.colorbar()
    #     plt.show()


    B, H, N, _ = x.shape 
    special_token_num = 5
    num_tokens_per_image_total = num_patches_per_image + special_token_num

    attention_map_before_knockout, attention_map_after_knockout = None, None
    if debug_mode:
        i, j = 1, 3
        patch_row, patch_col = 12, 18
        attention_map_before_knockout = get_attention_map(x, i, j, patch_row, patch_col, num_tokens_per_image_total, special_token_num, width, height)


    scores = torch.zeros(x.shape[-2:], device="cpu")
    special_token_start_idx = torch.arange(0, num_images * num_tokens_per_image_total, num_tokens_per_image_total)
    special_token_indices = (
        special_token_start_idx[:, None] + torch.arange(special_token_num)
    ).reshape(-1)
    scores[special_token_indices, :] = 1
    scores[:, special_token_indices] = 1

    for i in range(num_images):
        for j in range(num_images):
            scores[special_token_num+num_tokens_per_image_total*i:num_tokens_per_image_total*(i+1), special_token_num+num_tokens_per_image_total*j:num_tokens_per_image_total*(j+1)] = torch.tensor(visible_scores[i * num_images + j])

    mask = (scores < 0.1).unsqueeze(0).unsqueeze(0).to(x.device)
    x.masked_fill_(mask, fill_value)

    if debug_mode:
        attention_map_after_knockout = get_attention_map(x, i, j, patch_row, patch_col, num_tokens_per_image_total, special_token_num, width, height)
        from matplotlib import pyplot as plt
        plt.imshow(np.concatenate([attention_map_before_knockout, attention_map_after_knockout], axis=0))
        plt.colorbar()
        plt.show()


    print("attention map mean after knockout: ", x.mean().item())
    print("attention map sum after knockout: ", x.sum().item())
    print("knockout ratio: ", mask.sum().item() / mask.numel())
    del scores, mask

    return x

def non_corresponding_knockout(x: Tensor, num_images: int, num_patches_per_image: int, width: int, height: int,
                           visible_scores: List[Tensor], fill_value: float = float('-inf')) -> Tensor:
    """
    Knock out the attention values in the input tensor x based on the visible scores.

    Args:
        x: Tensor, the input tensor to be knocked out.
        num_images: int, the number of images.
        num_patches_per_image: int, the number of patches per image.
        width: int, the width of the input tensor.
        height: int, the height of the input tensor.
        visible_scores: List[Tensor], the visible scores of the input tensor.
        fill_value: float, the value to fill the knocked out attention values.
        
    Returns:
        Tensor, the knocked out tensor.
    """
    print("attention map mean before knockout: ", x.mean().item())
    print("attention map max before knockout: ", x.max().item())
    print("attention map min before knockout: ", x.min().item())
    print("attention map sum before knockout: ", x.sum().item())

    debug_mode = False
    # if debug_mode:
    #     visible_mask_grid = visible_scores.cpu().numpy().reshape(4, 4, height, width)
    #     visible_mask = np.concatenate([
    #         np.concatenate([visible_mask_grid[i, j] for j in range(4)], axis=1)
    #         for i in range(4)
    #     ], axis=0)
    #     from matplotlib import pyplot as plt
    #     plt.imshow(visible_mask)
    #     plt.colorbar()
    #     plt.show()


    B, H, N, _ = x.shape 
    special_token_num = 5
    num_tokens_per_image_total = num_patches_per_image + special_token_num

    attention_map_before_knockout, attention_map_after_knockout = None, None
    if debug_mode:
        i, j = 1, 3
        patch_row, patch_col = 12, 18
        attention_map_before_knockout = get_attention_map(x, i, j, patch_row, patch_col, num_tokens_per_image_total, special_token_num, width, height)


    scores = torch.zeros(x.shape[-2:], device="cpu")
    special_token_start_idx = torch.arange(0, num_images * num_tokens_per_image_total, num_tokens_per_image_total)
    special_token_indices = (
        special_token_start_idx[:, None] + torch.arange(special_token_num)
    ).reshape(-1)
    scores[special_token_indices, :] = 1
    scores[:, special_token_indices] = 1

    for i in range(num_images):
        for j in range(num_images):
            scores[special_token_num+num_tokens_per_image_total*i:num_tokens_per_image_total*(i+1), special_token_num+num_tokens_per_image_total*j:num_tokens_per_image_total*(j+1)] = torch.tensor(visible_scores[i * num_images + j])

    mask = (scores < 0.1).unsqueeze(0).unsqueeze(0).to(x.device)
    x.masked_fill_(mask, fill_value)

    if debug_mode:
        attention_map_after_knockout = get_attention_map(x, i, j, patch_row, patch_col, num_tokens_per_image_total, special_token_num, width, height)
        from matplotlib import pyplot as plt
        plt.imshow(np.concatenate([attention_map_before_knockout, attention_map_after_knockout], axis=0))
        plt.colorbar()
        plt.show()


    print("attention map mean after knockout: ", x.mean().item())
    print("attention map sum after knockout: ", x.sum().item())
    print("knockout ratio: ", mask.sum().item() / mask.numel())
    del scores, mask

    return x

def top_k_preserved_knockout(x: Tensor, num_images: int, num_patches_per_image: int, width: int, height: int, k: int, fill_value: float = float('-inf')) -> Tensor:
    """
    Knock out the attention values in the input tensor x, preserving only the top-k values
    (in addition to special tokens).

    Args:
        x: Tensor, the input tensor to be knocked out.
        num_images: int, the number of images.
        num_patches_per_image: int, the number of patches per image.
        width: int, the width of the input tensor.
        height: int, the height of the input tensor.
        k: int, the number of top attention values to preserve.
        fill_value: float, the value to fill the knocked out attention values.
        
    Returns:
        Tensor, the knocked out tensor.
    """
    debug_mode = False
    print("attention map mean before knockout: ", x.mean().item())
    print("attention map max before knockout: ", x.max().item())
    print("attention map min before knockout: ", x.min().item())
    print("attention map sum before knockout: ", x.sum().item())

    B, H, N, _ = x.shape 
    special_token_num = 5
    num_tokens_per_image_total = num_patches_per_image + special_token_num

    attention_map_before_knockout, attention_map_after_knockout = None, None
    if debug_mode:
        i, j = 1, 3
        patch_row, patch_col = 12, 18
        attention_map_before_knockout = get_attention_map(x, i, j, patch_row, patch_col, num_tokens_per_image_total, special_token_num, width, height)

    preserve_mask = torch.zeros(x.shape[-2:], device="cpu")
    special_token_start_idx = torch.arange(0, num_images * num_tokens_per_image_total, num_tokens_per_image_total)
    special_token_indices = (
        special_token_start_idx[:, None] + torch.arange(5)
    ).reshape(-1)
    preserve_mask[special_token_indices, :] = 1
    preserve_mask[:, special_token_indices] = 1

    can_knockout_mask = (preserve_mask == 0)

    x_reshaped = x[:, :, can_knockout_mask].reshape(B, H, num_patches_per_image*num_images, num_patches_per_image*num_images)

    x_reshaped = x_reshaped.reshape(-1, num_patches_per_image)
    top_k_values, top_k_indices = torch.topk(x_reshaped, k, dim=-1)

    x_reshaped.fill_(fill_value)
    x_reshaped.scatter_(1, top_k_indices, top_k_values)

    x[:, :, can_knockout_mask] = x_reshaped.reshape(B, H, -1)


    if debug_mode:
        attention_map_after_knockout = get_attention_map(x, i, j, patch_row, patch_col, num_tokens_per_image_total, special_token_num, width, height)
        from matplotlib import pyplot as plt
        plt.imshow(np.concatenate([attention_map_before_knockout, attention_map_after_knockout], axis=0))
        plt.colorbar()
        plt.show()
    
    print("attention map mean after knockout: ", x.mean().item())
    print("attention map max after knockout: ", x.max().item())
    print("attention map min after knockout: ", x.min().item())
    print("attention map sum after knockout: ", x.sum().item())
    # 清理中间变量以节省内存
    del preserve_mask, can_knockout_mask
    if 'x_reshaped' in locals():
        del x_reshaped
    
    return x

def corres_mask_knockout(x: Tensor, num_images: int, num_patches_per_image: int, width: int, height: int, corres_masks: List[List[int]], fill_value: float = float('-inf')) -> Tensor:
    debug_mode = False
    print("attention map mean before knockout: ", x.mean().item())
    print("attention map max before knockout: ", x.max().item())
    print("attention map min before knockout: ", x.min().item())
    print("attention map sum before knockout: ", x.sum().item())

    B, H, N, _ = x.shape 
    special_token_num = 5
    num_tokens_per_image_total = num_patches_per_image + special_token_num

    attention_map_before_knockout, attention_map_after_knockout = None, None
    if debug_mode:
        display_i, display_j = 1, 3
        patch_row, patch_col = 10, 14
        attention_map_before_knockout = get_attention_map(x, display_i, display_j, patch_row, patch_col, num_tokens_per_image_total, special_token_num, width, height)

    for i in range(num_images):
        for r in range(height):
            for c in range(width):
                idx = i * num_patches_per_image + r * width + c
                query_idx = i * num_tokens_per_image_total + 5 + r * width + c
                valid_keys = corres_masks[idx]
                mask = torch.ones(x.shape[-1], dtype=torch.bool, device=x.device)
                if len(valid_keys) > 0:
                    mask[valid_keys] = False
                x[:, :, query_idx, mask] = fill_value
    
    if debug_mode:
        attention_map_after_knockout = get_attention_map(x, display_i, display_j, patch_row, patch_col, num_tokens_per_image_total, special_token_num, width, height)
        from matplotlib import pyplot as plt
        plt.imshow(np.concatenate([attention_map_before_knockout, attention_map_after_knockout], axis=0))
        plt.colorbar()
        plt.show()
        
    print("attention map mean after knockout: ", x.mean().item())
    print("attention map max after knockout: ", x.max().item())
    print("attention map min after knockout: ", x.min().item())
    print("attention map sum after knockout: ", x.sum().item())
    return x
