import cv2 as cv
import os, sys
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from vggt.utils.load_fn import load_and_preprocess_images

def resize_attention_map(attn, target_size):
    return cv.resize(attn, (target_size[1], target_size[0]))  # 注意cv2的宽高顺序

rows = 25
# rows = 37
cols = 37
patch_size = 14   
image_height = rows * patch_size
image_weight = cols * patch_size
total_block_num = 24
num_tokens_per_image = rows * cols + 5

if __name__ == '__main__':
    save_dir = "/home/rokae/zcl/vggt/agg_attn_maps_exp1"
    frame_attn_map_prefix = "frame_attn_map"
    global_attn_map_prefix = "global_attn_map"
    original_image_path = []

    for i in range(10):
        original_image_path += ["/home/rokae/zcl/vggt/examples/kitchen/images/{:02d}.png".format(i)]
    original_image = load_and_preprocess_images(original_image_path)
    original_image = [i.numpy().transpose(1, 2, 0) for i in original_image]

    
    def load_global_attn_map(query_image_idx:int, block_idx:int):
        """
        p2p stands for patch 2 patch attention
        """
        p2p_list = []
        frame_attn_path = os.path.join(save_dir, f"{global_attn_map_prefix}_{block_idx}.pt")
        print(f"Load attn map: {frame_attn_path}")
        frame_attn = torch.load(frame_attn_path).to(dtype=torch.float32)
        frame_attn = frame_attn.softmax(dim=-1)
        frame_attn = frame_attn.sum(dim=1)
        image_num = int(frame_attn.shape[-1] / num_tokens_per_image)
        for i in range(image_num):
            p2p_list.append(
                frame_attn[
                    0, 
                    num_tokens_per_image * query_image_idx + 5:num_tokens_per_image * (query_image_idx+1), 
                    num_tokens_per_image * i + 5: num_tokens_per_image * (i+1) 
                ].unsqueeze(0))
        p2p = torch.concat(p2p_list, dim=0)
        return p2p
    
    query_image_idx = 0
    block_idx = 15
    # p2p_1 = load_global_attn_map(query_image_idx, 1)
    # p2p_4 = load_global_attn_map(query_image_idx, 4)
    # p2p_11 = load_global_attn_map(query_image_idx, 11)
    # p2p_11 = load_global_attn_map(query_image_idx, 15)
    # p2p_17 = load_global_attn_map(query_image_idx, 17)
    # p2p_23 = load_global_attn_map(query_image_idx, 23)
    p2p = torch.concat([
        load_global_attn_map(query_image_idx, 1), 
        load_global_attn_map(query_image_idx, 4), 
        load_global_attn_map(query_image_idx, 11), 
        load_global_attn_map(query_image_idx, 15), 
        load_global_attn_map(query_image_idx, 17), 
        load_global_attn_map(query_image_idx, 23), 
        load_global_attn_map(query_image_idx, 24), 
    ], dim=0)
    print(p2p.shape)
    
    def apply_heatmap(image, heatmap, alpha=0.8):
        """
        :param image: 原始图像，shape (H, W, 3)
        :param heatmap: 注意力图，shape (H, W)，值范围 [0, 1]
        :param alpha: 热图透明度
        """
        # 确保图像数据类型一致
        image = 255 * (image - image.min()) / (image.max() - image.min())
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        heatmap = cv.cvtColor(heatmap, cv.COLOR_GRAY2RGB) if len(heatmap.shape)==2 else heatmap
        heatmap_colored = cv.applyColorMap(np.uint8(255 * heatmap), cv.COLORMAP_JET)
        heatmap_colored = heatmap_colored[..., [2, 1, 0]]
        # heatmap_colored = cv.cvtColor(heatmap, cv.COLOR_BGR2RGB)
        
        # 如果图像不是 uint8 类型，进行转换
        if heatmap_colored.dtype != np.uint8:
            heatmap_colored = heatmap_colored.astype(np.uint8)
        
        # image_enhanced = cv.convertScaleAbs(image, alpha=1.5, beta=0)
        output = cv.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        return output
    """
    def make_grid(images, max_cols=None):
        n = len(images)
        if n == 0:
            raise ValueError("No images to stack")

        # 自动决定行列数
        if max_cols is None:
            cols = int(math.ceil(math.sqrt(n)))  # 列数≈√n
        else:
            cols = min(max_cols, n)
        rows = int(math.ceil(n / cols))

        # 填充到完整矩阵
        h, w, c = images[0].shape
        blank = np.zeros_like(images[0])
        padded = images + [blank] * (rows * cols - n)

        # 拼接
        grid = np.vstack([
            np.hstack(padded[i*cols:(i+1)*cols])
            for i in range(rows)
        ])
        return grid
    """

    def make_grid(images, rows=None, max_cols=None):
        """
        自动将一组图像拼接成网格。
        
        参数：
            images: list[np.ndarray]，每张图尺寸相同
            rows: 指定行数，可选（默认自动）
            max_cols: 最大列数，可选（默认根据数量自动选最接近正方形的形状）
        返回：
            grid_image: np.ndarray 拼接好的图像
        """
        n = len(images)
        if n == 0:
            raise ValueError("No images to stack")

        # 自动决定行列数
        if rows is not None:
            cols = int(math.ceil(n / rows))
        elif max_cols is not None:
            cols = min(max_cols, n)
            rows = int(math.ceil(n / cols))
        else:
            cols = int(math.ceil(math.sqrt(n)))
            rows = int(math.ceil(n / cols))

        # 填充到完整矩阵
        h, w, c = images[0].shape
        blank = np.zeros_like(images[0])
        padded = images + [blank] * (rows * cols - n)

        # 拼接
        grid = np.vstack([
            np.hstack(padded[i*cols:(i+1)*cols])
            for i in range(rows)
        ])
        return grid

    def get_patch_image(idx):
        patch_attn = p2p[:, idx, :]
        patch_attn_image = patch_attn.numpy().reshape(patch_attn.shape[0], rows, cols)
        attn_map_display = [resize_attention_map(patch_attn_image[i], [image_height, image_weight])
                            for i in range(patch_attn_image.shape[0])]
        attn_map_display = [(i - i.min()) / (i.max() - i.min()) for i in attn_map_display]
        attn_map_display_applied_heatmap = []
        for i in range(len(attn_map_display)):
            attn_map_display_applied_heatmap.append(
                apply_heatmap(original_image[i % 10], attn_map_display[i])
            )
        attn_map_display = attn_map_display_applied_heatmap

        pad = 5
        attn_maps_padded = [np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0) for img in attn_map_display]

        grid_image = make_grid(attn_maps_padded, 7)

        image_with_patch = original_image[query_image_idx].copy()
        # 计算 patch 行列位置
        patch_row = idx // cols
        patch_col = idx % cols
        # 计算像素坐标
        top_left = (patch_col * patch_size, patch_row * patch_size)
        bottom_right = ((patch_col + 1) * patch_size, (patch_row + 1) * patch_size)
        # 画红色框
        image_with_patch_display = (255 * image_with_patch / image_with_patch.max()).astype(np.uint8)
        cv.rectangle(image_with_patch_display, top_left, bottom_right, color=(255, 0, 0), thickness=2)

        ref_image_left = np.vstack([np.pad(image_with_patch_display, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)] * 7)
        grid_image = np.hstack([ref_image_left, grid_image])

        ref_image_top = [np.zeros(original_image[0].shape)] + original_image
        ref_image_top = [np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0) for img in ref_image_top]
        ref_image_top = np.hstack(ref_image_top)
        ref_image_top = 255 * (ref_image_top - np.min(ref_image_top)) / (np.max(ref_image_top) - np.min(ref_image_top))
        ref_image_top = ref_image_top.astype(np.uint8)

        grid_image = np.vstack([ref_image_top, grid_image])

        return grid_image, image_with_patch_display
    
    init_idx = 0
    attn_map_display, orig_with_patch = get_patch_image(init_idx)

    # 只创建一个子图
    fig, ax1 = plt.subplots(figsize=(6, 4))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0.15)

    # 显示初始图像
    im1 = ax1.imshow(attn_map_display)
    ax1.axis('off')

    # 滑条控件
    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(ax_slider, 'idx', 0, rows * cols - 1, valinit=init_idx, valstep=1)

    # 更新函数
    def update(val):
        idx = int(slider.val)
        attn_map_display, _ = get_patch_image(idx)  # 只取第一张
        im1.set_data(attn_map_display)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()


#     # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
#     fig, (ax1, ax2) = plt.subplots(
#     1, 2,
#     figsize=(12, 4),
#     gridspec_kw={'width_ratios': [3, 1]}  # 左右宽度比例 3:1
# )
#     plt.subplots_adjust(left=0, right=1, top=1, bottom=0.15, wspace=0.05, hspace=0.05)

#     im1 = ax1.imshow(attn_map_display)
#     ax1.axis('off')
#     ax2.imshow(orig_with_patch)
#     ax2.axis('off')

#     # 滑条
#     ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
#     slider = Slider(ax_slider, 'idx', 0, rows*cols - 1, valinit=init_idx, valstep=1)

#     def update(val):
#         idx = int(slider.val)
#         attn_map_display, orig_with_patch = get_patch_image(idx)
#         im1.set_data(attn_map_display)
#         ax2.images[0].set_data(orig_with_patch)
#         fig.canvas.draw_idle()

#     slider.on_changed(update)
#     plt.show()
