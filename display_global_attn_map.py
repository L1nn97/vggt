import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':
    save_dir = "/home/rokae/zcl/vggt/global_attn_heatmap"
    summary_save_dir = os.path.join(save_dir, "summary")
    layer_num = 24
    sequence_length = 10
    
    
    for j in range(sequence_length):
        fig, axes = plt.subplots(4, 6, figsize=(6 * 5, 5 * 4)) # 根据层数调整图形大小
        for i in range(layer_num):
            image_path = os.path.join(save_dir, f"{i}_{j}.png")
            if os.path.exists(image_path):
                image = cv.imread(image_path)
                axes[i // 6, i % 6].imshow(image)
            else:
                print(f"Warning: No file found at {image_path}")
            axes[i // 6, i % 6].axis('off') # 不显示坐标轴
    
        plt.suptitle(f'Global Attention Layer Image {j}')
        plt.tight_layout()
        # plt.show()
        fig.savefig(os.path.join(summary_save_dir, f"{j}.png"), dpi=150, bbox_inches='tight') 
        