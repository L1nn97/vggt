import argparse
import os
import numpy as np

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


def align_pred_to_gt(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    valid_mask: np.ndarray,
    min_valid_pixels: int = 100,
) -> tuple[float, float, np.ndarray]:
    """
    Code snippet from: https://github.com/facebookresearch/vggt/issues/208

    Align predicted depth map to ground truth depth map using a scale and shift,
    such that: gt_depth ≈ scale * pred_depth + shift.

    Args:
        pred_depth (np.ndarray): (H, W) predicted depth.
        gt_depth (np.ndarray): (H, W) ground truth depth.
        valid_mask (np.ndarray): (H, W) mask indicating valid pixels.
        min_valid_pixels (int): Minimum valid pixels required for alignment.

    Returns:
        tuple[float, float, np.ndarray]:
            scale: The calculated scale value (NaN if failed).
            shift: The calculated shift value (NaN if failed).
            aligned_pred_depth: (H, W) predicted depth after alignment 
                                (input pred_depth if alignment failed).
    """
    if pred_depth.shape != gt_depth.shape:
        raise ValueError(
            f"Predicted depth shape {pred_depth.shape} must match GT depth shape {gt_depth.shape}"
        )

    valid_mask = valid_mask.astype(bool)

    # Extract valid depth values
    gt_masked = gt_depth[valid_mask]
    pred_masked = pred_depth[valid_mask]

    if len(gt_masked) < min_valid_pixels:
        print(
            f"Warning: Not enough valid pixels ({len(gt_masked)} < {min_valid_pixels}) to align. "
            "Using all pixels."
        )
        gt_masked = gt_depth.reshape(-1)
        pred_masked = pred_depth.reshape(-1)


    # Handle case where pred_masked has no variance (e.g., all zeros or a constant value)
    if np.std(pred_masked) < 1e-6: # Small epsilon to check for near-constant values
        print(
            "Warning: Predicted depth values in the valid mask have near-zero variance. "
            "Scale is ill-defined. Setting scale=1 and solving for shift only."
        )
        scale = 1.0
        shift = np.mean(gt_masked) - np.mean(pred_masked) # or np.median(gt_masked) - np.median(pred_masked)
    else:
        A = np.vstack([pred_masked, np.ones_like(pred_masked)]).T
        try:
            x, residuals, rank, s_values = np.linalg.lstsq(A, gt_masked, rcond=None)
            scale, shift = x[0], x[1]
        except np.linalg.LinAlgError as e:
            print(f"Warning: Least squares alignment failed ({e}). Returning original prediction.")
            return np.nan, np.nan, pred_depth.copy()


    aligned_pred_depth = scale * pred_depth + shift
    return scale, shift, aligned_pred_depth



def calc_aligned_depth_filter_outliers(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    gt_mask: np.ndarray,
    use_local_display: bool = True,
    verbose: bool = False,
    filter_outliers: bool = True,
    filter_method: str = "percentile",
    filter_threshold: float = 99.0,
):
    """
    对齐预测深度与GT深度，计算误差统计，并可视化结果。

    Args:
        pred_depth: 预测深度图，形状为 [H, W] 或 [H, W, 1]
        gt_depth: GT深度图，可以是numpy数组或tensor
        gt_mask: GT mask，可以是numpy数组或tensor
        use_local_display: 是否本地显示图像
        verbose: 是否打印详细信息
        filter_outliers: 是否过滤噪声点（异常值）
        filter_method: 过滤方法，可选：
            - "percentile": 使用分位数过滤，移除超过filter_threshold百分位数的点（默认）
            - "sigma": 使用3-sigma规则，移除超过均值±filter_threshold倍标准差的点
            - "iqr": 使用四分位距（IQR）方法，filter_threshold为IQR倍数（默认1.5）
            - "absolute": 使用绝对阈值，移除误差超过filter_threshold的点
        filter_threshold: 过滤阈值，根据filter_method不同含义不同：
            - percentile: 分位数（0-100），默认99.0
            - sigma: 标准差倍数，默认3.0
            - iqr: IQR倍数，默认1.5
            - absolute: 绝对误差阈值

    Returns:
        tuple: (scale, shift, aligned_pred_depth, stats_dict) 
            - scale: 对齐缩放因子
            - shift: 对齐偏移量
            - aligned_pred_depth: 对齐后的预测深度
            - stats_dict: 深度误差统计字典，包含以下键：
                - mean: 均值
                - std: 标准差
                - median: 中位数
                - min: 最小值
                - max: 最大值
                - q25: 25%分位数
                - q75: 75%分位数
                - q95: 95%分位数
                - q99: 99%分位数
                - valid_pixels: 有效像素数
                - total_pixels: 总像素数
                - valid_ratio: 有效像素比例（百分比）
    """
    # 转换为numpy数组
    if isinstance(gt_depth, torch.Tensor):
        gt_depth = gt_depth.cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()

    # 对齐预测深度到GT
    scale, shift, aligned_pred_depth = align_pred_to_gt(
        pred_depth, gt_depth, gt_mask, 100
    )

    # 打印基本信息（仅在verbose模式）
    if verbose:
        print(f"{'Scale':<15}: {scale:.6f}")
        print(f"{'Shift':<15}: {shift:.6f}")

    # 处理维度，确保都是 [H, W]
    gt_depth_view = gt_depth.copy()
    gt_mask_view = gt_mask.copy()

    if gt_depth_view.ndim > 2:
        gt_depth_view = gt_depth_view.squeeze()
    if aligned_pred_depth.ndim > 2:
        aligned_pred_depth = aligned_pred_depth.squeeze()
    if gt_mask_view.ndim > 2:
        gt_mask_view = gt_mask_view.squeeze()

    # 计算差值（绝对值误差）
    depth_diff = np.abs(gt_depth_view - aligned_pred_depth)
    # 只考虑mask有效区域
    valid_mask = gt_mask_view.astype(bool)
    depth_diff_masked = np.where(valid_mask, depth_diff, np.nan)

    # 计算差值的统计数据
    valid_diff = depth_diff[valid_mask] if np.any(valid_mask) else np.array([])
    
    # 过滤噪声点（异常值）
    filtered_valid_diff = valid_diff.copy()
    num_filtered = 0
    if filter_outliers and len(valid_diff) > 0:
        original_count = len(valid_diff)
        
        # 根据方法选择合理的阈值（如果使用默认值99.0）
        threshold = filter_threshold
        if filter_method == "sigma" and threshold == 99.0:
            threshold = 3.0  # 3-sigma规则的合理默认值
        elif filter_method == "iqr" and threshold == 99.0:
            threshold = 1.5  # IQR方法的常见默认值
        
        if filter_method == "percentile":
            # 分位数过滤：移除超过指定百分位数的点
            percentile_threshold = np.percentile(valid_diff, threshold)
            mask_filtered = valid_diff <= percentile_threshold
            filtered_valid_diff = valid_diff[mask_filtered]
            
        elif filter_method == "sigma":
            # 3-sigma规则：移除超过均值±N倍标准差的点
            mean_val = np.mean(valid_diff)
            std_val = np.std(valid_diff)
            lower_bound = mean_val - threshold * std_val
            upper_bound = mean_val + threshold * std_val
            mask_filtered = (valid_diff >= lower_bound) & (valid_diff <= upper_bound)
            filtered_valid_diff = valid_diff[mask_filtered]
            
        elif filter_method == "iqr":
            # IQR方法：使用四分位距过滤异常值
            q25 = np.percentile(valid_diff, 25)
            q75 = np.percentile(valid_diff, 75)
            iqr = q75 - q25
            lower_bound = q25 - threshold * iqr
            upper_bound = q75 + threshold * iqr
            mask_filtered = (valid_diff >= lower_bound) & (valid_diff <= upper_bound)
            filtered_valid_diff = valid_diff[mask_filtered]
            
        elif filter_method == "absolute":
            # 绝对阈值过滤：移除误差超过指定绝对值的点
            mask_filtered = valid_diff <= threshold
            filtered_valid_diff = valid_diff[mask_filtered]
            
        else:
            # 未知方法，不进行过滤
            if verbose:
                print(f"警告: 未知的过滤方法 '{filter_method}'，跳过过滤")
        
        num_filtered = original_count - len(filtered_valid_diff)
        if verbose and num_filtered > 0:
            print(f"过滤了 {num_filtered} 个噪声点 ({num_filtered/original_count*100:.2f}%)")
    
    # 使用过滤后的数据计算统计信息
    stats_diff = filtered_valid_diff
    
    # 构建统计信息字典（基于过滤后的数据）
    stats_dict = {}
    if len(stats_diff) > 0:
        stats_dict = {
            "mean": float(np.mean(stats_diff)),
            "std": float(np.std(stats_diff)),
            "median": float(np.median(stats_diff)),
            "min": float(np.min(stats_diff)),
            "max": float(np.max(stats_diff)),
            "q25": float(np.percentile(stats_diff, 25)),
            "q75": float(np.percentile(stats_diff, 75)),
            "q95": float(np.percentile(stats_diff, 95)),
            "q99": float(np.percentile(stats_diff, 99)),
            "valid_pixels": int(len(valid_diff)),  # 过滤前的有效像素数
            "filtered_pixels": int(len(stats_diff)),  # 过滤后的有效像素数
            "num_filtered": int(num_filtered),  # 被过滤的像素数
            "total_pixels": int(valid_mask.size),
            "valid_ratio": float(len(valid_diff) / valid_mask.size * 100.0),
            "filtered_ratio": float(len(stats_diff) / valid_mask.size * 100.0),
        }
    else:
        # 如果没有有效像素，返回空值
        stats_dict = {
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "q95": np.nan,
            "q99": np.nan,
            "valid_pixels": int(len(valid_diff)) if len(valid_diff) > 0 else 0,
            "filtered_pixels": 0,
            "num_filtered": 0,
            "total_pixels": int(valid_mask.size),
            "valid_ratio": 0.0,
            "filtered_ratio": 0.0,
        }
    
    if verbose:
        if len(stats_diff) > 0:
            print("\n" + "=" * 60)
            if filter_outliers and num_filtered > 0:
                print(f"深度误差统计信息 (已过滤 {num_filtered} 个异常值, 方法: {filter_method}):")
            else:
                print("深度误差统计信息 (仅在有效mask区域):")
            print("=" * 60)
            print(f"  均值 (Mean):           {stats_dict['mean']:.6f}")
            print(f"  标准差 (Std):          {stats_dict['std']:.6f}")
            print(f"  中位数 (Median):       {stats_dict['median']:.6f}")
            print(f"  最小值 (Min):          {stats_dict['min']:.6f}")
            print(f"  最大值 (Max):          {stats_dict['max']:.6f}")
            print(f"  25%分位数 (Q25):       {stats_dict['q25']:.6f}")
            print(f"  75%分位数 (Q75):       {stats_dict['q75']:.6f}")
            print(f"  95%分位数 (Q95):       {stats_dict['q95']:.6f}")
            print(f"  99%分位数 (Q99):       {stats_dict['q99']:.6f}")
            print(f"  过滤前有效像素数:      {stats_dict['valid_pixels']}")
            if filter_outliers:
                print(f"  过滤后有效像素数:      {stats_dict['filtered_pixels']}")
                print(f"  被过滤的像素数:        {stats_dict['num_filtered']}")
            print(f"  总像素数:              {stats_dict['total_pixels']}")
            print(f"  有效像素比例:          {stats_dict['valid_ratio']:.2f}%")
            if filter_outliers:
                print(f"  过滤后像素比例:        {stats_dict['filtered_ratio']:.2f}%")
            print("=" * 60 + "\n")
        else:
            print("警告: 没有有效的mask区域用于计算误差统计")

    # 水平拼接三个深度图: GT, 对齐后的预测, 差值
    combined_depth = np.concatenate(
        [gt_depth_view, aligned_pred_depth, depth_diff_masked], axis=1
    )

    # 显示拼接后的深度图（由use_local_display控制）
    if use_local_display:
        plt.figure(figsize=(24, 8))
        im = plt.imshow(combined_depth, cmap="viridis")
        plt.colorbar(im, label="Depth / Error")
        plt.title(
            "Left: GT Depth | Middle: Aligned Predicted Depth | Right: Absolute Error"
        )
        plt.axvline(
            x=gt_depth_view.shape[1],
            color="red",
            linestyle="--",
            linewidth=2,
            label="Split line 1",
        )
        plt.axvline(
            x=gt_depth_view.shape[1] * 2,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Split line 2",
        )
        plt.legend()
        plt.show()
        plt.close()

    # 绘制误差统计直方图（由use_local_display控制）
    if len(stats_diff) > 0 and use_local_display:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 左图: 线性尺度直方图
        axes[0].hist(
            stats_diff, bins=100, edgecolor="black", alpha=0.7, color="skyblue"
        )
        axes[0].axvline(
            np.mean(stats_diff),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(stats_diff):.6f}",
        )
        axes[0].axvline(
            np.median(stats_diff),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {np.median(stats_diff):.6f}",
        )
        axes[0].set_xlabel("Absolute Depth Error", fontsize=12)
        axes[0].set_ylabel("Frequency", fontsize=12)
        title_prefix = "Filtered " if filter_outliers and num_filtered > 0 else ""
        axes[0].set_title(f"{title_prefix}Depth Error Histogram (Linear Scale)", fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 右图: 对数尺度直方图
        axes[1].hist(
            stats_diff, bins=100, edgecolor="black", alpha=0.7, color="lightcoral"
        )
        axes[1].axvline(
            np.mean(stats_diff),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(stats_diff):.6f}",
        )
        axes[1].axvline(
            np.median(stats_diff),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {np.median(stats_diff):.6f}",
        )
        axes[1].set_xlabel("Absolute Depth Error", fontsize=12)
        axes[1].set_ylabel("Frequency (Log Scale)", fontsize=12)
        axes[1].set_title("Depth Error Histogram (Log Scale)", fontsize=14)
        axes[1].set_yscale("log")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        plt.close()

    return scale, shift, aligned_pred_depth, stats_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align predicted depth to ground truth depth")
    parser.add_argument("--pred_depth", type=str, required=True, help="Path to predicted depth .npy file")
    parser.add_argument("--gt_depth", type=str, required=True, help="Path to ground truth depth .npy file")
    parser.add_argument("--mask", type=str, default=None, help="Path to mask .npy file (optional)")
    
    args = parser.parse_args()

    def load_depth_from_file(file_path: str) -> np.ndarray:
        """从.npy文件加载深度图"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        depth = np.load(file_path)
        if depth.ndim > 2:
            depth = depth.squeeze()
        if depth.ndim != 2:
            raise ValueError(f"Expected 2D depth map, got shape: {depth.shape}")
        return depth


    def load_mask_from_file(file_path: str) -> np.ndarray:
        """从.npy文件加载mask"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        mask = np.load(file_path)
        if mask.ndim > 2:
            mask = mask.squeeze()
        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask, got shape: {mask.shape}")
        return mask > 0
    
    # 加载文件
    pred_depth = load_depth_from_file(args.pred_depth)
    gt_depth = load_depth_from_file(args.gt_depth)
    
    # 创建mask
    if args.mask is not None:
        valid_mask = load_mask_from_file(args.mask)
    else:
        valid_mask = gt_depth > 1e-3
    
    # 执行对齐
    scale, shift, aligned_pred_depth = align_pred_to_gt(pred_depth, gt_depth, valid_mask)
    
    print(f"Scale: {scale:.6f}")
    print(f"Shift: {shift:.6f}")
