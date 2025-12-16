import torch
import numpy as np
import open3d as o3d

def stat_filter(pnts, colors, nb_neighbors=20, std_ratio=2.0):
    log_prefix = "[stat_filter]"
    print(f"{log_prefix} start")
    print(f"{log_prefix} input points: {pnts.shape}, colors: {colors.shape}, "
          f"nb_neighbors={nb_neighbors}, std_ratio={std_ratio}")

    pcd_filter = o3d.geometry.PointCloud()
    pcd_filter.points = o3d.utility.Vector3dVector(pnts)
    pcd_filter.colors = o3d.utility.Vector3dVector(colors)

    pcd_filter, ind = pcd_filter.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )

    kept = len(ind)
    total = len(pnts)
    print(f"{log_prefix} kept {kept}/{total} points "
          f"({kept / max(total, 1):.4f} kept, {(total - kept) / max(total, 1):.4f} removed)")
    print(f"{log_prefix} done")

    return pnts[ind], colors[ind], ind

def remove_Nan_Zero_Inf(pnts, colors):
    log_prefix = "[remove_Nan_Zero_Inf]"
    print(f"{log_prefix} start")
    print(f"{log_prefix} input points: {pnts.shape}, colors: {colors.shape}")

    # Check for finite values (excludes NaN and Inf)
    finite_mask = np.isfinite(pnts).all(axis=1)  # (N,)
    # Check that not all coordinates are zero (exclude zero vectors)
    nonzero_mask = ~(pnts == 0).all(axis=1)  # (N,)
    
    valid_mask = finite_mask & nonzero_mask  # (N,)
    
    filtered_count = np.sum(~valid_mask)
    if filtered_count > 0:
        print(f"{log_prefix} filtered out {filtered_count} invalid points (NaN/Inf/Zero)")
    
    print(f"{log_prefix} valid mask: {valid_mask.shape}, valid points: {np.sum(valid_mask)}")
    print(f"{log_prefix} done")
    
    return pnts[valid_mask], colors[valid_mask]

def filter_depth_by_conf(depth, conf, conf_percentile, conf_score=None, verbose=False):
    """
    Filter depth map based on confidence percentile threshold.
    
    Args:
        depth: Depth tensor of shape (B, 1, H, W) or (B, H, W) or (H, W)
        conf: Confidence tensor of shape (B, 1, H, W) or (B, H, W) or (H, W)
        conf_percentile: Percentile threshold (0-100) for confidence filtering
        conf_score: Optional absolute confidence threshold. If not None, this
            value will be used as the threshold instead of conf_percentile.
        verbose: If True, print filtering statistics
    
    Returns:
        tuple: (filtered_depth, conf_mask)
            - filtered_depth: Filtered depth tensor with same shape as input depth
            - conf_mask: Confidence mask tensor with same shape as input depth
    """

    log_prefix = "[filter_depth_by_conf]"
    print(f"{log_prefix} start")
    # Ensure both are tensors
    if not isinstance(depth, torch.Tensor):
        depth = torch.from_numpy(depth) if isinstance(depth, np.ndarray) else torch.tensor(depth)
    if not isinstance(conf, torch.Tensor):
        conf = torch.from_numpy(conf) if isinstance(conf, np.ndarray) else torch.tensor(conf)
    
    # Store original shape
    original_depth_shape = depth.shape
    original_depth_dim = depth.dim()
    
    # Calculate confidence threshold
    if conf_score is not None:
        # Use provided absolute confidence score as threshold
        if isinstance(conf_score, torch.Tensor):
            conf_threshold = float(conf_score.item())
        else:
            conf_threshold = float(conf_score)
        threshold_source = f"score={conf_threshold:.4f}"
    else:
        # Use percentile over all confidence values
        conf_flat = conf.flatten().numpy()
        conf_threshold = np.percentile(conf_flat, conf_percentile)
        threshold_source = f"p={conf_percentile}"
    
    if verbose:
        print(f"{log_prefix} conf shape: {conf.shape}")
        print(f"{log_prefix} threshold ({threshold_source}): {conf_threshold:.4f}")
    
    # Generate mask and apply to depth
    # Handle shape matching - squeeze if needed to match dimensions
    depth_squeezed = depth.squeeze() if depth.dim() > 2 else depth
    conf_squeezed = conf.squeeze() if conf.dim() > 2 else conf
    
    conf_mask = (conf_squeezed >= conf_threshold)
    
    if verbose:
        print(f"{log_prefix} conf_mask shape: {conf_mask.shape}")
        num_masked = conf_mask.sum().item()
        total = conf_mask.numel()
        print(f"{log_prefix} num pts masked: {num_masked}, ratio: {num_masked / total:.4f}")
    
    # Apply mask to depth
    filtered_depth = depth_squeezed * conf_mask
    
    # Restore original shape for both depth and mask
    if original_depth_dim != filtered_depth.dim():
        # Try to restore original shape
        if original_depth_dim == 4 and filtered_depth.dim() == 3:
            filtered_depth = filtered_depth.unsqueeze(1)
            conf_mask = conf_mask.unsqueeze(1)
        elif original_depth_dim == 3 and filtered_depth.dim() == 2:
            filtered_depth = filtered_depth.unsqueeze(0)
            conf_mask = conf_mask.unsqueeze(0)
    
    # Ensure mask has the same shape as original depth
    if conf_mask.shape != original_depth_shape:
        # Reshape mask to match original depth shape
        conf_mask = conf_mask.view(original_depth_shape)
    
    print(f"{log_prefix} done")
    return filtered_depth, conf_mask