import numpy as np
import open3d as o3d
import sys


def display_point_clouds(
    points: list[np.ndarray],
    colors: list[np.ndarray],
    title: str = "Point Clouds",
):
    """
    Display point clouds using Open3D visualization.
    
    Args:
        points: List of point cloud arrays, each of shape (N, 3)
        colors: List of color arrays, each of shape (N, 3) with values in [0, 1] or [0, 255]
        title: Window title for visualization
    """
    log_prefix = "[display_point_clouds]"
    # print(f"{log_prefix} start")
    
    # Check if points list is empty
    if not points or len(points) == 0:
        print(f"{log_prefix} ERROR: Empty points list!")
        return
    
    # Concatenate all point clouds
    try:
        points_concat = np.concatenate(points, axis=0)
        colors_concat = np.concatenate(colors, axis=0)
    except Exception as e:
        print(f"{log_prefix} ERROR: Failed to concatenate points/colors: {e}")
        return
    
    # print(f"{log_prefix} total points: {len(points_concat)}, colors: {colors_concat.shape}")
    
    # Check if point cloud is empty
    if len(points_concat) == 0:
        print(f"{log_prefix} ERROR: Point cloud is empty!")
        return
    
    # Validate shapes
    if points_concat.shape[1] != 3:
        print(f"{log_prefix} ERROR: Points must have shape (N, 3), got {points_concat.shape}")
        return
    
    if colors_concat.shape[1] != 3:
        print(f"{log_prefix} ERROR: Colors must have shape (N, 3), got {colors_concat.shape}")
        return
    
    if len(points_concat) != len(colors_concat):
        print(f"{log_prefix} ERROR: Points and colors must have same length: "
              f"{len(points_concat)} vs {len(colors_concat)}")
        return
    
    # Normalize colors to [0, 1] range if needed
    if colors_concat.max() > 1.0:
        print(f"{log_prefix} normalizing colors from [0, 255] to [0, 1]")
        colors_concat = colors_concat / 255.0
    
    # Ensure colors are in [0, 1] range
    colors_concat = np.clip(colors_concat, 0.0, 1.0)
    
    # Filter out NaN and Inf values
    valid_mask = np.isfinite(points_concat).all(axis=1)
    nan_count = np.sum(~valid_mask)
    
    if nan_count > 0:
        print(f"{log_prefix} WARNING: Found {nan_count} points with NaN/Inf values, filtering them out")
        points_concat = points_concat[valid_mask]
        colors_concat = colors_concat[valid_mask]
        print(f"{log_prefix} after filtering: {len(points_concat)} valid points")
    
    if len(points_concat) == 0:
        print(f"{log_prefix} ERROR: No valid points after filtering NaN/Inf values!")
        return
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_concat.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors_concat.astype(np.float64))
    
    # Check point cloud validity
    if len(pcd.points) == 0:
        print(f"{log_prefix} ERROR: Point cloud has no points after creation!")
        return
    
    # print(f"{log_prefix} point cloud created: {len(pcd.points)} points")
    # print(f"{log_prefix} point range: [{points_concat.min(axis=0)}, {points_concat.max(axis=0)}]")
    # print(f"{log_prefix} color range: [{colors_concat.min(axis=0)}, {colors_concat.max(axis=0)}]")
    
    # Try to display
    try:
        # print(f"{log_prefix} opening visualization window...")
        o3d.visualization.draw_geometries([pcd], window_name=title)
        # print(f"{log_prefix} visualization closed")
    except Exception as e:
        print(f"{log_prefix} ERROR: Failed to display point cloud: {e}")
        print(f"{log_prefix} This might be due to:")
        print(f"{log_prefix}   - No display available (headless server)")
        print(f"{log_prefix}   - Open3D visualization backend not available")
        print(f"{log_prefix}   - Try using o3d.visualization.Visualizer() for more control")
    
    # print(f"{log_prefix} done")
