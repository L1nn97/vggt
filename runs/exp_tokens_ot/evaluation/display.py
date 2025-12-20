import numpy as np
import open3d as o3d
import sys
import matplotlib.pyplot as plt
import trimesh


def display_point_clouds(
    points: list[np.ndarray],
    colors: list[np.ndarray] = None,
    title: str = "Point Clouds",
):
    """
    Display point clouds using Open3D visualization.
    
    Args:
        points: List of point cloud arrays, each of shape (N, 3)
        colors: List of color arrays, each of shape (N, 3) with values in [0, 1] or [0, 255], or None
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
    except Exception as e:
        print(f"{log_prefix} ERROR: Failed to concatenate points: {e}")
        return

    # print(f"{log_prefix} total points: {len(points_concat)}")
    
    # Check if point cloud is empty
    if len(points_concat) == 0:
        print(f"{log_prefix} ERROR: Point cloud is empty!")
        return
    
    # Validate point shape
    if points_concat.shape[1] != 3:
        print(f"{log_prefix} ERROR: Points must have shape (N, 3), got {points_concat.shape}")
        return

    # Prepare colors if given, otherwise set for no colors
    use_colors = colors is not None
    if use_colors:
        try:
            colors_concat = np.concatenate(colors, axis=0)
        except Exception as e:
            print(f"{log_prefix} ERROR: Failed to concatenate colors: {e}")
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
    else:
        colors_concat = None

    # Filter out NaN and Inf values
    valid_mask = np.isfinite(points_concat).all(axis=1)
    nan_count = np.sum(~valid_mask)
    
    if nan_count > 0:
        print(f"{log_prefix} WARNING: Found {nan_count} points with NaN/Inf values, filtering them out")
        points_concat = points_concat[valid_mask]
        if use_colors:
            colors_concat = colors_concat[valid_mask]
        print(f"{log_prefix} after filtering: {len(points_concat)} valid points")
    
    if len(points_concat) == 0:
        print(f"{log_prefix} ERROR: No valid points after filtering NaN/Inf values!")
        return
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_concat.astype(np.float64))
    if use_colors:
        pcd.colors = o3d.utility.Vector3dVector(colors_concat.astype(np.float64))
    
    # Check point cloud validity
    if len(pcd.points) == 0:
        print(f"{log_prefix} ERROR: Point cloud has no points after creation!")
        return
    
    # print(f"{log_prefix} point cloud created: {len(pcd.points)} points")
    # print(f"{log_prefix} point range: [{points_concat.min(axis=0)}, {points_concat.max(axis=0)}]")
    # if use_colors:
    #     print(f"{log_prefix} color range: [{colors_concat.min(axis=0)}, {colors_concat.max(axis=0)}]")
    
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


def render_point_cloud_from_view(
    pred_points_world: np.ndarray,
    pred_colors: np.ndarray,
    gt_points_world: np.ndarray,
    gt_colors: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    image_height: int,
    image_width: int,
    view_idx: int,
    save_path: str,
    point_size: float = 2.0,
) -> bool:
    """
    从指定相机视角渲染点云并保存为图像（仅渲染该视角可见的点）。
    
    Args:
        pred_points_world: 预测点云，世界坐标系，形状 (N, 3)
        pred_colors: 预测点云颜色，形状 (N, 3)，值范围 [0, 1] 或 [0, 255]
        gt_points_world: GT点云，世界坐标系，形状 (M, 3)
        gt_colors: GT点云颜色，形状 (M, 3)，值范围 [0, 1] 或 [0, 255]
        intrinsic: 相机内参矩阵，形状 (3, 3)
        extrinsic: 相机外参矩阵 [R|t]，形状 (3, 4)，世界坐标系到相机坐标系
        image_height: 图像高度
        image_width: 图像宽度
        view_idx: 视角索引（用于日志输出）
        save_path: 保存路径
        point_size: 点大小（默认2.0）
    
    Returns:
        bool: 是否成功保存
    """
    log_prefix = "[render_point_cloud_from_view]"
    
    try:
        import open3d as o3d
    except ImportError:
        print(f"{log_prefix} ERROR: Open3D not available")
        return False
    
    # --- 3D→2D 可见性裁剪，筛选该视角可见点 ---------------------------------
    K = intrinsic  # (3,3)
    E = extrinsic  # (3,4) [R|t] 世界坐标系到相机坐标系
    R = E[:, :3]  # (3,3)
    t = E[:, 3:4]  # (3,1)
    
    H, W = image_height, image_width
    
    # 预测点云：世界坐标 → 相机坐标（用于可见性筛选）
    pts_pred_cam = (R @ pred_points_world.T + t).T  # (N,3) 相机坐标系
    z_pred = pts_pred_cam[:, 2]
    
    # 只保留 z>0 的点
    valid_pred = z_pred > 0
    pts_pred_cam_filtered = pts_pred_cam[valid_pred]
    z_pred_filtered = pts_pred_cam_filtered[:, 2]
    
    # 投影到像素坐标进行可见性筛选
    x_pred = pts_pred_cam_filtered[:, 0] / z_pred_filtered
    y_pred = pts_pred_cam_filtered[:, 1] / z_pred_filtered
    u_pred = K[0, 0] * x_pred + K[0, 2]
    v_pred = K[1, 1] * y_pred + K[1, 2]
    
    in_img_pred = (u_pred >= 0) & (u_pred < W) & (v_pred >= 0) & (v_pred < H)
    # 获取可见点在世界坐标系中的索引
    valid_and_visible_pred = np.where(valid_pred)[0][in_img_pred]
    pts_pred_view_world = pred_points_world[valid_and_visible_pred]  # 使用世界坐标
    cols_pred_view = pred_colors[valid_and_visible_pred]
    
    # GT 点云同样处理（世界坐标 → 当前预测相机系）
    # pts_gt_cam = (R @ gt_points_world.T + t).T  # (M,3) 相机坐标系
    # z_gt = pts_gt_cam[:, 2]
    # valid_gt = z_gt > 0
    # pts_gt_cam_filtered = pts_gt_cam[valid_gt]
    # z_gt_filtered = pts_gt_cam_filtered[:, 2]
    # 
    # x_gt = pts_gt_cam_filtered[:, 0] / z_gt_filtered
    # y_gt = pts_gt_cam_filtered[:, 1] / z_gt_filtered
    # u_gt = K[0, 0] * x_gt + K[0, 2]
    # v_gt = K[1, 1] * y_gt + K[1, 2]
    # 
    # in_img_gt = (u_gt >= 0) & (u_gt < W) & (v_gt >= 0) & (v_gt < H)
    # valid_and_visible_gt = np.where(valid_gt)[0][in_img_gt]
    # pts_gt_view_world = gt_points_world[valid_and_visible_gt]  # 使用世界坐标
    # cols_gt_view = gt_colors[valid_and_visible_gt]
    
    if len(pts_pred_view_world) == 0:
        print(f"{log_prefix} view {view_idx}: no visible points, skip rendering")
        return False
    
    # 颜色归一化到 [0,1]
    if cols_pred_view.size > 0 and cols_pred_view.max() > 1.0:
        cols_pred_view = cols_pred_view / 255.0
    # if cols_gt_view.size > 0 and cols_gt_view.max() > 1.0:
    #     cols_gt_view = cols_gt_view / 255.0
    cols_pred_view = np.clip(cols_pred_view, 0.0, 1.0)
    # cols_gt_view = np.clip(cols_gt_view, 0.0, 1.0)
    
    # --- 构建 Open3D 点云 ------------------------------------------------
    # Open3D渲染器期望点云在世界坐标系中，然后用extrinsic矩阵进行变换
    pcd_pred = o3d.geometry.PointCloud()
    if len(pts_pred_view_world) > 0:
        pcd_pred.points = o3d.utility.Vector3dVector(pts_pred_view_world.astype(np.float64))
        pcd_pred.colors = o3d.utility.Vector3dVector(cols_pred_view.astype(np.float64))
    
    # pcd_gt = o3d.geometry.PointCloud()
    # if len(pts_gt_view_world) > 0:
    #     pcd_gt.points = o3d.utility.Vector3dVector(pts_gt_view_world.astype(np.float64))
    #     pcd_gt.colors = o3d.utility.Vector3dVector(cols_gt_view.astype(np.float64))
    
    cam_params = o3d.camera.PinholeCameraParameters()
    
    # 内参
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    cam_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    
    # 外参: 世界坐标系到相机坐标系的变换矩阵 [R|t] -> 4x4
    extrinsic_4x4 = np.eye(4, dtype=np.float64)
    extrinsic_4x4[:3, :4] = extrinsic
    cam_params.extrinsic = extrinsic_4x4
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        visible=False, width=int(W), height=int(H), window_name=f"PCD_Render_{view_idx}"
    )
    vis.add_geometry(pcd_pred)
    # vis.add_geometry(pcd_gt)
    
    # 渲染设置：黑色背景、增大点大小，避免"看起来全白"
    opt = vis.get_render_option()
    opt.background_color = np.array([0.0, 0.0, 0.0])
    opt.point_size = point_size
    
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
    
    # 确保渲染完成
    vis.poll_events()
    vis.update_renderer()
    # 多次poll和update确保渲染稳定
    for _ in range(5):
        vis.poll_events()
        vis.update_renderer()
    
    vis.capture_screen_image(save_path, do_render=True)
    print(f"{log_prefix} Saved rendered image for view {view_idx}: {save_path}")
    vis.destroy_window()
    
    return True


def display_rgb_depth_overlay(
    rgb_image: np.ndarray,
    depth_image: np.ndarray,
    view_idx: int = 1,
    figsize: tuple = (18, 5),
) -> None:
    """
    显示RGB原图、depth原图和RGB+Depth热图叠加的三张图像。
    
    Args:
        rgb_image: RGB图像，形状为 (H, W, 3)，值范围可以是 [0, 1] 或 [0, 255]
        depth_image: Depth图像，形状为 (H, W)，值范围可以是任意正数
        view_idx: 视角索引，用于标题显示（默认1）
        figsize: 图像大小，默认为 (18, 5)
    """
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    # 确保RGB图像值在 [0, 1] 范围内
    rgb_normalized = rgb_image.copy()
    if rgb_normalized.max() > 1.0:
        rgb_normalized = rgb_normalized / 255.0
    rgb_normalized = np.clip(rgb_normalized, 0, 1)

    # 第一张：RGB原图
    axs[0].imshow(rgb_normalized)
    axs[0].set_title(f"RGB Image (View {view_idx})")
    axs[0].axis("off")

    # 第二张：Depth原图（灰度显示，自动归一化）
    axs[1].imshow(depth_image, cmap='gray')
    axs[1].set_title(f"Depth Image (View {view_idx})")
    axs[1].axis("off")
    
    # 第三张：RGB + Depth热图叠加（简化版本）
    axs[2].imshow(rgb_normalized)
    axs[2].imshow(depth_image, cmap='jet', alpha=0.5)
    axs[2].set_title(f"RGB + Depth Heatmap (View {view_idx})")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()


def display_depth_comparison(pred_depth: np.ndarray,
                           gt_depth: np.ndarray,
                           view_idx: int = 0,
                           figsize: tuple = (18, 5)) -> None:
    """
    显示预测深度图、真实深度图以及它们之间的绝对差异对比。

    Args:
        pred_depth: 预测深度图，形状为 (H, W)
        gt_depth: 真实深度图，形状为 (H, W)
        view_idx: 视角索引，用于标题显示（默认0）
        figsize: 图像大小，默认为 (18, 5)
    """
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    # 计算显示范围（统一颜色映射）
    vmin = min(pred_depth.min(), gt_depth.min())
    vmax = max(pred_depth.max(), gt_depth.max())

    # 第一张：预测深度图
    axs[0].imshow(pred_depth, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0].set_title(f"Predicted Depth (View {view_idx})")
    axs[0].axis('off')

    # 第二张：真实深度图
    axs[1].imshow(gt_depth, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[1].set_title(f"GT Depth (View {view_idx})")
    axs[1].axis('off')

    # 第三张：绝对差异
    diff_img = np.abs(pred_depth - gt_depth)
    diff_img = pred_depth - gt_depth
    axs[2].imshow(diff_img)
    axs[2].set_title(f"Depth Absolute Diff (View {view_idx})")
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()


def display_trimesh(mesh, width=800, height=600, window_name="Open3D Mesh Viewer",
                    point_size=2.0, background_color=None, show_coordinate_frame=True,
                    coordinate_frame_size=50.0, verbose=True):
    """
    显示trimesh对象的可视化
    
    参数:
        mesh: trimesh.Trimesh 对象，要显示的mesh
        width: int, 窗口宽度（默认800）
        height: int, 窗口高度（默认600）
        window_name: str, 窗口名称（默认"Open3D Mesh Viewer"）
        point_size: float, 点云渲染的点大小（默认2.0）
        background_color: np.ndarray, 背景颜色RGB值，形状(3,)，默认None使用白色[1,1,1]
        show_coordinate_frame: bool, 是否显示坐标轴（默认True）
        coordinate_frame_size: float, 坐标轴大小（默认50.0）
        verbose: bool, 是否打印信息（默认True）
    
    返回:
        bool, 是否成功显示（True表示成功，False表示失败或跳过）
    """
    # 检查是否有顶点可以可视化
    if len(mesh.vertices) == 0:
        if verbose:
            print("警告: mesh没有顶点，跳过可视化")
        return False
    
    # 将trimesh对象mesh转换为Open3D点云对象
    mesh_vertices = np.asarray(mesh.vertices)
    o3d_point_cloud_from_mesh = o3d.geometry.PointCloud()
    o3d_point_cloud_from_mesh.points = o3d.utility.Vector3dVector(mesh_vertices)
    
    vis = o3d.visualization.Visualizer()
    window_created = vis.create_window(window_name=window_name, width=width, height=height)
    
    if not window_created:
        if verbose:
            print("警告: 无法创建Open3D可视化窗口，跳过可视化")
        return False
    
    # 添加点云
    vis.add_geometry(o3d_point_cloud_from_mesh)
    
    # 添加坐标轴（如果需要）
    if show_coordinate_frame:
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=coordinate_frame_size))
    
    # 设置渲染选项
    render_option = vis.get_render_option()
    if render_option is not None:
        if background_color is None:
            background_color = np.array([1, 1, 1])
        render_option.background_color = background_color
        render_option.point_size = point_size
    
    # 运行可视化
    vis.run()
    vis.destroy_window()
    
    return True
