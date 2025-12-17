import open3d as o3d
import numpy as np

def register_point_clouds_open3d_icp(pred_points_original, gt_points_original):
    log_prefix = "[open3d_icp]"
    print(f"{log_prefix} start")
    print(f"{log_prefix} pred_points_original shape: {pred_points_original.shape}, "
          f"gt_points_original shape: {gt_points_original.shape}")

    # Randomly select 10000 points from pred_points and gt_points if they have more than 10000 points
    num_sample = 10000
    pred_num = pred_points_original.shape[0]
    gt_num = gt_points_original.shape[0]
    
    # Initialize with original points (will be overwritten if sampling is needed)
    pred_points = pred_points_original
    gt_points = gt_points_original
    
    if pred_num > num_sample and gt_num > num_sample:
        idx_pred = np.random.choice(pred_num, num_sample, replace=False)
        idx_gt = np.random.choice(gt_num, num_sample, replace=False)
        pred_points = pred_points_original[idx_pred]
        gt_points = gt_points_original[idx_gt]
    elif pred_num > num_sample:
        idx_pred = np.random.choice(pred_num, num_sample, replace=False)
        pred_points = pred_points_original[idx_pred]
        # gt_points already set to gt_points_original above
    elif gt_num > num_sample:
        idx_gt = np.random.choice(gt_num, num_sample, replace=False)
        gt_points = gt_points_original[idx_gt]
        # pred_points already set to pred_points_original above
    
    print(f"{log_prefix} using pred_points shape: {pred_points.shape}, "
          f"gt_points shape: {gt_points.shape}")


    pred_cloud = o3d.geometry.PointCloud()
    pred_cloud.points = o3d.utility.Vector3dVector(pred_points)
    gt_cloud = o3d.geometry.PointCloud()
    gt_cloud.points = o3d.utility.Vector3dVector(gt_points)

    print(f"{log_prefix} running ICP (max_iteration=100, distance_threshold=10.0)")
    icp_result = o3d.pipelines.registration.registration_icp(
        pred_cloud,
        gt_cloud,
        10.0,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )

    print(f"{log_prefix} fitness: {icp_result.fitness:.4f}, "
          f"inlier_rmse: {icp_result.inlier_rmse:.4f}")
    print(f"{log_prefix} transformation:\n{icp_result.transformation}")

    transformed_pred_cloud = o3d.geometry.PointCloud()
    transformed_pred_cloud.points = o3d.utility.Vector3dVector(pred_points_original)
    transformed_pred_cloud.transform(icp_result.transformation)
    transformed_pred_points = np.asarray(transformed_pred_cloud.points)

    print(f"{log_prefix} transformed_pred_points shape: {transformed_pred_points.shape}")
    print(f"{log_prefix} done")

    return transformed_pred_points, icp_result.transformation