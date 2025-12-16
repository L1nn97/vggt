import open3d as o3d
import numpy as np

def register_point_clouds_open3d_icp(pred_points, gt_points):
    log_prefix = "[open3d_icp]"
    print(f"{log_prefix} start")
    print(f"{log_prefix} pred_points shape: {pred_points.shape}, "
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

    pred_cloud.transform(icp_result.transformation)
    transformed_pred_points = np.asarray(pred_cloud.points)

    print(f"{log_prefix} transformed_pred_points shape: {transformed_pred_points.shape}")
    print(f"{log_prefix} done")

    return transformed_pred_points, icp_result.transformation