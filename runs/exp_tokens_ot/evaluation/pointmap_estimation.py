import numpy as np
import open3d as o3d
from scipy.linalg import svd

# Import umeyama_alignment for internal use in eval_trajectory
def umeyama_alignment(src, dst, estimate_scale=True):
    # Ensure inputs have correct shape
    assert (
        src.shape == dst.shape
    ), f"Input shapes don't match: src {src.shape}, dst {dst.shape}"
    assert src.shape[0] == 3, f"Expected point cloud dimension (3,N), got {src.shape}"

    # Compute centroids
    src_mean = src.mean(axis=1, keepdims=True)
    dst_mean = dst.mean(axis=1, keepdims=True)

    # Center the point clouds
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    # Compute covariance matrix
    cov = dst_centered @ src_centered.T

    try:
        # Singular Value Decomposition
        U, D, Vt = svd(cov)
        V = Vt.T

        # Handle reflection case
        det_UV = np.linalg.det(U @ V.T)
        S = np.eye(3)
        if det_UV < 0:
            S[2, 2] = -1

        # Compute rotation matrix
        R = U @ S @ V.T

        if estimate_scale:
            # Compute scale factor - fix dimension issue
            src_var = np.sum(src_centered * src_centered)
            if src_var < 1e-10:
                print(
                    "Warning: Source point cloud variance close to zero, setting scale factor to 1.0"
                )
                scale = 1.0
            else:
                # Fix potential dimension issue with np.diag(S)
                # Use diagonal elements directly
                scale = np.sum(D * np.diag(S)) / src_var
        else:
            scale = 1.0

        # Compute translation vector
        t = dst_mean.ravel() - scale * (R @ src_mean).ravel()

        return scale, R, t

    except Exception as e:
        print(f"Error in umeyama_alignment computation: {e}")
        print(
            "Returning default transformation: scale=1.0, rotation=identity matrix, translation=centroid difference"
        )
        # Return default transformation
        scale = 1.0
        R = np.eye(3)
        t = (dst_mean - src_mean).ravel()
        return scale, R, t
