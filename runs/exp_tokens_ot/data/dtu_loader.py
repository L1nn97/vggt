import os, sys
import torch
import glob
import numpy as np
import cv2
import open3d as o3d
from typing import List, Tuple, Dict, Optional

current_work_dir = os.getcwd()
print(f"Current working directory: {current_work_dir}")

sys.path += [
    os.path.join(current_work_dir),
    # os.path.join(current_work_dir, "../vggt"),
    # os.path.join(current_work_dir, "../spann3r")
]
from vggt.utils.load_fn import load_and_preprocess_images


def load_cam_mvsnet(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取MVSNet格式的相机文件，返回:
      - intrinsic_3x3: (3,3)
      - extrinsic_4x4: (4,4)
    文件格式示例:
      extrinsic
      4x4矩阵
      intrinsic
      3x3矩阵
      depth_start depth_interval depth_num depth_end (可选)
    """
    with open(file_path, 'r') as f:
        words = f.read().split()

    cam = np.zeros((2, 4, 4), dtype=np.float32)

    # extrinsic 4x4
    for i in range(4):
        for j in range(4):
            cam[0, i, j] = float(words[1 + 4 * i + j])

    # intrinsic 3x3
    for i in range(3):
        for j in range(3):
            cam[1, i, j] = float(words[18 + 3 * i + j])

    extrinsic = cam[0].astype(np.float32)  # (4,4)
    intrinsic_4x4 = cam[1].astype(np.float32)  # (4,4) 但仅前3x3有效
    intrinsic_3x3 = intrinsic_4x4[:3, :3].copy()

    return intrinsic_3x3, extrinsic


def adjust_intrinsic_for_resize(K: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    """
    根据图像缩放比例调整内参矩阵
    
    Args:
        K: 原始内参矩阵 (3,3)
        scale_x: x方向缩放比例 (new_width / original_width)
        scale_y: y方向缩放比例 (new_height / original_height)
    
    Returns:
        调整后的内参矩阵 (3,3)
    """
    K_adjusted = K.copy()
    K_adjusted[0, 0] *= scale_x  # fx
    K_adjusted[1, 1] *= scale_y  # fy
    K_adjusted[0, 2] *= scale_x  # cx
    K_adjusted[1, 2] *= scale_y  # cy
    return K_adjusted


def adjust_intrinsic_for_crop(K: np.ndarray, crop_x: int, crop_y: int) -> np.ndarray:
    """
    裁剪后需要更新主点坐标 (cx, cy)。

    Args:
        K: (3,3) 内参
        crop_x: 在 x 方向裁掉的像素（左侧）
        crop_y: 在 y 方向裁掉的像素（上方）
    """
    K_adjusted = K.copy()
    K_adjusted[0, 2] -= crop_x
    K_adjusted[1, 2] -= crop_y
    return K_adjusted


def default_image_loader(image_paths: List[str], device: Optional[str] = None, target_size: Optional[Tuple[int, int]] = None):
    """
    加载并可选缩放图像
    
    Args:
        target_size: (width, height) 目标尺寸，如果为None则保持原尺寸
    """
    if load_and_preprocess_images is not None and torch is not None and target_size is None:
        # 如果不需要缩放，直接使用VGGT的预处理
        imgs = load_and_preprocess_images(image_paths)
        if device:
            imgs = imgs.to(device)
        return imgs

    images = []
    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {p}")
        
        # 如果需要缩放
        if target_size is not None:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        
        # 归一化到 [0, 1] 并转换为 float32，参考 load_and_preprocess_images 的处理
        img = img.astype(np.float32) / 255.0
        images.append(img)
    
    # 如果需要转换为tensor
    if torch is not None and device is not None:
        # 转换为tensor: (H, W, 3) -> (3, H, W)，并归一化到 [0, 1]
        imgs_tensor = torch.stack([torch.from_numpy(img).permute(2, 0, 1) for img in images])
        return imgs_tensor.to(device)
    
    return images  # List[np.ndarray(H,W,3)]

def default_mask_loader(mask_paths: List[str], device: Optional[str] = None, target_size: Optional[Tuple[int, int]] = None):
    """
    加载并可选缩放掩码
    """
    if load_and_preprocess_images is not None and torch is not None and target_size is None:
        # 如果不需要缩放，直接使用VGGT的预处理
        masks = load_and_preprocess_images(mask_paths)
        if device:
            masks = masks.to(device)
        return masks
    masks = []
    for p in mask_paths:
        mask = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask: {p}")
        if target_size is not None:
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_LINEAR)
        masks.append(mask)

    if torch is not None and device is not None:
        masks_tensor = torch.stack([torch.from_numpy(mask) for mask in masks])
        return masks_tensor.to(device)
    return masks

def default_depth_loader(depth_paths: List[str], device: Optional[str] = None, target_size: Optional[Tuple[int, int]] = None):
    """
    加载并可选缩放深度
    深度文件是 .npy 格式，需要使用 np.load 加载
    """
    depths = []
    for p in depth_paths:
        depth = np.load(p)
        if depth is None:
            raise FileNotFoundError(f"Failed to read depth: {p}")
        
        # 如果深度图是3D (H, W, 1)，压缩为2D
        if len(depth.shape) == 3 and depth.shape[2] == 1:
            depth = depth.squeeze(-1)
        
        if target_size is not None:
            depth = cv2.resize(depth, target_size, interpolation=cv2.INTER_LINEAR)
        depths.append(depth)
    
    # 如果需要转换为tensor
    if torch is not None and device is not None:
        depths_tensor = torch.stack([torch.from_numpy(depth) for depth in depths])
        return depths_tensor.to(device)
    
    return depths

def image_name_to_cam_name(image_name: str) -> str:
    """
    将图像文件名映射为对应的MVSNet cam文件名。
    规则：'{8位序号}.jpg' -> '{8位序号}_cam.txt'
    """
    base = os.path.basename(image_name)
    stem, _ = os.path.splitext(base)
    return f"{stem}_cam.txt"

def image_name_to_mask_name(image_name: str) -> str:
    """
    将图像文件名映射为对应的二进制掩码文件名。
    规则：'{8位序号}.jpg' -> '{8位序号}.png'
    """
    base = os.path.basename(image_name)
    stem, _ = os.path.splitext(base)
    return f"{stem}.png"

def image_name_to_depth_name(image_name: str) -> str:
    """
    将图像文件名映射为对应的深度文件名。
    规则：'{8位序号}.jpg' -> '{8位序号}.npy'
    """
    base = os.path.basename(image_name)
    stem, _ = os.path.splitext(base)
    return f"{stem}.npy"

class DTUScanLoader:
    """
    加载 data/dtu/dtu_mvs/scan{scan_id} 下的图像与MVSNet相机。

    目录结构假定：
      {root}/scan{scan_id}/images/*.jpg 或 *.png
      {root}/scan{scan_id}/cams/{8位}_cam.txt
    """

    def __init__(
        self,
        dtu_mvs_root: str,
        scan_id: int,
        num_views: Optional[int] = None,
        step: int = 1,
        device: Optional[str] = None,
        target_size: Optional[Tuple[int, int]] = None,
        keep_ratio: bool = True,
    ):
        self.scan_dir = os.path.join(dtu_mvs_root, f"scan{int(scan_id)}")
        self.images_dir = os.path.join(self.scan_dir, "images")
        self.cams_dir = os.path.join(self.scan_dir, "cams")
        self.masks_dir = os.path.join(self.scan_dir, "binary_masks")
        self.depths_dir = os.path.join(self.scan_dir, "depths")

        self.num_views = num_views
        self.step = max(1, int(step))
        self.device = device
        self.requested_target_size = target_size
        self.keep_ratio = bool(keep_ratio)

        if not os.path.isdir(self.scan_dir):
            raise FileNotFoundError(f"Scan directory not found: {self.scan_dir}")
        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not os.path.isdir(self.cams_dir):
            raise FileNotFoundError(f"Cams directory not found: {self.cams_dir}")
        if not os.path.isdir(self.masks_dir):
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")
        if not os.path.isdir(self.depths_dir):
            raise FileNotFoundError(f"Depths directory not found: {self.depths_dir}")

        self.image_paths = self._collect_images()
        self.mask_paths = self._collect_masks()
        self.depth_paths = self._collect_depths()
        self.point_path = os.path.join(self.scan_dir, f"{int(scan_id):03d}_pcd.ply")

        self.image_paths = self._subsample(self.image_paths, self.num_views, self.step)

        # 对应的cam文件路径（与图像顺序一致）
        self.cam_paths = [os.path.join(self.cams_dir, image_name_to_cam_name(p)) for p in self.image_paths]
        self.mask_paths = [os.path.join(self.masks_dir, image_name_to_mask_name(p)) for p in self.image_paths]
        self.depth_paths = [os.path.join(self.depths_dir, image_name_to_depth_name(p)) for p in self.image_paths]

        self._check_cam_files()

        # 读取一张图像，缓存原始尺寸
        first_img = cv2.imread(self.image_paths[0], cv2.IMREAD_COLOR)
        if first_img is None:
            raise RuntimeError(f"Failed to read first image: {self.image_paths[0]}")
        self.H_orig, self.W_orig = first_img.shape[:2]
        
        # 计算缩放 + 裁剪参数
        self.resize_size: Optional[Tuple[int, int]] = None  # 缩放后的尺寸
        self.crop_offsets: Tuple[int, int] = (0, 0)         # (crop_x, crop_y)
        self.crop_needed: bool = False

        if self.requested_target_size is not None:
            (
                resize_w,
                resize_h,
                self.scale_x,
                self.scale_y,
            ) = self._compute_resize_params(self.requested_target_size, self.keep_ratio)
            self.resize_size = (resize_w, resize_h)

            if self.keep_ratio:
                crop_x, crop_y = self._compute_crop_offsets(
                    self.resize_size, self.requested_target_size
                )
                self.crop_offsets = (crop_x, crop_y)
                self.crop_needed = crop_x > 0 or crop_y > 0
                final_w, final_h = map(int, self.requested_target_size)
            else:
                self.crop_needed = False  # keep_ratio=False 时不裁剪
                final_w, final_h = resize_w, resize_h

            self.W, self.H = final_w, final_h
        else:
            self.W, self.H = self.W_orig, self.H_orig
            self.scale_x = self.scale_y = 1.0

        # target_size 表示 loader 输出（裁剪后）的尺寸
        self.target_size = (self.W, self.H)

    def _compute_resize_params(
        self,
        requested_size: Tuple[int, int],
        keep_ratio: bool,
    ) -> Tuple[int, int, float, float]:
        """
        根据是否保持纵横比决定缩放后的尺寸与缩放系数。
        返回: (new_W, new_H, scale_x, scale_y)
        """
        req_w, req_h = map(int, requested_size)
        if req_w <= 0 or req_h <= 0:
            raise ValueError(f"Invalid target_size: {requested_size}")

        if keep_ratio:
            scale = max(req_w / self.W_orig, req_h / self.H_orig)
            if scale <= 0:
                raise ValueError(f"Computed non-positive scale {scale} from target_size={requested_size}")
            new_w = max(1, int(round(self.W_orig * scale)))
            new_h = max(1, int(round(self.H_orig * scale)))
            return new_w, new_h, scale, scale

        new_w, new_h = req_w, req_h
        scale_x = new_w / self.W_orig
        scale_y = new_h / self.H_orig
        return new_w, new_h, scale_x, scale_y

    def _compute_crop_offsets(
        self,
        resized_size: Tuple[int, int],
        target_size: Tuple[int, int],
    ) -> Tuple[int, int]:
        """
        计算中心裁剪偏移量 (crop_x, crop_y)。
        """
        resized_w, resized_h = map(int, resized_size)
        target_w, target_h = map(int, target_size)

        if target_w > resized_w or target_h > resized_h:
            raise ValueError(
                f"Target size {target_size} larger than resized size {resized_size}, 无法裁剪。"
            )

        crop_x = max(0, (resized_w - target_w) // 2)
        crop_y = max(0, (resized_h - target_h) // 2)
        return crop_x, crop_y

    def _crop_tensor_or_list(self, data):
        """
        针对 tensor / numpy list 做中心裁剪。
        """
        if not self.crop_needed:
            return data

        crop_x, crop_y = self.crop_offsets
        target_w, target_h = self.target_size

        if isinstance(data, torch.Tensor):
            if data.dim() == 4:
                return data[:, :, crop_y : crop_y + target_h, crop_x : crop_x + target_w]
            if data.dim() == 3:
                return data[:, crop_y : crop_y + target_h, crop_x : crop_x + target_w]
            raise ValueError(f"Unsupported tensor shape for cropping: {data.shape}")

        if isinstance(data, list):
            cropped = []
            for item in data:
                if item.ndim == 3:
                    cropped.append(item[crop_y : crop_y + target_h, crop_x : crop_x + target_w, ...])
                elif item.ndim == 2:
                    cropped.append(item[crop_y : crop_y + target_h, crop_x : crop_x + target_w])
                else:
                    raise ValueError(f"Unsupported array shape for cropping: {item.shape}")
            return cropped

        return data

    def _collect_images(self) -> List[str]:
        jpgs = glob.glob(os.path.join(self.images_dir, "*.jpg"))
        pngs = glob.glob(os.path.join(self.images_dir, "*.png"))
        all_imgs = sorted(jpgs + pngs)
        if len(all_imgs) == 0:
            raise FileNotFoundError(f"No images found under {self.images_dir}")
        return all_imgs
    
    def _collect_masks(self) -> List[str]:
        masks = glob.glob(os.path.join(self.masks_dir, "*.png"))
        if len(masks) == 0:
            raise FileNotFoundError(f"No masks found under {self.masks_dir}")
        return masks
    
    def _collect_depths(self) -> List[str]:
        depths = glob.glob(os.path.join(self.depths_dir, "*.npy"))
        if len(depths) == 0:
            raise FileNotFoundError(f"No depths found under {self.depths_dir}")
        return depths

    @staticmethod
    def _subsample(items: List[str], num_views: Optional[int], step: int) -> List[str]:
        items = items[::step]
        if num_views is not None and num_views > 0:
            items = items[:num_views]
        return items

    def _check_cam_files(self):
        missing = [p for p in self.cam_paths if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(f"Missing cam files: {missing[:3]} ... total={len(missing)}")

    def load_images(self):
        imgs = default_image_loader(self.image_paths, self.device, self.resize_size)
        return self._crop_tensor_or_list(imgs)

    def load_masks(self):
        masks = default_mask_loader(self.mask_paths, self.device, self.resize_size)
        return self._crop_tensor_or_list(masks)

    def load_depths(self):
        depths = default_depth_loader(self.depth_paths, self.device, self.resize_size)
        return self._crop_tensor_or_list(depths)
    
    def load_points(self):
        pcd = o3d.io.read_point_cloud(self.point_path)

        # Convert to numpy arrays
        points = np.asarray(pcd.points)
        colors = None
        try:
            if pcd.has_colors():
                colors = np.asarray(pcd.colors)
        except Exception:
            colors = None

        return torch.from_numpy(points).to(self.device), torch.from_numpy(colors).to(self.device)

    def load_cameras(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回:
          - intrinsics: (S, 3, 3) - 已根据图像缩放调整
          - extrinsics: (S, 4, 4) - 保持不变
        """
        intrinsics, extrinsics = [], []
        for cam_path in self.cam_paths:
            K, E = load_cam_mvsnet(cam_path)
            
            # 如果图像被缩放了，调整内参矩阵
            if self.scale_x != 1.0 or self.scale_y != 1.0:
                K = adjust_intrinsic_for_resize(K, self.scale_x, self.scale_y)
            
            if self.crop_needed:
                K = adjust_intrinsic_for_crop(K, *self.crop_offsets)

            intrinsics.append(K)
            extrinsics.append(E)  # 外参矩阵不变
        return np.stack(intrinsics, axis=0), np.stack(extrinsics, axis=0)

    def info(self) -> Dict:
        return {
            "scan_dir": self.scan_dir,
            "num_images": len(self.image_paths),
            "original_size": (self.H_orig, self.W_orig),
            "current_size": (self.H, self.W),
            "scale_factors": (self.scale_x, self.scale_y),
            "images_dir": self.images_dir,
            "cams_dir": self.cams_dir,
            "masks_dir": self.masks_dir,
            "depths_dir": self.depths_dir,
        }

    def num_tokens_per_view(self, patch_size: int) -> int:
        """
        返回每个视图的patch token数量 P = (H/patch_size) * (W/patch_size)
        用于将投影到像素坐标映射到token索引。
        """
        H_p = self.H // patch_size
        W_p = self.W // patch_size
        return int(H_p * W_p)

    def project_points(self, points_world: np.ndarray, K: np.ndarray, E: np.ndarray) -> np.ndarray:
        """
        将世界坐标点投影到像素坐标:
          points_world: (N,3)
          K: (3,3)
          E: (4,4)  (camera-to-world的逆? 视MVSNet文件而定)
        注意: 本函数假定E为外参矩阵 (world->camera) 的逆为 camera_pose。
        若需要，请在调用前做适配。
        """
        # 这里E通常是 camera extrinsic 矩阵 (world->camera)
        # 将点扩展为齐次
        N = points_world.shape[0]
        homog = np.concatenate([points_world, np.ones((N, 1), dtype=np.float32)], axis=1)  # (N,4)
        pc = (E @ homog.T).T  # (N,4) camera coords (齐次)

        # 归一化
        xyz = pc[:, :3]
        z = xyz[:, 2:3].copy()
        uv = (K @ xyz.T).T  # (N,3)
        uv = uv[:, :2] / np.clip(z, 1e-6, None)

        return np.concatenate([uv, z], axis=1)  # (N,3): u, v, depth

    def pixel_to_patch_index(self, u: float, v: float, patch_size: int) -> Optional[int]:
        """
        将像素坐标(u,v)映射到patch网格索引，如果越界则返回None。
        """
        H_p, W_p = self.H // patch_size, self.W // patch_size
        if u < 0 or v < 0 or u >= self.W or v >= self.H:
            return None
        pu, pv = int(u // patch_size), int(v // patch_size)
        if pu < 0 or pv < 0 or pu >= W_p or pv >= H_p:
            return None
        return pv * W_p + pu

    def token_index(self, view_idx: int, patch_idx: int, P: int) -> int:
        """
        全局token索引: token = view_idx * P + patch_idx
        """
        return view_idx * P + patch_idx


if __name__ == "__main__":
    # 简单自检示例
    dtu_root = "/home/vision/ws/datasets/SampleSet/dtu_mvs/"
    scan_id = 1
    target_size = (518, 350)

    loader = DTUScanLoader(dtu_root, scan_id, num_views=4, step=1, device=None, target_size=target_size, keep_ratio=True)

    info = loader.info()
    imgs = loader.load_images()
    masks = loader.load_masks()
    depths = loader.load_depths()
    K_all, E_all = loader.load_cameras()
    P = loader.num_tokens_per_view(patch_size=14)

    print("--------------------------------"        )
    print(f"Loaded {len(imgs)} images")
    print("Loader info:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    
    print("imgs[0].shape: ", imgs[0].shape)
    print("masks[0].shape: ", masks[0].shape)
    print("depths[0].shape: ", depths[0].shape)
    print(f"Tokens per view: {P}")

    # 使用 imshow 显示图片
    for i in range(len(imgs)):
        image = imgs[i]
        # 如果是 RGB，转换为 BGR 用于 cv2.imshow
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = image[..., [2, 1, 0]]  # RGB to BGR
        if image.mean() < 1.0:
            image *= 255.0
        image = image.astype(np.uint8)
        
        cv2.imshow(f"Image {i}", image)

        # depth = depths[i]
        # depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255
        # depth_normalized = depth_normalized.astype(np.uint8)
        # cv2.imshow(f"Depth {i}", depth_normalized)

        # mask = masks[i].astype(np.uint8)
        # cv2.imshow(f"Mask {i}", mask)
        
        print(f"Press any key to view next image (image {i+1}/{len(imgs)})...")
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()

