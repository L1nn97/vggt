#!/usr/bin/env python3
"""
DTU数据集完整预处理脚本 - 将DTU原始数据集转换为spann3r所需格式

使用方法:
python process_dtu_complete.py --scan_ids 1 4 9 10 11 12 13 15 23 24 29 32 33 34 48 49 62 75 77 110 114 118

或者处理所有扫描:
python process_dtu_complete.py --scan_ids all
"""

import os
import sys
import shutil
import cv2
import numpy as np
import trimesh
import pyrender
import open3d as o3d
from tqdm import tqdm
import argparse
from pathlib import Path
import json
import scipy

np.set_printoptions(suppress=True) 

def load_cam_mvsnet(file_path):
    """ read camera txt file """
    with open(file_path, 'r') as file:
        extrinsic = np.zeros((4, 4))
        intrinsic = np.zeros((3, 3))
        words = file.read().split()
        # read extrinsic
        for i in range(0, 4):
            for j in range(0, 4):
                extrinsic_index = 4 * i + j + 1
                extrinsic[i][j] = words[extrinsic_index]

        # read intrinsic
        for i in range(0, 3):
            for j in range(0, 3):
                intrinsic_index = 3 * i + j + 18
                intrinsic[i][j] = words[intrinsic_index]
        # print('extrinsic:', extrinsic)
        # print('intrinsic:', intrinsic)

        return intrinsic, extrinsic

def convert_dtu_camera_to_mvsnet(dtu_cam_path, output_path, K):
    """将DTU相机参数转换为MVSNet格式"""
    with open(dtu_cam_path, 'r') as f:
        lines = f.readlines()
    
    # 解析DTU格式的相机参数
    extrinsic = np.zeros((4, 4))
    
    # 读取外参矩阵的前3行
    for i in range(3):
        values = list(map(float, lines[i].strip().split()))
        extrinsic[i] = values
    
    # 第4行是齐次坐标
    extrinsic[3] = [0, 0, 0, 1]
    
    # DTU的内参参数 (从Calib_Results_right.m中获取)
    intrinsic = K

    Rt = np.linalg.inv(intrinsic) @ extrinsic[:3, :]
    extrinsic[:3, :] = Rt

    # print('Rt:', Rt)
    # print('extrinsic:', extrinsic)
    # print('intrinsic:', intrinsic)
    
    # 写入MVSNet格式
    with open(output_path, 'w') as f:
        f.write('extrinsic\n')
        for i in range(4):
            for j in range(4):
                f.write(f'{extrinsic[i, j]:.6f} ')
            f.write('\n')
        f.write('\n')
        
        f.write('intrinsic\n')
        for i in range(3):
            for j in range(3):
                f.write(f'{intrinsic[i, j]:.6f} ')
            f.write('\n')
        f.write('\n')
        
def render_depth_maps(mesh, poses, K, H, W, near=0.01, far=5000.0):
    """使用PyRender渲染深度图"""
    try:
        # 设置环境变量以避免显示问题
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene = pyrender.Scene()
        scene.add(mesh)
        camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=near, zfar=far)
        camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
        scene.add_node(camera_node)
        renderer = pyrender.OffscreenRenderer(W, H)
        render_flags = pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.DEPTH_ONLY

        depth_maps = []
        for pose in poses:
            scene.set_pose(camera_node, pose)
            depth = renderer.render(scene, render_flags)
            depth_maps.append(depth)
        print('render depth maps done')
        return depth_maps
    except Exception as e:
        print(f"PyRender渲染失败: {e}")
        # 返回空深度图作为fallback
        return [np.zeros((H, W), dtype=np.float32)]

def filter_mesh_outliers(mesh, nb_neighbors=30, std_ratio=2.0, verbose=True):
    """
    使用统计离群点移除方法对mesh进行滤波，去除飞点
    
    参数:
        mesh: trimesh.Trimesh 对象，输入mesh
        nb_neighbors: int, 用于计算平均距离的邻居点数量（默认30）
        std_ratio: float, 标准差倍数阈值，用于判断离群点（默认2.0）
        verbose: bool, 是否打印详细信息（默认True）
    
    返回:
        mesh_filtered: trimesh.Trimesh 对象，滤波后的mesh
    """
    if verbose:
        print("\n" + "=" * 60)
        print("Mesh Outlier Removal (去除飞点)")
        print("=" * 60)
    
    # 将 mesh 的顶点转为 Open3D 点云
    mesh_vertices = np.asarray(mesh.vertices)
    if verbose:
        print(f"原始 Mesh 顶点数: {len(mesh_vertices)}")
    
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(mesh_vertices)
    
    # 使用统计离群点移除
    _, mask_indices = o3d_pc.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    
    # 转换索引为numpy数组并创建布尔掩码
    mask_indices = np.asarray(list(mask_indices))
    mask_np = np.zeros(len(mesh_vertices), dtype=bool)
    mask_np[mask_indices] = True
    filtered_vertices = mesh_vertices[mask_np]
    
    if verbose:
        print(f"滤波后保留的顶点数: {len(filtered_vertices)}/{len(mesh_vertices)}")
    
    # 获取被保留顶点的原始索引
    ind = np.where(mask_np)[0]
    index_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(ind)}
    
    # 过滤面：只保留所有顶点都被保留的面
    mask_faces = np.all(np.isin(mesh.faces, ind), axis=1)
    filtered_faces = mesh.faces[mask_faces].copy()
    
    # 重映射面索引到新的顶点索引
    for i in range(filtered_faces.shape[0]):
        for j in range(3):
            filtered_faces[i, j] = index_map[filtered_faces[i, j]]
    
    # 构建新的 mesh
    mesh_filtered = trimesh.Trimesh(vertices=filtered_vertices, faces=filtered_faces, process=False)
    
    if verbose:
        print(f"Mesh 滤波后顶点数: {len(mesh_filtered.vertices)}，面数: {len(mesh_filtered.faces)}")
    
    return mesh_filtered

def filter_depth_outliers(depth, kernel_size=5, std_ratio=2.0, median_filter_size=5):
    """
    对深度图进行滤波，去除飞点并生成有效区域的mask
    
    参数:
        depth: np.ndarray, 输入深度图，形状为 (H, W)
        kernel_size: int, 用于统计分析的邻域大小（默认5）
        std_ratio: float, 标准差倍数阈值，用于判断离群点（默认2.0）
        median_filter_size: int, 中值滤波核大小（默认5）
        verbose: bool, 是否打印详细信息（默认True）
    
    返回:
        depth_filtered: np.ndarray, 滤波后的深度图
        mask: np.ndarray, 有效区域的mask，1表示有效，0表示无效
    """
    H, W = depth.shape
    depth_filtered = depth.copy().astype(np.float32)
    
    # 第一步：中值滤波去除小的飞点
    if median_filter_size > 0:
        depth_median = cv2.medianBlur(depth_filtered.astype(np.uint16), median_filter_size).astype(np.float32)
        # 只对非零区域应用中值滤波
        valid_mask = (depth_filtered > 0) & (depth_median > 0)
        depth_filtered[valid_mask] = depth_median[valid_mask]

    # 增加一次形态学，用于去除小的散点（开运算）
    # 这里采用cv2的morphologyEx，先膨胀后腐蚀（开操作），有效地去掉孤立小区域
    if median_filter_size > 0:
        # 保证mask是uint8类型
        morph_mask = valid_mask.astype(np.uint8)
        kernel = np.ones((median_filter_size, median_filter_size), np.uint8)
        # 开操作，去除小的孤立点区域
        morph_mask = cv2.morphologyEx(morph_mask, cv2.MORPH_OPEN, kernel)
        # 用形态学处理后的mask作为最终有效区域
        valid_mask = (morph_mask == 1)
        # 可选：将处理后的mask中为0的像素的深度设置为0
        depth_filtered[~valid_mask] = 0

    return depth_filtered, valid_mask

def get_mesh_from_ply(ply_path, output_path, depth=9, density_thresh=0.1):
    """从PLY点云生成网格"""
    pcd = o3d.io.read_point_cloud(ply_path)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth)
    
    vertices_to_remove = densities < np.quantile(densities, density_thresh)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    o3d.io.write_triangle_mesh(output_path, mesh)
    return mesh

def generate_binary_mask(depth_map, threshold=900):
    """生成二值掩膜"""
    mask = np.ones_like(depth_map) * 255
    mask[depth_map == 0] = 0
    mask[depth_map > threshold] = 0
    return mask

def generate_pair_file(scan_dir, num_views):
    """生成pair.txt文件"""
    pair_path = os.path.join(scan_dir, 'pair.txt')
    
    with open(pair_path, 'w') as f:
        f.write(f'{num_views}\n')
        
        for i in range(num_views):
            # 为每个参考视图选择相邻视图
            f.write(f'{i}\n')
            f.write('10 ')  # 10个相邻视图
            
            # 选择相邻的10个视图
            for j in range(10):
                view_idx = (i + j + 1) % num_views
                score = 1.0 - abs(j) * 0.1  # 距离越近分数越高
                f.write(f'{view_idx} {score:.3f} ')
            f.write('\n')

def get_available_scans(dtu_root):
    """获取可用的扫描列表"""
    rectified_dir = os.path.join(dtu_root, 'Rectified')
    if not os.path.exists(rectified_dir):
        return []
    
    scans = []
    for item in os.listdir(rectified_dir):
        if item.startswith('scan') and os.path.isdir(os.path.join(rectified_dir, item)):
            scan_id = int(item.replace('scan', ''))
            scans.append(scan_id)
    
    return sorted(scans)

def process_single_scan(scan_id, dtu_root, output_root, light_condition=3, skip_existing=True, mesh_type='camp'):
    """处理单个扫描场景"""
    print(f"处理扫描场景 {scan_id}")
    
    # 创建输出目录
    scan_output_dir = os.path.join(output_root, f'scan{scan_id}')
    
    # 检查是否已存在
    if skip_existing and os.path.exists(scan_output_dir):
        print(f"  跳过已存在的扫描 {scan_id}")
        return True
    
    os.makedirs(scan_output_dir, exist_ok=True)
    
    for subdir in ['images', 'cams', 'depths', 'binary_masks']:
        os.makedirs(os.path.join(scan_output_dir, subdir), exist_ok=True)
    
    # 路径设置
    dtu_rectified_dir = os.path.join(dtu_root, 'Rectified', f'scan{scan_id}')
    dtu_calibration_dir = os.path.join(dtu_root, 'Calibration', 'cal18')
    dtu_ply_path = os.path.join(dtu_root, 'Points', 'stl', f'stl{int(scan_id):03d}_total.ply')
    # 支持可选的mesh路径类型: furu, camp, tola
    dtu_mesh_path = os.path.join(dtu_root, 'Surfaces', mesh_type, f'{mesh_type}{int(scan_id):03d}_l3_surf_11_trim_8.ply')

    # 检查输入文件是否存在
    if not os.path.exists(dtu_rectified_dir):
        print(f"  错误: 未找到扫描 {scan_id} 的Rectified目录")
        return False
    
    if not os.path.exists(dtu_ply_path):
        print(f"  错误: 未找到扫描 {scan_id} 的PLY文件")
        return False
    
    if not os.path.exists(dtu_mesh_path):
        print(f"  错误: 未找到扫描 {scan_id} 的MESH文件")
        return False

    # 1. 获取点云和mesh
    point_cloud = o3d.io.read_point_cloud(dtu_ply_path)
    mesh = trimesh.load_mesh(dtu_mesh_path)
    
    print(f"  点云: {len(point_cloud.points)} 顶点")
    print(f"  mesh: {len(mesh.vertices)} 顶点, {len(mesh.faces)} 面")
    
    cloud_output_path = os.path.join(scan_output_dir, f'{int(scan_id):03d}_pcd.ply')
    mesh_output_path = os.path.join(scan_output_dir, f'{int(scan_id):03d}_mesh.ply')

    o3d.io.write_point_cloud(cloud_output_path, point_cloud)
    mesh.export(mesh_output_path)
    print(f"  点云和mesh已保存到 {cloud_output_path} 和 {mesh_output_path}")

    # 2. 获取图像列表
    image_files = [f for f in os.listdir(dtu_rectified_dir) 
                   if f.endswith(f'_{light_condition}_r5000.png')]
    image_files.sort()
    
    if not image_files:
        print(f"  错误: 未找到光照条件 {light_condition} 的图像")
        return False
    
    print(f"  找到 {len(image_files)} 张图像")
    
    # 3. 处理图像和相机参数
    print(f"  处理图像和相机参数...")
    calib_data_path = os.path.join(dtu_calibration_dir, 'Calib_Results_right.mat')
    calib_data = scipy.io.loadmat(calib_data_path)
    K = np.zeros((3, 3))
    K[0, 0] = calib_data['fc'][0, 0]
    K[1, 1] = calib_data['fc'][1, 0]
    K[0, 2] = calib_data['cc'][0, 0]
    K[1, 2] = calib_data['cc'][1, 0]
    K[2, 2] = 1

    for idx, image_file in enumerate(tqdm(image_files, desc="  转换图像")):
        # 提取视图编号
        view_num = int(image_file.split('_')[1])
        
        # 复制并重命名图像
        src_image_path = os.path.join(dtu_rectified_dir, image_file)
        dst_image_path = os.path.join(scan_output_dir, 'images', f'{idx:08d}.jpg')
        
        # 转换PNG到JPG
        img = cv2.imread(src_image_path)
        cv2.imwrite(dst_image_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # 处理相机参数
        dtu_cam_path = os.path.join(dtu_calibration_dir, f'pos_{view_num:03d}.txt')
        mvsnet_cam_path = os.path.join(scan_output_dir, 'cams', f'{idx:08d}_cam.txt')
        
        if os.path.exists(dtu_cam_path):
            convert_dtu_camera_to_mvsnet(dtu_cam_path, mvsnet_cam_path, K)
        else:
            print(f"  警告: 未找到相机文件 {dtu_cam_path}")
    
    # 4. 渲染深度图
    print(f"  渲染深度图...")
    try:
        # 加载网格  
        mesh = trimesh.load_mesh(dtu_mesh_path)
        # 获取图像尺寸
        first_img = cv2.imread(os.path.join(scan_output_dir, 'images', '00000000.jpg'))
        H, W = first_img.shape[:2]
        
        # 渲染每个视角的深度图
        for idx in tqdm(range(len(image_files)), desc="  渲染深度图"):
            cam_path = os.path.join(scan_output_dir, 'cams', f'{idx:08d}_cam.txt')
            depth_path = os.path.join(scan_output_dir, 'depths', f'{idx:08d}.npy')
            mask_path = os.path.join(scan_output_dir, 'binary_masks', f'{idx:08d}.png')
            
            if os.path.exists(cam_path):
                # 读取相机参数
                intrinsic, extrinsic = load_cam_mvsnet(cam_path)

                camera_pose = np.linalg.inv(extrinsic)
                camera_pose[:, 1:3] *= -1.0  # OpenGL坐标系转换
                
                # 渲染深度图
                depth = render_depth_maps(mesh, [camera_pose], intrinsic[:3, :3], H, W)[0]
                depth, valid_mask = filter_depth_outliers(depth)
                print(f'depth: {depth.shape} mean: {depth.mean()} std: {depth.std()}')
                
                # 保存深度图
                np.save(depth_path, depth)
                
                # 生成掩膜
                mask = generate_binary_mask(depth)
                cv2.imwrite(mask_path, mask)
    
    except Exception as e:
        print(f"  错误: 深度图渲染失败 - {e}")
        return False
    
    # 5. 生成pair.txt
    generate_pair_file(scan_output_dir, len(image_files))
    
    print(f"  完成扫描 {scan_id}")
    return True

def main():
    parser = argparse.ArgumentParser(description='DTU数据集预处理为spann3r格式')
    parser.add_argument('--dtu_root', type=str, 
                       default='/home/vision/ws/datasets/SampleSet/dtu_sample',
                       help='DTU原始数据集根目录')
    parser.add_argument('--output_root', type=str,
                       default='/home/vision/ws/datasets/SampleSet/dtu_mvs',
                       help='输出根目录')
    parser.add_argument('--scan_ids', type=str, nargs='+',
                       default=['1', '6'],
                       help='要处理的扫描ID列表，或使用 "all" 处理所有可用扫描')
    parser.add_argument('--light_condition', type=int, default=3,
                       help='光照条件 (0-6)')
    parser.add_argument('--mesh_type', type=str, default='camp',
                       choices=['furu', 'camp', 'tola'],
                       help='Mesh类型: furu, camp, 或 tola (默认: camp)')
    parser.add_argument('--skip_existing', action='store_true', default=False,
                       help='跳过已存在的扫描')
    parser.add_argument('--force_reprocess', action='store_true',
                       help='强制重新处理所有扫描')
    
    args = parser.parse_args()
    
    # 处理参数
    if args.force_reprocess:
        args.skip_existing = False
    
    # 获取扫描列表
    if 'all' in args.scan_ids:
        available_scans = get_available_scans(args.dtu_root)
        scan_ids = [str(s) for s in available_scans]
        print(f"找到 {len(available_scans)} 个可用扫描")
    else:
        scan_ids = args.scan_ids
    
    print("DTU数据集预处理脚本")
    print(f"输入目录: {args.dtu_root}")
    print(f"输出目录: {args.output_root}")
    print(f"处理扫描: {scan_ids}")
    print(f"光照条件: {args.light_condition}")
    print(f"Mesh类型: {args.mesh_type}")
    print(f"跳过已存在: {args.skip_existing}")
    print("-" * 50)
    
    # 创建输出目录
    os.makedirs(args.output_root, exist_ok=True)
    
    # 处理每个扫描
    success_count = 0
    failed_scans = []
    
    for scan_id in tqdm(scan_ids, desc="处理扫描"):
        try:
            if process_single_scan(scan_id, args.dtu_root, args.output_root, 
                                 args.light_condition, args.skip_existing, args.mesh_type):
                success_count += 1
            else:
                failed_scans.append(scan_id)
        except Exception as e:
            print(f"处理扫描 {scan_id} 时出错: {e}")
            failed_scans.append(scan_id)
    
    print("-" * 50)
    print(f"处理完成: {success_count}/{len(scan_ids)} 个扫描成功")
    
    if failed_scans:
        print(f"失败的扫描: {failed_scans}")
    
    # 保存处理日志
    log_path = os.path.join(args.output_root, 'processing_log.json')
    log_data = {
        'timestamp': str(np.datetime64('now')),
        'input_dir': args.dtu_root,
        'output_dir': args.output_root,
        'light_condition': args.light_condition,
        'mesh_type': args.mesh_type,
        'total_scans': len(scan_ids),
        'successful_scans': success_count,
        'failed_scans': failed_scans,
        'processed_scans': [s for s in scan_ids if s not in failed_scans]
    }
    
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"处理日志已保存到: {log_path}")

if __name__ == '__main__':
    main()
