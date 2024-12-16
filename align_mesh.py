import trimesh
import numpy as np
from PIL import Image
import os
import argparse
import cv2
import open3d as o3d
import copy
import torch
import plyfile

def mesh2points(mesh, num_points = 10000):
    """
    Convert a mesh to a list of points.
    """
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points

def align_meshes_pcl(mesh, pcl, max_iters=100000, threshold=1e-6):
    """
    Align a mesh to a point cloud.
    """
    # Sample points on the surface of the mesh
    mesh_points = mesh2points(mesh)

    mesh_center = mesh_points.mean(axis=0)
    pcl_center = pcl.mean(axis=0)

    mesh_scale = np.linalg.norm(mesh_points - mesh_center, axis=1).max()
    pcl_scale = np.linalg.norm(pcl - pcl_center, axis=0).max()
    rough_scale = pcl_scale / mesh_scale
    # rough_scale = 1.0
    scaled_mesh_points = (mesh_points - mesh_center) * rough_scale + pcl_center
    # Compute the transformation matrix
    T, transformed, cost = trimesh.registration.icp(scaled_mesh_points, 
                                                    pcl,
                                                    max_iterations=max_iters,
                                                    reflection=False,
                                                    scale=True,
                                                    threshold=threshold)
    # Apply the transformation to the mesh

    # mesh.apply_transform(T)
    mesh.vertices = (mesh.vertices - mesh_center) * rough_scale + pcl_center
    mesh.apply_transform(T)
    return mesh

def get_depth(depth_image, mask):
    depth_mask = np.where(mask > 0, 1, 0)
    depth_image = depth_image * depth_mask
    return depth_image

def depth_to_pcl(depth_image, intrinsics):
    """
    Convert a depth image to a point cloud.
    """
    h, w = depth_image.shape
    u0, v0 = intrinsics[0, 2], intrinsics[1, 2]
    fu, fv = intrinsics[0, 0], intrinsics[1, 1]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u_flat = u.flatten() - u0
    v_flat = v.flatten() - v0
    z_flat = depth_image.flatten()
    u_flat = u_flat[z_flat > 0]
    v_flat = v_flat[z_flat > 0]
    z_flat = z_flat[z_flat > 0]
    unporj_u = u_flat * z_flat / fu
    unporj_v = v_flat * z_flat / fv
    return np.vstack((unporj_u, unporj_v, z_flat)).T 

def write_ply(ply_path, points, colors= np.array([])):
        """Write mesh, point cloud, or oriented point cloud to ply file.
        Args:
            ply_path (str): Output ply path.
            points (float): Nx3 x,y,z locations for each point
            colors (uchar): Nx3 r,g,b color for each point
        We provide this function for you.
        """
        with open(ply_path, 'w') as f:
            # Write header.
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write('element vertex {}\n'.format(len(points)))
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            if len(colors) != 0:
                f.write('property uchar red\n')
                f.write('property uchar green\n')
                f.write('property uchar blue\n')
            f.write('end_header\n')
            # Write points.
            for i in range(len(points)):
                f.write('{0} {1} {2}'.format(
                    points[i][0],
                    points[i][1],
                    points[i][2]))
                if len(colors) != 0:
                    f.write(' {0} {1} {2}'.format(
                        int(colors[i][0]),
                        int(colors[i][1]),
                        int(colors[i][2])))

def main():
    parser = argparse.ArgumentParser(description="Align a mesh to a point cloud.")
    parser.add_argument("-m", "--mesh_file", default="",  help="Provide mesh file.")    
    parser.add_argument("--depth_image", default="",  help="Provide depth image.")
    parser.add_argument("--mask", default="",  help="Provide mask file.")
    
    args = parser.parse_args()
    mesh_file = args.mesh_file
    depth_image = args.depth_image
    mask_file = args.mask

    mesh = trimesh.load_mesh(mesh_file)
    # read depth image
    # depth_image = Image.open(depth_image)
    depth_image = np.load(depth_image)
    # read mask
    mask = Image.open(mask_file)
    mask = np.array(mask)

    source_depth = depth_image
    depth = get_depth(depth_image, mask)
    # depth = depth_image

    img_size = 512
    fxfy = float(img_size)

    intrinsic_np = np.array([
        [fxfy, 0., img_size/2],
        [0. ,fxfy, img_size/2],
        [0., 0., 1.0]
    ])
    # get point cloud
    pcl = depth_to_pcl(depth, intrinsic_np)
    source_pcl = depth_to_pcl(source_depth, intrinsic_np)
    # write_ply("pcl.ply", pcl)
    # write_ply("source_pcl.ply", source_pcl)
    
    aligned_mesh = align_meshes_pcl(mesh, pcl)
    # aligned_mesh.export("aligned_mesh.obj")
    mesh_points = mesh2points(aligned_mesh)
    write_ply("aligned_mesh.ply", mesh_points)





if __name__ == "__main__":
    main()