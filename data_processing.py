"""
pcd_obs_env with:
1. object/background segmentation
2. object registration
3. goal sampling
4. reward calculation
"""

import numpy as np
from PIL import Image
import os
import argparse
from PIL import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import open3d as o3d
import numpy as np
from ctypes import * # convert float to uint32
# from matplotlib import pyplot as plt
import copy
import torch

# import sensor_msgs.point_cloud2 as pc2
from numpy.linalg import inv
# from lib_cloud_conversion_between_Open3D_and_ROS import convertCloudFromRosToOpen3d
from scipy.spatial.transform import Rotation
from utils import *

OPENESS_TH = 0.35 # Threshold to decide if a gripper opens

def save_mesh(pcd):
    print("0")
    pcd = pcd.farthest_point_down_sample(10000)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=10))
    print("1")
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
    print("2")
    bbox = pcd.get_axis_aligned_bounding_box()
    p_mesh_crop = poisson_mesh.crop(bbox)
    print("3")
    #o3d.io.write_triangle_mesh("p_mesh_c.obj", p_mesh_crop)

    #o3d.visualization.draw_geometries([p_mesh_crop])

    radii = [ 0.005, 0.01,0.02]
    #radii = [0.001, 0.002]
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    o3d.visualization.draw_geometries([pcd])
    o3d.visualization.draw_geometries([rec_mesh])

def process_episode(data, cam_extrinsic, o3d_intrinsic, original_image_size, resized_intrinsic_o3d, resized_image_size, bound_box):


    length = 1

    for idx in range( length ):    
        

        point = data[idx]
        bgr = point['bgr']
        rgb = bgr[...,::-1].copy()
        save_np_image(rgb, file_name = "original.jpg")

        depth = point['depth']
        np.save("original_depth", depth)

        # rgb, depth = transfer_camera_param(bgr, depth, o3d_intrinsic.intrinsic_matrix, original_image_size, resized_intrinsic_o3d.intrinsic_matrix, resized_image_size )
        # # print("rgb: ", type(rgb))
                
        # im_color = o3d.geometry.Image(rgb)
        # im_depth = o3d.geometry.Image(depth)
        # rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        #     im_color, im_depth, depth_scale=1000, depth_trunc=2000, convert_rgb_to_intensity=False)
        
        # all_valid_resized_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        #         rgbd,
        #         resized_intrinsic_o3d,
        # )
        # all_valid_resized_pcd.transform( cam_extrinsic )

        # # visualize_pcd(all_valid_resized_pcd)
        # xyz = xyz_from_depth(depth, resized_intrinsic_o3d.intrinsic_matrix, cam_extrinsic )

        # cropped_rgb, cropped_xyz = cropping( rgb, xyz, bound_box)
        # save_np_image(rgb)
        # np.save("test_depth", depth)
        # filtered_rgb, filtered_xyz = denoise(cropped_rgb, cropped_xyz, debug= True)
        # pcd_rgb = cropped_rgb.reshape(-1, 3) / 255.0
        # pcd_xyz = cropped_xyz.reshape(-1, 3)

        bound_box2 = np.array( [ [0.1, 0.75], [ -0.5 , 0.5], [ -0.1 , 0.1] ] )
        xyz = xyz_from_depth(depth, o3d_intrinsic.intrinsic_matrix, cam_extrinsic )
        cropped_rgb, cropped_xyz = cropping( rgb, xyz, bound_box2)
        pcd_rgb = cropped_rgb.reshape(-1, 3) / 255.0
        pcd_xyz = cropped_xyz.reshape(-1, 3)
        # pcd_xyz[:,2] = -0.05
        pcd = o3d.geometry.PointCloud()
        # pcd = o3d.t.geometry.PointCloud()        
        pcd.colors = o3d.utility.Vector3dVector( pcd_rgb )
        pcd.points = o3d.utility.Vector3dVector( pcd_xyz )
        #visualize_pcd( pcd )
        save_mesh( pcd )
        # pcd.project_to_depth_image(512, 512, resized_intrinsic_o3d, cam_extrinsic)

def main():
    
    parser = argparse.ArgumentParser(description="extract interested object and traj from rosbag")
    parser.add_argument("-d", "--data_index", default=1,  help="Input data index.")    
    parser.add_argument("-t", "--task", default="stack_blocks",  help="Input task name.")
    
    args = parser.parse_args()
    # bag_dir = "./segmented_" + args.task + "/" + str(args.data_index) + ".bag"
    # traj_dir = "./segmented_" + args.task + "/" + str(args.data_index) + ".npy"

    cam_extrinsic = get_transform( [-0.13913296, 0.053, 0.43643044 , -0.63127772, 0.64917582, -0.31329509, 0.28619116])
    offset = get_transform( [ 0., 0., 1.0 , 0., 0., 0., 1.])
    #print("look at: ", cam_extrinsic @ np.array( [0,0,1,0] ))
    print("look at: ", cam_extrinsic @ offset ) 
    #print("look at: ", offset @ cam_extrinsic )   
    print("ext:", cam_extrinsic)
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(1920, 1080, 734.1779174804688, 734.1779174804688, 993.6226806640625, 551.8895874023438)

    img_size = 512
    resized_img_size = (img_size,img_size)
    original_image_size = (1080, 1920) #(h,)
    # resized_intrinsic = o3d.camera.PinholeCameraIntrinsic( 256., 25, 80., 734.1779174804688*scale_y, 993.6226806640625*scale_x, 551.8895874023438*scale_y)
    fxfy = float(img_size)
    # fxfy = 128
    resized_intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(img_size, img_size, fxfy, fxfy, img_size / 2, img_size / 2)
    resized_intrinsic_np = np.array([
        [fxfy, 0., img_size/2],
        [0. ,fxfy, img_size/2],
        [0., 0., 1.0]
    ])

    bound_box = np.array( [ [0.05, 0.65], [ -0.5 , 0.5], [ -0.1 , 0.6] ] )
    task_name = args.task 
    print("task_name: ", task_name)

    processed_data_dir = "./processed"
    if ( os.path.isdir(processed_data_dir) == False ):
        os.mkdir(processed_data_dir)

    
    dir_path = './data/' + task_name + '/'

    save_data_dir = processed_data_dir + '/' + task_name
    if ( os.path.isdir(save_data_dir) == False ):
        os.mkdir(save_data_dir)
        
   
    file = str(args.data_index) + ".npy"
    print("processing: ", dir_path+file)
    data = np.load(dir_path+file, allow_pickle = True)

    process_episode(data, cam_extrinsic, o3d_intrinsic, original_image_size, resized_intrinsic_o3d, resized_img_size, bound_box)

if __name__ == "__main__":
    main()

    # [frame_ids],  # we use chunk and max_episode_length to index it
    # [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256) 
    #     obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
    # [action_tensors],  # wrt frame_ids, (2, 8)
    # [camera_dicts],
    # [gripper_tensors],  # wrt frame_ids, (2, 8) ,curretn state
    # [trajectories]  # wrt frame_ids, (N_i, 2, 8)
    # List of tensors
