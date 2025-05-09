"""
HOH-Grasps Dataset Utilities
Author: Xinchao Song
"""

import os
import argparse
import json

import numpy as np
import open3d as o3d

num_points = 20000  # Number of points to sample from the point cloud


def get_data(hoh_grasps_root, frame, scene_id):
    """
    Load data from the HOH-Grasps dataset in the way for training the HI-Grasp network.

    Args:
        hoh_grasps_root (str): Root directory of the HOH-Grasps dataset.
        frame (str): Frame to load, either 'o' or 't'.
        scene_id (str): Scene ID to load.
    """

    scene_path = os.path.join(hoh_grasps_root, 'scenes', scene_id)

    # Load scene point cloud
    scene_pcd_path = os.path.join(scene_path, f'{scene_id}_scene_{frame}.ply')
    scene_pcd = o3d.io.read_point_cloud(scene_pcd_path)
    scene_pcd_points = np.asarray(scene_pcd.points)
    scene_pcd_colors = np.asarray(scene_pcd.colors)

    # Sample points
    if len(scene_pcd_points) >= num_points:
        idxs = np.random.choice(len(scene_pcd_points), num_points, replace=False)
    else:
        idxs1 = np.arange(len(scene_pcd_points))
        idxs2 = np.random.choice(len(scene_pcd_points), num_points - len(scene_pcd_points), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    pcd_points_sampled = scene_pcd_points[idxs]
    pcd_colors_sampled = scene_pcd_colors[idxs]

    # Load objectness labels
    objectness_label_path = os.path.join(scene_path, f'{scene_id}_objectness.npz')
    objectness_label = np.load(objectness_label_path)[frame]
    objectness_label = objectness_label[idxs]

    # Load object pose
    with open(os.path.join(scene_path, f'{scene_id}_transforms.json'), 'r') as f:
        all_transforms = json.load(f)
    object_pose = np.asarray(all_transforms[f'obj_to_{frame}'])

    # Load object grasp labels
    obj_id = scene_id[scene_id.rfind('_') + 1:]
    grasp_label_path = os.path.join(hoh_grasps_root, 'object_grasp_labels', 'graspnet', f'{obj_id}_labels.npz')
    grasp_label = np.load(grasp_label_path)
    grasp_points = grasp_label['points']
    grasp_offsets = grasp_label['offsets']
    fric_coefs = grasp_label['scores']
    grasp_tolerance = grasp_label['tolerance']
    scene_collision_path = os.path.join(scene_path, f'{scene_id}_scene_collision.npz')
    scene_collision = np.load(scene_collision_path)[frame]

    # Sample grasps
    idxs = np.random.choice(len(grasp_points), min(max(int(len(grasp_points) / 4), 300), len(grasp_points)),
                            replace=False)
    grasp_points = grasp_points[idxs]
    grasp_offsets = grasp_offsets[idxs]
    collision = scene_collision[idxs]
    fric_coefs = fric_coefs[idxs]
    fric_coefs[collision] = 0
    grasp_tolerance = grasp_tolerance[idxs]
    grasp_tolerance[collision] = 0

    ret_dict = {}
    ret_dict['point_clouds'] = pcd_points_sampled.astype(np.float32)
    ret_dict['cloud_colors'] = pcd_colors_sampled.astype(np.float32)
    ret_dict['objectness_label'] = objectness_label.astype(np.int64)
    ret_dict['object_poses_list'] = object_pose.astype(np.float32)
    ret_dict['grasp_points_list'] = grasp_points.astype(np.float32)
    ret_dict['grasp_offsets_list'] = grasp_offsets.astype(np.float32)
    ret_dict['grasp_labels_list'] = fric_coefs.astype(np.float32)
    ret_dict['grasp_tolerance_list'] = grasp_tolerance.astype(np.float32)

    return ret_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hoh_grasps_root', required=True, help='HOH-Grasps dataset root')
    parser.add_argument('--frame', default='o', help='Frame to load, either "o" or "t" [default: "o"]')
    args = parser.parse_args()

    data = get_data(args.hoh_grasps_root, args.frame, '01638-46157-S1_1_322')
    for key, value in data.items():
        print(f'{key}: {value.shape if isinstance(value, np.ndarray) else type(value)}')
