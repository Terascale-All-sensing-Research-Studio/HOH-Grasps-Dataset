"""
HOH-Grasps Dataset Utilities
Author: Xinchao Song
"""

import os
import argparse
import json

import numpy as np
import open3d as o3d

from graspnetAPI.utils.utils import generate_views, transform_points
from graspnetAPI.utils.rotation import batch_viewpoint_params_to_matrix
from graspnetAPI.grasp import GraspGroup


def process_grasps_graspnet(obj_id, points, offsets, fric_coefs, collision, transform,
                            quality_thresh=0.4, num_grasps=50):
    """
    Process grasps into a GraspNet GraspGroup object for visualization.
    Modified from the GraspNet API: https://github.com/graspnet/graspnetAPI/blob/master/graspnetAPI/graspnet.py#L567.

    Args:
        obj_id (str): Object ID.
        points (np.ndarray): Grasp points.
        offsets (np.ndarray): Grasp offsets.
        fric_coefs (np.ndarray): Friction coefficients for the grasps.
        collision (np.ndarray): Collision information for the grasps.
        transform (np.ndarray): Transformation matrix to apply to the grasp points.
        quality_thresh (float): Quality threshold for grasps (higher scores for higher quality).
        num_grasps (int): Number of grasps to visualize.

    Returns:
        GraspGroup: Processed grasp group containing the filtered grasps.
    """

    num_views, num_angles, num_depths = 300, 12, 4
    point_inds = np.arange(points.shape[0])
    num_points = len(point_inds)
    target_points = points[:, np.newaxis, np.newaxis, np.newaxis, :]
    target_points = np.tile(target_points, [1, num_views, num_angles, num_depths, 1])

    template_views = generate_views(num_views)
    template_views = template_views[np.newaxis, :, np.newaxis, np.newaxis, :]
    template_views = np.tile(template_views, [1, 1, num_angles, num_depths, 1])
    views = np.tile(template_views, [num_points, 1, 1, 1, 1])

    angles = offsets[:, :, :, :, 0]
    depths = offsets[:, :, :, :, 1]
    widths = offsets[:, :, :, :, 2]

    mask = ((fric_coefs <= quality_thresh) & (fric_coefs > 0) & ~collision)
    target_points = target_points[mask]
    target_points = transform_points(target_points, transform)
    views = views[mask]
    angles = angles[mask]
    depths = depths[mask]
    widths = widths[mask]
    fric_coefs = fric_coefs[mask]

    Rs = batch_viewpoint_params_to_matrix(-views, angles)
    Rs = np.matmul(transform[np.newaxis, :3, :3], Rs)

    num_grasp = widths.shape[0]
    scores = (1.1 - fric_coefs).reshape(-1, 1)
    widths = widths.reshape(-1, 1)
    heights = 0.02 * np.ones((num_grasp, 1))
    depths = depths.reshape(-1, 1)
    rotations = Rs.reshape((-1, 9))

    if 'F001' in obj_id:
        obj_id_encoded = 6001
    elif 'T001' in obj_id:
        obj_id_encoded = 20001
    else:
        obj_id_encoded = int(obj_id[-3:])
    obj_ids = np.full((num_grasp, 1), int(obj_id_encoded), dtype=np.int32)

    grasp_group_array = np.hstack([scores, widths, heights, depths, rotations, target_points, obj_ids]).astype(
        np.float32)
    random_indices = np.random.choice(grasp_group_array.shape[0], size=num_grasps, replace=False)
    grasp_group = GraspGroup()
    grasp_group.grasp_group_array = grasp_group_array[random_indices]

    return grasp_group


def load_grasps_graspnet(hoh_grasps_root, frame, scene_id, quality_thresh, num_grasps):
    """
    Load and demonstrate GraspNet style grasps for a given scene.

    Args:
        hoh_grasps_root (str): Root directory of the HOH-Grasps dataset.
        frame (str): Frame to load, either 'o' or 't'.
        scene_id (str): Scene ID to load.
        quality_thresh (float): Quality threshold for grasps (higher scores for higher quality).
        num_grasps (int): Number of grasps to visualize.
    """

    scene_path = os.path.join(hoh_grasps_root, 'scenes', scene_id)

    scene_pcd_path = os.path.join(scene_path, f'{scene_id}_scene_{frame}.ply')
    scene_pcd = o3d.io.read_point_cloud(scene_pcd_path)

    with open(os.path.join(scene_path, f'{scene_id}_transforms.json'), 'r') as f:
        all_transforms = json.load(f)
    obj_mesh_to_scene_transform = np.asarray(all_transforms[f'obj_to_{frame}'])

    obj_id = scene_id[scene_id.rfind('_') + 1:]
    grasp_label_path = os.path.join(hoh_grasps_root, 'object_grasp_labels', 'graspnet', f'{obj_id}_labels.npz')
    grasp_label = np.load(grasp_label_path)
    points = grasp_label['points']
    offsets = grasp_label['offsets']
    fric_coefs = grasp_label['scores']

    scene_collision_path = os.path.join(scene_path, f'{scene_id}_scene_collision.npz')
    scene_collision = np.load(scene_collision_path)[frame]

    grasp_group = process_grasps_graspnet(
        obj_id, points, offsets, fric_coefs, scene_collision, obj_mesh_to_scene_transform, quality_thresh, num_grasps
    )
    grippers = grasp_group.to_open3d_geometry_list()

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([
        scene_pcd,
        *grippers,
        coordinate_frame
    ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hoh_grasps_root', required=True, help='HOH-Grasps dataset root')
    parser.add_argument('--frame', default='o', help='Frame to load, either "o" or "t" [default: "o"]')
    parser.add_argument('--num_grasps', type=int, default=50,
                        help='Number of grasps to visualize [default: 50]')
    parser.add_argument('--quality_thresh', type=float, default=0.4,
                        help='Quality threshold for grasps (higher scores for higher quality) [default: 0.4]')
    args = parser.parse_args()

    load_grasps_graspnet(args.hoh_grasps_root,
                         args.frame,
                         '01638-46157-S1_5_318',
                         args.quality_thresh,
                         args.num_grasps)
