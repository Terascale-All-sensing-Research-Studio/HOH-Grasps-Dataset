"""
HOH-Grasps Dataset Utilities
Author: Xinchao Song
"""

import os
import argparse

import numpy as np
import open3d as o3d

from graspnetAPI.utils.utils import plot_gripper_pro_max


def process_grasps_plain_6d(translations, rotation_matrices, depths, widths, scores, quality_thresh=0.8, num_grasps=50):
    """
    Process grasps into a GraspNet GraspGroup object for visualization.

    Args:
        translations (np.ndarray): Grasp translations.
        rotation_matrices (np.ndarray): Grasp rotation_matrices.
        depths (np.ndarray): Grasp depths.
        widths (np.ndarray): Grasp widths.
        scores (np.ndarray): Grasp scores.
        quality_thresh (float): Quality threshold for grasps (higher scores for higher quality).
        num_grasps (int): Number of grasps to visualize.

    Returns:
        GraspGroup: Processed grasp group containing the filtered grasps.
    """

    mask = (scores >= quality_thresh)
    translations = translations[mask]
    rotation_matrices = rotation_matrices[mask]
    widths = widths[mask]
    depths = depths[mask]
    scores = scores[mask]

    grippers = []
    selected_indices = np.random.choice(translations.shape[0], size=num_grasps, replace=False)
    for selected_index in selected_indices:
        gripper = plot_gripper_pro_max(translations[selected_index],
                                       rotation_matrices[selected_index],
                                       widths[selected_index],
                                       depths[selected_index],
                                       scores[selected_index],
                                       color=[0, 1, 0])
        grippers.append(gripper)

    return grippers


def load_grasps_plain_6d_good(hoh_grasps_root, frame, scene_id, quality_thresh, num_grasps):
    """
    Load and demonstrate plain 6D style good grasps for a given scene.

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

    grasp_label_path = os.path.join(scene_path, f'{scene_id}_grasps_good_{frame}.npz')
    grasp_label = np.load(grasp_label_path)
    translations = grasp_label['translations']
    rotation_matrices = grasp_label['rotation_matrices']
    widths = grasp_label['widths']
    depths = grasp_label['depths']
    scores = grasp_label['scores']

    grippers = process_grasps_plain_6d(translations, rotation_matrices, depths, widths, scores,
                                       quality_thresh, num_grasps)

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
    parser.add_argument('--quality_thresh', type=float, default=0.8,
                        help='Quality threshold for grasps (higher scores for higher quality) [default: 0.8]')
    args = parser.parse_args()

    load_grasps_plain_6d_good(args.hoh_grasps_root,
                              args.frame,
                              '01638-46157-S1_5_318',
                              args.quality_thresh,
                              args.num_grasps)
