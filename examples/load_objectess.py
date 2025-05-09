"""
HOH-Grasps Dataset Utilities
Author: Xinchao Song
"""

import os
import argparse

import numpy as np
import open3d as o3d


def load_objectess(hoh_grasps_root, frame, scene_id):
    """
    Load and demonstrate the objectess labels from the HOH-Grasps dataset.

    Args:
        hoh_grasps_root (str): Root directory of the HOH-Grasps dataset.
        frame (str): Frame to load, either 'o' or 't'.
        scene_id (str): Scene ID to load.
    """

    scene_path = os.path.join(hoh_grasps_root, 'scenes', scene_id)

    scene_pcd_path = os.path.join(scene_path, f'{scene_id}_scene_{frame}.ply')
    scene_pcd = o3d.io.read_point_cloud(scene_pcd_path)

    objectness_label_path = os.path.join(scene_path, f'{scene_id}_objectness.npz')
    objectness_label = np.load(objectness_label_path)[frame]

    scene_pcd_colors = np.asarray(scene_pcd.colors)
    scene_pcd_colors[objectness_label == 0] = [0, 0, 1]
    scene_pcd_colors[objectness_label == 1] = [1, 0, 0]
    scene_pcd.colors = o3d.utility.Vector3dVector(scene_pcd_colors)

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries(
        [
            scene_pcd,
            coordinate_frame
        ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hoh_grasps_root', required=True, help='HOH-Grasps dataset root')
    parser.add_argument('--frame', default='o', help='Frame to load, either "o" or "t" [default: "o"]')
    args = parser.parse_args()

    load_objectess(args.hoh_grasps_root, args.frame, '01638-46157-S1_5_318')
