"""
HOH-Grasps Dataset Utilities
Author: Xinchao Song
"""

import os
import argparse
import json

import numpy as np
import open3d as o3d


def load_scene_pcds_centroids(hoh_grasps_root, frame, scene_id):
    """
    Load and demonstrate the scene point clouds and centroids from the HOH-Grasps dataset.

    Args:
        hoh_grasps_root (str): Root directory of the HOH-Grasps dataset.
        frame (str): Frame to load, either 'o' or 't'.
        scene_id (str): Scene ID to load.
    """

    scene_path = os.path.join(hoh_grasps_root, 'scenes', scene_id)

    with open(os.path.join(scene_path, f'{scene_id}_centroids.json'), 'r') as f:
        all_centroids = json.load(f)

    scene_pcd_path = os.path.join(scene_path, f'{scene_id}_scene_{frame}.ply')
    scene_pcd = o3d.io.read_point_cloud(scene_pcd_path)

    hand_giver_pcd_path = os.path.join(scene_path, f'{scene_id}_hand_giver_{frame}.ply')
    hand_giver_pcd = o3d.io.read_point_cloud(hand_giver_pcd_path)

    hand_receiver_pcd_path = os.path.join(scene_path, f'{scene_id}_hand_receiver_{frame}.ply')
    hand_receiver_pcd = o3d.io.read_point_cloud(hand_receiver_pcd_path)

    obj_centroid = np.asarray(all_centroids[f'obj_{frame}'])
    obj_centroid_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.0075)
    obj_centroid_mesh.translate(obj_centroid)
    obj_centroid_mesh.paint_uniform_color([0, 1, 0])

    hand_giver_centroid = np.asarray(all_centroids[f'hand_giver_{frame}'])
    hand_giver_centroid_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.0075)
    hand_giver_centroid_mesh.translate(hand_giver_centroid)
    hand_giver_centroid_mesh.paint_uniform_color([1, 0, 1])

    hand_receiver_centroid = np.asarray(all_centroids[f'hand_receiver_{frame}'])
    hand_receiver_centroid_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.0075)
    hand_receiver_centroid_mesh.translate(hand_receiver_centroid)
    hand_receiver_centroid_mesh.paint_uniform_color([0, 1, 1])

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries(
        [scene_pcd,
         hand_giver_pcd,
         hand_receiver_pcd,
         hand_giver_centroid_mesh,
         hand_receiver_centroid_mesh,
         obj_centroid_mesh,
         coordinate_frame]
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hoh_grasps_root', required=True, help='HOH-Grasps dataset root')
    parser.add_argument('--frame', default='o', help='Frame to load, either "o" or "t" [default: "o"]')
    args = parser.parse_args()

    load_scene_pcds_centroids(args.hoh_grasps_root, args.frame, '01638-46157-S1_5_318')
