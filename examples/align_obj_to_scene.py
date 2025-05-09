"""
HOH-Grasps Dataset Utilities
Author: Xinchao Song
"""

import os
import json
import argparse

import numpy as np
import open3d as o3d


def align_obj_to_scene(hoh_grasps_root, hoh_root, frame, scene_id):
    """
    Align the object mesh to the scene point cloud using the transformation from the HOH dataset.

    Args:
        hoh_grasps_root (str): Root directory of the HOH-Grasps dataset.
        hoh_root (str): Root directory of the HOH dataset.
        frame (str): Frame to load, either 'o' or 't'.
        scene_id (str): Scene ID to load.
    """

    scene_path = os.path.join(hoh_grasps_root, 'scenes', scene_id)

    with open(os.path.join(scene_path, f'{scene_id}_transforms.json'), 'r') as f:
        all_transforms = json.load(f)

    scene_pcd_path = os.path.join(scene_path, f'{scene_id}_scene_{frame}.ply')
    scene_pcd = o3d.io.read_point_cloud(scene_pcd_path)

    obj_id = scene_id[scene_id.rfind('_') + 1:]
    obj_path = os.path.join(hoh_root, 'HOH_Objects', 'simplified', f'{obj_id}_simplified.obj')
    obj_mesh = o3d.io.read_triangle_mesh(obj_path)
    obj_mesh.scale(0.001, center=(0, 0, 0))
    obj_mesh.transform(np.asarray(all_transforms[f'obj_to_{frame}']))

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries(
        [scene_pcd,
         obj_mesh,
         coordinate_frame]
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hoh_grasps_root', required=True, help='HOH-Grasps dataset root')
    parser.add_argument('--hoh_root', required=True, help='HOH dataset root')
    parser.add_argument('--frame', default='o', help='Frame to load, either "o" or "t" [default: "o"]')
    args = parser.parse_args()

    align_obj_to_scene(args.hoh_grasps_root, args.hoh_root, args.frame, '01638-46157-S1_5_318')
