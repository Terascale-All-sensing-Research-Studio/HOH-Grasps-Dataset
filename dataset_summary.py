"""
HOH-Grasps Dataset Utilities
Author: Xinchao Song
"""

import os
import argparse
import json

grasp_label_sets = ['graspnet', 'plain_6d_valid', 'plain_6d_good']


def grasp_labels_summary(hoh_grasps_root):
    """
    Summarize the object grasp labels in the HOH-Grasps dataset.

    Args:
        hoh_grasps_root (str): Root directory of the HOH-Grasps dataset.
    """

    num_files = 0

    for grasp_label_set in grasp_label_sets:
        grasp_label_dir = os.path.join(hoh_grasps_root, 'object_grasp_labels', grasp_label_set)
        num_files += len(os.listdir(grasp_label_dir))

    assert num_files == 408

    print('HOH-Grasps Dataset Object Grasp Labels Summary:')
    print(f'Total number of label files: {num_files}')
    print()


def scenes_summary(hoh_grasps_root):
    """
    Summarize the scenes in the HOH-Grasps dataset.

    Args:
        hoh_grasps_root (str): Root directory of the HOH-Grasps dataset.
    """

    num_plys = 0
    num_npzs = 0
    num_jsons = 0

    scene_ids = json.load(open(os.path.join(hoh_grasps_root, 'scene_ids.json'), 'r'))
    all_scene_ids = scene_ids['all']
    num_scenes = len(all_scene_ids)

    for scene_id in all_scene_ids:
        scene_path = os.path.join(hoh_grasps_root, 'scenes', scene_id)
        files = os.listdir(scene_path)
        for file in files:
            if file.endswith('.ply'):
                num_plys += 1
            elif file.endswith('.npz'):
                num_npzs += 1
            elif file.endswith('.json'):
                num_jsons += 1

    num_files = num_plys + num_npzs + num_jsons

    assert num_scenes == 2720
    assert num_plys == 24480
    assert num_npzs == 16320
    assert num_jsons == 5440
    assert num_files == 46240

    print('HOH-Grasps Dataset Scenes Summary:')
    print(f'Total number of scenes: {num_scenes}')
    print(f'Total number of .ply files: {num_plys}')
    print(f'Total number of .npz files: {num_npzs}')
    print(f'Total number of .json files: {num_jsons}')
    print(f'Total number of files: {num_plys + num_npzs + num_jsons}')
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hoh_grasps_root', required=True, help='HOH-Grasps dataset root')
    args = parser.parse_args()

    grasp_labels_summary(args.hoh_grasps_root)
    scenes_summary(args.hoh_grasps_root)
