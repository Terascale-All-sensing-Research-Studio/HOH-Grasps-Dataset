"""
HOH-Grasps Dataset Generation
Author: Xinchao Song
"""

import os
import json
import argparse
from multiprocessing import Pool

import numpy as np
import open3d as o3d
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# Transformation from HOH to upright coordinate system
hoh2upright_transformation = [
    [0.64019561, 0.32223993, -0.69736003, (-7.59690508 + 829)],
    [0.76784799, -0.29635357, 0.56796481, (4.80058189 - 845)],
    [-0.0236442, -0.89907507, -0.43715554, (-9.5397194 + 775)],
    [0.0, 0.0, 0.0, 1.0]
]

# Reorientation matrix
rotation180 = R.from_euler('x', 180, degrees=True)
lift = np.array([0, 0, -500])
rotated_lift = rotation180.as_matrix() @ lift
reorient_matrix = np.eye(4)
reorient_matrix[:3, :3] = rotation180.as_matrix()
reorient_matrix[:3, 3] = rotated_lift
reorient_matrix = reorient_matrix @ hoh2upright_transformation

# Rotation for 180 degrees around the Z-axis
r_z180 = R.from_euler('z', 180, degrees=True).as_matrix()
t_z180 = np.eye(4)
t_z180[:3, :3] = r_z180

# Tabletop configuration
tabletop_size = 0.2
x_range = np.linspace(-tabletop_size, tabletop_size, 100)
y_range = np.linspace(-tabletop_size, tabletop_size, 100)
xx, yy = np.meshgrid(x_range, y_range)
xx_flat = xx.flatten()
yy_flat = yy.flatten()


def process_left_giver_pcd(pcd, left_giver):
    """ Rotate the point cloud for left seated givers."""
    if left_giver:
        pcd.transform(t_z180)
    return pcd


def process_left_giver_centroid(point, left_giver):
    """ Rotate the centroid for left seated givers."""
    if left_giver:
        point = r_z180 @ point
    return point


def process_left_giver_absolute_transform(transform, left_giver):
    """ Rotate the absolute transformation for left seated givers."""
    if left_giver:
        transform = t_z180 @ transform
    return transform


def process_left_giver_relative_transform(transform, left_giver):
    """ Rotate the relative transformation for left seated givers."""
    if left_giver:
        transform = t_z180 @ transform @ np.linalg.inv(t_z180)
    return transform


def scale_reorient_pcd(pcd, scale_factor=0.001):
    """ Scale and reorient the point cloud."""
    pcd.transform(hoh2upright_transformation)
    pcd_points = np.asarray(pcd.points)
    pcd_colors = np.asarray(pcd.colors)
    pcd_points_transformed = pcd_points * scale_factor
    pcd_points_transformed += np.array([0, 0, -0.500])
    pcd_points_transformed = rotation180.apply(pcd_points_transformed)
    pcd.points = o3d.utility.Vector3dVector(pcd_points_transformed)
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
    return pcd


def compute_pcd_centroid(pcd):
    """ Compute the centroid of a point cloud."""
    pcd_points = np.asarray(pcd.points)
    centroid = np.mean(pcd_points, axis=0)
    return centroid


def compute_mesh_centroid(obj_id):
    """ Compute the centroid of a mesh object."""
    obj_mesh_path = os.path.join(args.hoh_objects, 'full', f'{obj_id}_cleaned.obj')
    obj_mesh = o3d.io.read_triangle_mesh(obj_mesh_path)
    obj_mesh.scale(0.001, center=(0, 0, 0))
    vertices = np.asarray(obj_mesh.vertices)
    centroid = vertices.mean(axis=0)

    return centroid


def generate_scene(scene_id):
    """ Generate a scene from the HOH dataset."""
    all_transforms = {}
    all_centroids = {}

    # Create the scene directory
    scene_dir = os.path.join(scene_root, scene_id)
    if not os.path.exists(scene_dir):
        os.makedirs(scene_dir)

    # HOH metadata
    capture_dir = scene_id[:scene_id.find('_')]
    handover_idx = scene_id[scene_id.find('_') + 1:scene_id.rfind('_')]
    obj_id = scene_id[scene_id.rfind('_') + 1:]
    metadata_path = os.path.join(args.hoh_minimum, capture_dir, 'reference',
                                 f'{capture_dir}.json')
    with open(metadata_path, 'r') as json_file:
        data = json.load(json_file)
    left_giver = data['left_seated_giver']
    keyframes = data['keyframes']
    keyframes_idx = keyframes[int(handover_idx)]
    T_frame_value = keyframes_idx['t_frame_index']
    G_frame_value = keyframes_idx['g_frame_index']
    O_frame_value = keyframes_idx['o_frame_index']

    # HOH transformations
    frame_transforms_path = os.path.join(args.hoh_minimum, capture_dir, '3dModelAlignments',
                                         f'{capture_dir}_{handover_idx}_transformations.json')
    frame_transforms = json.load(open(frame_transforms_path, 'r'))
    O_to_G = np.array(frame_transforms[f'{O_frame_value}_{G_frame_value}'])
    G_to_O = np.linalg.inv(O_to_G)
    G_to_T = np.eye(4)
    for i in range(G_frame_value, T_frame_value):
        G_to_T = np.array(frame_transforms[f'{i}_{i + 1}']) @ G_to_T
    O_to_T = G_to_T @ O_to_G
    T_to_O = np.linalg.inv(O_to_T)

    # Get the O-to-T transformation
    O_to_T_reoriented = reorient_matrix @ O_to_T @ np.linalg.inv(reorient_matrix)
    O_to_T_reoriented[:3, 3] *= 0.001
    O_to_T_reoriented = process_left_giver_relative_transform(O_to_T_reoriented, left_giver)
    all_transforms['o_to_t'] = O_to_T_reoriented.tolist()

    # Get the T-to-O transformation
    T_to_O_reoriented = reorient_matrix @ T_to_O @ np.linalg.inv(reorient_matrix)
    T_to_O_reoriented[:3, 3] *= 0.001
    T_to_O_reoriented = process_left_giver_relative_transform(T_to_O_reoriented, left_giver)
    all_transforms['t_to_o'] = T_to_O_reoriented.tolist()

    # Get the object-to-scene transform at O frame
    mesh_to_O = np.array(frame_transforms['3dm_to_O'])
    obj_to_o = reorient_matrix @ mesh_to_O
    obj_to_o[:3, 3] *= 0.001
    obj_to_o = process_left_giver_absolute_transform(obj_to_o, left_giver)
    all_transforms['obj_to_o'] = obj_to_o.tolist()

    # Get the object-to-scene transform at T frame
    obj_to_t = O_to_T_reoriented @ obj_to_o
    all_transforms['obj_to_t'] = obj_to_t.tolist()

    # Save all transforms
    transforms_path = os.path.join(scene_dir, f'{scene_id}_transforms.json')
    with open(transforms_path, 'w') as f:
        json.dump(all_transforms, f, indent=2)

    # Get the object point cloud at O frame
    obj_o_pcd_path = os.path.join(args.hoh_pre_handover_objects, 'cleaned_point_clouds',
                                  f'{capture_dir}_{handover_idx}_{O_frame_value}_.ply')
    obj_o_pcd = o3d.io.read_point_cloud(obj_o_pcd_path)
    obj_o_pcd = scale_reorient_pcd(obj_o_pcd)
    obj_o_pcd = process_left_giver_pcd(obj_o_pcd, left_giver)
    o3d.io.write_point_cloud(os.path.join(scene_dir, f'{scene_id}_obj_o.ply'), obj_o_pcd)

    # Get the object point cloud at T frame
    obj_t_pcd_path = os.path.join(args.hoh_minimum, capture_dir, 'PCFiltered', handover_idx,
                                  f'object_frame{T_frame_value}.ply')
    obj_t_pcd = o3d.io.read_point_cloud(obj_t_pcd_path)
    obj_t_pcd = scale_reorient_pcd(obj_t_pcd)
    obj_t_pcd = process_left_giver_pcd(obj_t_pcd, left_giver)
    o3d.io.write_point_cloud(os.path.join(scene_dir, f'{scene_id}_obj_t.ply'), obj_t_pcd)

    # Get the tabletop
    obj_o_points = np.asarray(obj_o_pcd.points)
    tabletop_z = np.max(obj_o_points[:, 2])
    zz_flat = np.full(xx_flat.shape, tabletop_z)
    tabletop_points = np.vstack((xx_flat, yy_flat, zz_flat)).T
    tabletop_colors = np.full(tabletop_points.shape, [1, 0, 0]).astype(np.float64)
    tabletop_pcd = o3d.geometry.PointCloud()
    tabletop_pcd.points = o3d.utility.Vector3dVector(tabletop_points)
    tabletop_pcd.colors = o3d.utility.Vector3dVector(tabletop_colors)
    tabletop_pcd = process_left_giver_pcd(tabletop_pcd, left_giver)
    o3d.io.write_point_cloud(os.path.join(scene_dir, f'{scene_id}_tabletop.ply'), tabletop_pcd)

    # Get the scene point cloud at O frame
    obj_o_colors = np.asarray(obj_o_pcd.colors)
    scene_points = np.vstack((obj_o_points, tabletop_points))
    scene_colors = np.vstack((obj_o_colors, tabletop_colors))
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(scene_points)
    scene_pcd.colors = o3d.utility.Vector3dVector(scene_colors)
    o3d.io.write_point_cloud(os.path.join(scene_dir, f'{scene_id}_scene_o.ply'), scene_pcd)

    # Get the scene point cloud at T frame
    obj_t_points = np.asarray(obj_t_pcd.points)
    obj_t_colors = np.asarray(obj_t_pcd.colors)
    scene_points = np.vstack((obj_t_points, tabletop_points))
    scene_colors = np.vstack((obj_t_colors, tabletop_colors))
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(scene_points)
    scene_pcd.colors = o3d.utility.Vector3dVector(scene_colors)
    o3d.io.write_point_cloud(os.path.join(scene_dir, f'{scene_id}_scene_t.ply'), scene_pcd)

    # Get the giver's hand point cloud at O frame
    hand_giver_g_frame_pcd_path = os.path.join(args.hoh_minimum, capture_dir, 'PCFiltered', handover_idx,
                                               f'giver_frame{G_frame_value}.ply')
    hand_giver_g_frame_pcd = o3d.io.read_point_cloud(hand_giver_g_frame_pcd_path)
    hand_giver_o_pcd = hand_giver_g_frame_pcd.transform(G_to_O)
    hand_giver_o_pcd = scale_reorient_pcd(hand_giver_o_pcd)
    hand_giver_o_pcd = process_left_giver_pcd(hand_giver_o_pcd, left_giver)
    o3d.io.write_point_cloud(os.path.join(scene_dir, f'{scene_id}_hand_giver_o.ply'), hand_giver_o_pcd)

    # Get the giver's hand point cloud at T frame
    hand_giver_t_pcd_path = os.path.join(args.hoh_minimum, capture_dir, 'PCFiltered', handover_idx,
                                         f'giver_frame{T_frame_value}.ply')
    hand_giver_t_pcd = o3d.io.read_point_cloud(hand_giver_t_pcd_path)
    hand_giver_t_pcd = scale_reorient_pcd(hand_giver_t_pcd)
    hand_giver_t_pcd = process_left_giver_pcd(hand_giver_t_pcd, left_giver)
    o3d.io.write_point_cloud(os.path.join(scene_dir, f'{scene_id}_hand_giver_t.ply'), hand_giver_t_pcd)

    # Get the receiver's hand point cloud at O frame
    hand_receiver_t_pcd_path = os.path.join(args.hoh_minimum, capture_dir, 'PCFiltered', handover_idx,
                                            f'receiver_frame{T_frame_value}.ply')
    hand_receiver_t_pcd = o3d.io.read_point_cloud(hand_receiver_t_pcd_path)
    hand_receiver_o_pcd = hand_receiver_t_pcd.transform(T_to_O)
    hand_receiver_o_pcd = scale_reorient_pcd(hand_receiver_o_pcd)
    hand_receiver_o_pcd = process_left_giver_pcd(hand_receiver_o_pcd, left_giver)
    o3d.io.write_point_cloud(os.path.join(scene_dir, f'{scene_id}_hand_receiver_o.ply'),
                             hand_receiver_o_pcd)

    # Get the receiver's hand point cloud at T frame
    hand_receiver_t_pcd = o3d.io.read_point_cloud(hand_receiver_t_pcd_path)
    hand_receiver_t_pcd = scale_reorient_pcd(hand_receiver_t_pcd)
    hand_receiver_t_pcd = process_left_giver_pcd(hand_receiver_t_pcd, left_giver)
    o3d.io.write_point_cloud(os.path.join(scene_dir, f'{scene_id}_hand_receiver_t.ply'),
                             hand_receiver_t_pcd)

    # Get the object centroid at O frame
    obj_centroid = np.array(all_obj_centroids[obj_id])
    obj_centroid = obj_centroid[np.newaxis, :]
    obj_centroid_homogeneous = np.hstack((obj_centroid, np.ones((obj_centroid.shape[0], 1))))
    transformed_obj_centroid_o_homogeneous = np.dot(obj_centroid_homogeneous, obj_to_o.T)
    transformed_obj_centroid_o = transformed_obj_centroid_o_homogeneous[:, :3].squeeze()
    all_centroids['obj_o'] = transformed_obj_centroid_o.tolist()

    # Get the object centroid at T frame
    transformed_obj_centroid_t_homogeneous = np.dot(obj_centroid_homogeneous, obj_to_t.T)
    transformed_obj_centroid_t = transformed_obj_centroid_t_homogeneous[:, :3].squeeze()
    all_centroids['obj_t'] = transformed_obj_centroid_t.tolist()

    # Get the hand giver's centroid at O frame
    hand_giver_centroid_o = compute_pcd_centroid(hand_giver_o_pcd)
    all_centroids['hand_giver_o'] = hand_giver_centroid_o.tolist()

    # Get the hand giver's centroid at T frame
    hand_giver_centroid_t = compute_pcd_centroid(hand_giver_t_pcd)
    all_centroids['hand_giver_t'] = hand_giver_centroid_t.tolist()

    # Get the hand receiver's centroid at O frame
    hand_receiver_centroid_o = compute_pcd_centroid(hand_receiver_o_pcd)
    all_centroids['hand_receiver_o'] = hand_receiver_centroid_o.tolist()

    # Get the hand receiver's centroid at T frame
    hand_receiver_centroid_t = compute_pcd_centroid(hand_receiver_t_pcd)
    all_centroids['hand_receiver_t'] = hand_receiver_centroid_t.tolist()

    # Save all centroids
    centroids_path = os.path.join(scene_dir, f'{scene_id}_centroids.json')
    with open(centroids_path, 'w') as f:
        json.dump(all_centroids, f, indent=2)

    # Get the objectness at O frame
    object_o_mask = np.ones(len(obj_o_points))
    tabletop_mask = np.zeros(len(tabletop_points))
    objectness_o = np.concatenate([object_o_mask, tabletop_mask])

    # Get the objectness at T frame
    object_t_mask = np.ones(len(obj_t_points))
    objectness_t = np.concatenate([object_t_mask, tabletop_mask])

    # Save the objectness
    np.savez_compressed(os.path.join(scene_dir, f'{scene_id}_objectness.npz'),
                        o=objectness_o,
                        t=objectness_t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hoh_minimum', required=True, help='HOH_Minimum_Dataset root')
    parser.add_argument('--hoh_pre_handover_objects', required=True, help='HOH pre_handover_object root')
    parser.add_argument('--hoh_objects', required=True, help='HOH_Objects root')
    parser.add_argument('--output_path', required=True, help='Outputted path for HOH-Grasps dataset')
    args = parser.parse_args()

    # Load object IDs and compute their centroids
    obj_ids = json.load(open('object_ids.json', 'r'))
    print(f'Processing {len(obj_ids)} objects...')
    all_obj_centroids = {}
    for obj_id in tqdm(obj_ids):
        obj_centroid = compute_mesh_centroid(obj_id)
        all_obj_centroids[obj_id] = obj_centroid

    # Process all scenes
    scene_ids = json.load(open('scene_ids.json', 'r'))['all']
    print(f'Processing {len(scene_ids)} scenes...')

    scene_root = os.path.join(args.output_path, 'scenes')
    if not os.path.exists(scene_root):
        os.makedirs(scene_root)

    num_processes = os.cpu_count()
    with Pool(processes=num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(generate_scene, scene_ids), total=len(scene_ids)):
            pass
