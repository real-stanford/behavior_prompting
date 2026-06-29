"""Utilities for aligning and visualizing trajectories"""

import numpy as np
import open3d as o3d
from behavior_prompting.common.latency_util import regular_sample, get_latency
from behavior_prompting.common.transform_util import pose_4x4_to_6d, pose_6d_to_4x4, pose_4x4_to_quat_xyzw
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from datetime import datetime, timezone
from collections import OrderedDict
import pandas as pd
import os
from behavior_prompting.common.transform_util import pose_6d_to_4x4
import imageio


def align_poses_different_timescales(pose1, pose2, time1, time2, show_alignment=False):
    # Use if pose timestamps are with respect to different timescales (their clocks are not aligned). This function aligns them using cross correlation. Returns aligned poses and times.

    # start both timescales at 0 (but at this point they are not aligned still)
    time1 -= min(time1)
    time2 -= min(time2)

    # Align trajectories using cross correlation
    t2 = time2
    x2 = pose_4x4_to_6d(pose2)
    t1 = time1
    x1 = pose_4x4_to_6d(pose1)

    n_dims = x1.shape[1]
    fig, axes = plt.subplots(n_dims, 3)
    fig.set_size_inches(15, 15, forward=True)

    # get independent latency for each dimension of position and rotation
    latencies = []
    for i in range(n_dims):
        latency, info = get_latency(x2[...,i], t2, x1[...,i], t1, force_positive=False)

        row = axes[i]
        ax = row[0]
        ax.plot(info['lags'], info['correlation'])
        ax.set_xlabel('lag')
        ax.set_ylabel('cross-correlation')
        ax.set_title(f"Action Dim {i} Cross Correlation")

        ax = row[1]
        ax.plot(t2, x2[...,i], label='target')
        ax.plot(t1, x1[...,i], label='actual')
        ax.set_xlabel('time')
        ax.set_ylabel('gripper-width')
        ax.legend()
        ax.set_title(f"Action Dim {i} Raw observation")

        ax = row[2]
        t_samples = info['t_samples'] - info['t_samples'][0]
        ax.plot(t_samples, info['x_target'], label='target')
        ax.plot(t_samples-latency, info['x_actual'], label='actual-latency')
        ax.set_xlabel('time')
        
        ax.set_ylabel('gripper-width')
        ax.legend()
        ax.set_title(f"Action Dim {i} Aligned with latency={latency:.04f}")
        latencies.append(latency)

    fig.tight_layout()
    if show_alignment:
        plt.show()

    avg_latency = np.mean(sorted(latencies)[1:-1]) # cut outliers (assumed to be first and last entries)

    t1 -= avg_latency # shift actual time to align with target time

    pose1 = pose_6d_to_4x4(x1)
    pose2 = pose_6d_to_4x4(x2)
    
    return pose1, pose2, t1, t2

def error_between_poses(poses1, poses2):
    # compute position error
    pos_delta = poses1[:,:3,3] - poses2[:,:3,3] 
    pos_err = np.linalg.norm(pos_delta, axis=1).mean()

    # compute rotation error (pre cut)
    delta_rotation_mat = poses1[:, :3, :3] @ np.linalg.inv(poses2[:, :3, :3])
    delta_rotation_rotvec = Rotation.from_matrix(delta_rotation_mat).as_rotvec()
    rot_err = np.linalg.norm(delta_rotation_rotvec, axis=1).mean()

    return pos_err, rot_err

def world_align_poses(poses, world_align_transform=None):
    # normalize poses with respect to the specified world transform or, if None, such that the first pose is the identity transformation
    if world_align_transform is None:
        world_align_transform = np.linalg.inv(poses[0])
    poses = np.array([world_align_transform @ pose for pose in poses]) # normalize with respect to initial frame pose
    return poses, world_align_transform

def get_time_aligned_poses(pose1, pose2, time1, time2, resample_dt=1/60):
    # required that time1 and time2 are on the same time scale (meaning that the times are recorded with respect to the same reference time). It's not required that time1 and time2 represent the same time intervals. Returns 6d poses that are sampled at the same times. Poses are in 4x4 format.
    t_start = max(time1[0],time2[0])
    t_end = min(time1[-1],time2[-1])

    n_samples = int((t_end - t_start) / resample_dt)
    t_samples = np.arange(n_samples) * resample_dt + t_start

    pose1_samples = sample_poses_at_times(pose1, time1, t_samples)
    pose2_samples = sample_poses_at_times(pose2, time2, t_samples)

    return pose1_samples, pose2_samples, t_samples

def sample_poses_at_times(poses, pose_times, sample_times):
    poses = pose_4x4_to_6d(poses)
    pose_samples = np.array([regular_sample(poses[:,i], pose_times, sample_times) for i in range(6)]).T
    pose_samples = pose_6d_to_4x4(pose_samples)
    return pose_samples

def _step_time_aligned_trajectories_incremental(offscreen, scene_or_vis, time_aligned_trajectories, material, axis_size=0.04, include_base_frame=True, line_width=3, colors=None, axis_every_n_steps=30):
    """You can either provide an offscreen scene or a visualizer. This handles both properly"""
    T = len(time_aligned_trajectories[0])
    if colors is None:
        colors = [[0,0,0], [0,1,0], [0,0,1], [0.6,0,0.6]]

    line_material = o3d.visualization.rendering.MaterialRecord()
    line_material.shader = "unlitLine"
    line_material.line_width = line_width

    def add_geo(name, geo, mat):
        if offscreen:
            scene = scene_or_vis
            scene.add_geometry(name, geo, mat)
        else:
            vis = scene_or_vis
            vis.add_geometry(geo)

    yield

    if type(axis_every_n_steps) == int:
        axis_every_n_steps = [axis_every_n_steps] * len(time_aligned_trajectories)

    end_spheres = [None] * len(time_aligned_trajectories)
    traj_started = [False] * len(time_aligned_trajectories)
    for step_i in range(T-1):
        for traj_i, trajectory in enumerate(time_aligned_trajectories):            
            trajectory_1 = trajectory[step_i:step_i+2,:3,3]

            if np.all(trajectory_1[0] == trajectory_1[1]):
                # if the trajectory doesn't move at all, then don't plot this segment
                continue

            # add line
            points = o3d.utility.Vector3dVector(trajectory_1)
            lines = o3d.utility.Vector2iVector([[0, 1]])
            colors_1 = o3d.utility.Vector3dVector([colors[traj_i % len(colors)]])
            line_set = o3d.geometry.LineSet(points=points, lines=lines)
            line_set.colors = colors_1
            add_geo(f"line_{traj_i}_{step_i}", line_set, line_material)

            # Frame every N steps
            if step_i % axis_every_n_steps[traj_i] == 0 or not traj_started[traj_i]:
                translation = trajectory[step_i][:3, 3]
                rotation = trajectory[step_i][:3, :3]
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=translation)
                frame.rotate(rotation, center=translation)
                add_geo(f"frame_{traj_i}_{step_i}", frame, material)

            if not traj_started[traj_i]:
                # Green start sphere
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=axis_size/4)
                sphere.translate(trajectory_1[0])
                sphere.paint_uniform_color([0, 1, 0])
                add_geo(f"start_sphere_{traj_i}", sphere, material)

                if include_base_frame:
                    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2*axis_size)
                    add_geo("base_frame", base_frame, material)

            # Red moving end sphere
            if end_spheres[traj_i] is not None:
                if offscreen:
                    scene_or_vis.remove_geometry(f"end_sphere_{traj_i}")
                else:
                    scene_or_vis.remove_geometry(end_spheres[traj_i])
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=axis_size/4)
            sphere.translate(trajectory_1[-1])
            sphere.paint_uniform_color([1, 0, 0])
            add_geo(f"end_sphere_{traj_i}", sphere, material)
            end_spheres[traj_i] = sphere

            traj_started[traj_i] = True
        
        yield

def get_open3d_geo_trajectories(time_aligned_trajectories, colors=None, axis_size=0.04, include_base_frame=True):
    """
    Gets geometry objects for one or more trajectories when can be then passed into an open3d visualizer. This function is not suitable for headless rendering.
    
    `time_aligned_trajectories` is a list of matrices with shape (T, 4, 4), where T is the number of time steps. Returns list of Open3D geometry objects that can be visualized.
    """
    T = len(time_aligned_trajectories[0])  # Number of time steps

    if colors is None:
        colors = [[1,0,0], [0,1,0], [0,0,1], [0.6,0,0.6]]  # Default colors

    assert len(colors) >= len(time_aligned_trajectories)

    geometries = []

    for i, trajectory in enumerate(time_aligned_trajectories):
        assert trajectory.shape == (T, 4, 4)

        trajectory_1 = trajectory[:,:3,3]

        # Create LineSet for trajectory 1
        lines_1 = [[i, i + 1] for i in range(T - 1)]
        colors_1 = [colors[i] for j in range(T - 1)]

        line_set_1 = o3d.geometry.LineSet()
        line_set_1.points = o3d.utility.Vector3dVector(trajectory_1)
        line_set_1.lines = o3d.utility.Vector2iVector(lines_1)
        line_set_1.colors = o3d.utility.Vector3dVector(colors_1)

        def plot_trajectory_with_rgb_axes(poses, K=1):
            """
            Plots the trajectory with RGB axes using Open3D.

            Parameters:
            - poses: numpy array of shape (T, 4, 4), where T is the number of trajectories.
            - K: Subsampling factor, use every K-th frame in the trajectory.
            """
            # Create a list to hold the frames for visualization
            frames = []
            
            # Subsample the poses
            subsampled_poses = poses[::K]
            
            for i, pose in enumerate(subsampled_poses):
                # Extract the translation vector
                translation = pose[:3, 3]
                
                # Extract the rotation matrix
                rotation = pose[:3, :3]
                
                # Create the coordinate frame
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=translation)
                
                # Apply the rotation to the frame
                frame.rotate(rotation, center=translation)
                
                # Append the frame to the list
                frames.append(frame)
            
            # Create an Open3D visualization object
            return frames

        frames = plot_trajectory_with_rgb_axes(trajectory, K=30)

        if len(trajectory_1) > 1:
            geometries.append(line_set_1)
        if include_base_frame:
            base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2*axis_size)
            geometries.append(base_frame)
        
        # green sphere at the start of the trajectory
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=axis_size/4)
        sphere.translate(trajectory_1[0])
        sphere.paint_uniform_color([0, 1, 0])
        geometries.append(sphere)

        # red sphere at the end of the trajectory
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=axis_size/4)
        sphere.translate(trajectory_1[-1])
        sphere.paint_uniform_color([1, 0, 0])
        geometries.append(sphere)

        geometries.extend(frames)
    
    return geometries


def vis_pose_sequence(video_out_path, poses, out_width=480, out_height=360, fps=20, axis_size=0.04,
                      base_color=(0.2, 0.2, 1.0), pose_color=(1.0, 0.2, 0.2)):
    """Render a sequence of independent poses (T frames), each as a fresh coordinate frame.

    Unlike vis_trajectories, geometry is cleared between frames so there is no accumulation.
    This is appropriate when consecutive poses are expressed in different coordinate frames
    (e.g. left-arm pose relative to right-arm at each timestep).

    Args:
        poses: (T, 4, 4) array of poses.
    """
    video_out_path = os.path.abspath(video_out_path)
    poses = np.asarray(poses)
    T = len(poses)

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"

    original_display = os.environ.get('DISPLAY')
    if original_display:
        os.environ.pop('DISPLAY', None)
    try:
        renderer = o3d.visualization.rendering.OffscreenRenderer(out_width, out_height)
        scene = renderer.scene
        scene.set_background([1, 1, 1, 1])
    finally:
        if original_display:
            os.environ['DISPLAY'] = original_display

    video_writer = imageio.get_writer(video_out_path, fps=fps, codec='libx264')

    with tqdm(total=T, leave=False, desc='vis_pose_sequence') as pbar:
        for i in range(T):
            scene.clear_geometry()

            # Base frame at origin (reference arm) with a small blue sphere
            base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size * 3)
            scene.add_geometry("base", base, material)
            base_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=axis_size * 0.3)
            base_sphere.paint_uniform_color(list(base_color))
            scene.add_geometry("base_sphere", base_sphere, material)

            # Current pose frame (subject arm) with a red sphere
            translation = poses[i, :3, 3]
            rotation    = poses[i, :3, :3]
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size * 3, origin=translation)
            frame.rotate(rotation, center=translation)
            scene.add_geometry("pose", frame, material)
            pose_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=axis_size * 0.3)
            pose_sphere.translate(translation)
            pose_sphere.paint_uniform_color(list(pose_color))
            scene.add_geometry("pose_sphere", pose_sphere, material)

            # Camera: Z into screen, X right, Y down, 30° down from horizon
            bbox = scene.bounding_box
            center = bbox.get_center()
            view_dist = max(np.max(bbox.get_extent()[:2]) * 1.5, 0.3)
            tilt = np.radians(30)
            scene.camera.look_at(
                center=center,
                eye=center + np.array([0, -view_dist * np.sin(tilt), -view_dist * np.cos(tilt)]),
                up=[0, -1, 0],
            )

            rendered = renderer.render_to_image()
            video_writer.append_data(np.asarray(rendered))
            pbar.update()

    video_writer.close()
    print(f"Saved video to {video_out_path}")


def vis_trajectories(video_out_path, poses, is_lost=None, out_width=480, out_height=360, offscreen=True, fps=20, include_base_frame=False, axis_every_n_steps=30):
    """
    Plots a series of one or more trajectories by saving to a video file.

    `offscreen` flag should be true if you are on a headless machine and want to use OffscreenRenderer and false if you are on a machine with a display and want to use Visualizer.
    Note that line width changes and the setting of up axis only apply if offscreen is true.
    """
    video_out_path = os.path.abspath(video_out_path)

    if type(poses) != list:
        poses = [poses]

    T = None
    for cur_poses in poses:
        if T is None:
            T = len(cur_poses)
        else:
            assert T == len(cur_poses)
    
    out_H = out_height
    out_W = out_width

    if is_lost is None:
        is_lost = np.zeros(T, dtype=bool)

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"

    # Setup renderer
    if offscreen:
        # Unset DISPLAY for EGL headless rendering (EGL conflicts with X11 forwarding)
        # Save original DISPLAY value and restore it after renderer creation
        original_display = os.environ.get('DISPLAY')
        if original_display:
            # Temporarily unset DISPLAY to allow EGL to initialize properly
            os.environ.pop('DISPLAY', None)
        try:
            renderer = o3d.visualization.rendering.OffscreenRenderer(out_W, out_H)
            scene = renderer.scene
            scene.set_background([1, 1, 1, 1])  # white background
        finally:
            # Restore DISPLAY if it was set
            if original_display:
                os.environ['DISPLAY'] = original_display
    else:
        renderer = o3d.visualization.Visualizer()
        renderer.create_window(visible=True, width=out_W, height=out_H)
        scene = renderer

    video_writer = imageio.get_writer(video_out_path, fps=fps, codec='libx264')

    present_poses = [cur_poses[np.where(~is_lost)] for cur_poses in poses]
    geom_stepper = _step_time_aligned_trajectories_incremental(offscreen, scene, present_poses, material, include_base_frame=include_base_frame, axis_every_n_steps=axis_every_n_steps)
    
    num_present_poses = len(present_poses[0])

    with tqdm(total=num_present_poses, leave=False, desc='vis_trajectories') as pbar:
        for i in range(T):
            if is_lost[i]:
                continue

            next(geom_stepper)

            if offscreen:
                # update camera position: Z into screen, X right, Y down, 30° down from horizon
                bbox = scene.bounding_box
                center = bbox.get_center()
                view_dist = max(np.max(bbox.get_extent()[:2]) * 1.5, 0.3)
                tilt = np.radians(30)
                scene.camera.look_at(
                    center=center,
                    eye=center + np.array([0, -view_dist * np.sin(tilt), -view_dist * np.cos(tilt)]),
                    up=[0, -1, 0],
                )

                # Render to image
                rendered = renderer.render_to_image()
                rendered_np = np.asarray(rendered)
                rendered_bgr = cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR)
            else:
                # Update the visualizer with the current frame
                renderer.poll_events()
                renderer.update_renderer()

                # Capture the visualizer screen image
                trajectory_image = renderer.capture_screen_float_buffer(do_render=False)
                trajectory_image = (255 * np.asarray(trajectory_image)).astype(np.uint8)
                rendered_bgr = cv2.cvtColor(trajectory_image, cv2.COLOR_RGB2BGR)

            video_writer.append_data(rendered_bgr[...,::-1]) # convert to RGB

            pbar.update()

    video_writer.close()

    if not offscreen:
        cv2.destroyAllWindows()
        renderer.destroy_window()

    print(f"Saved video to {video_out_path}")

def vis_video_aligned_trajectories(video_path, video_out_path, poses, is_lost=None, max_frames=-1, trajectory_render_size=360, offscreen=True, axis_every_n_steps=30):
    """
    Plots a video alongside one or more trajectories, saving the result as a video file.

    - The input video is scaled so its height matches trajectory_render_size, and its width is scaled proportionally.
    - The trajectory is rendered as a square of size (trajectory_render_size, trajectory_render_size).
    - Each output frame consists of:
        - Left: input video frame, resized to (scaled_W, trajectory_render_size), maintaining aspect ratio.
        - Right: trajectory rendering, always (trajectory_render_size, trajectory_render_size).
    - The output video has frame size (scaled_W + trajectory_render_size, trajectory_render_size).

    Args:
        video_path (str): Path to the input video.
        video_out_path (str): Path to save the output video.
        poses (list or np.ndarray): List of pose arrays, each of length T.
        is_lost (np.ndarray or None): Boolean mask for lost frames.
        max_frames (int): Maximum number of frames to visualize.
        trajectory_render_size (int): Height (and width) of the square trajectory rendering.
        offscreen (bool): Use offscreen rendering.
    """
    assert os.path.exists(video_path)
    cap = cv2.VideoCapture(video_path)
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if type(poses) != list:
        poses = [poses]

    for cur_poses in poses:
        assert T == len(cur_poses)
    
    # Compute scaled width for input video
    scale = trajectory_render_size / H
    scaled_W = int(W * scale)
    out_H = trajectory_render_size
    traj_W = trajectory_render_size  # Trajectory render is always square
    out_W = scaled_W + traj_W

    if is_lost is None:
        is_lost = np.zeros(T, dtype=bool)

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"

    # Setup renderer for trajectory
    if offscreen:
        # Unset DISPLAY for EGL headless rendering (EGL conflicts with X11 forwarding)
        # Save original DISPLAY value and restore it after renderer creation
        original_display = os.environ.get('DISPLAY')
        if original_display:
            # Temporarily unset DISPLAY to allow EGL to initialize properly
            os.environ.pop('DISPLAY', None)
        try:
            renderer = o3d.visualization.rendering.OffscreenRenderer(traj_W, out_H)
            scene = renderer.scene
            scene.set_background([1, 1, 1, 1])  # white background
        finally:
            # Restore DISPLAY if it was set
            if original_display:
                os.environ['DISPLAY'] = original_display
    else:
        renderer = o3d.visualization.Visualizer()
        renderer.create_window(visible=True, width=traj_W, height=out_H)
        scene = renderer

    video_writer = imageio.get_writer(video_out_path, fps=fps, codec='libx264')

    present_poses = [cur_poses[np.where(~is_lost)] for cur_poses in poses]
    geom_stepper = _step_time_aligned_trajectories_incremental(offscreen, scene, present_poses, material, include_base_frame=False, axis_every_n_steps=axis_every_n_steps)
    
    num_present_poses = len(present_poses[0])
    max_frames = num_present_poses if max_frames == -1 else min(num_present_poses, max_frames)
    num_frames_visualized = 0

    with tqdm(total=max_frames, leave=False, desc='vis_video_aligned_trajectories') as pbar:
        for i in range(T):
            found_frame, video_frame = cap.read()
            assert found_frame

            if is_lost[i]:
                continue

            next(geom_stepper)

            if offscreen:
                # update camera position: Z into screen, X right, Y down, 30° down from horizon
                bbox = scene.bounding_box
                center = bbox.get_center()
                view_dist = max(np.max(bbox.get_extent()[:2]) * 1.5, 0.3)
                tilt = np.radians(30)
                scene.camera.look_at(
                    center=center,
                    eye=center + np.array([0, -view_dist * np.sin(tilt), -view_dist * np.cos(tilt)]),
                    up=[0, -1, 0],
                )

                # Render to image
                rendered = renderer.render_to_image()
                rendered_np = np.asarray(rendered)
                rendered_bgr = cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR)
            else:
                # Update the visualizer with the current frame
                renderer.poll_events()
                renderer.update_renderer()

                # Capture the visualizer screen image
                trajectory_image = renderer.capture_screen_float_buffer(do_render=False)
                trajectory_image = (255 * np.asarray(trajectory_image)).astype(np.uint8)
                rendered_bgr = cv2.cvtColor(trajectory_image, cv2.COLOR_RGB2BGR)

            # Resize video frame to (scaled_W, out_H), maintaining aspect ratio
            video_frame_resized = cv2.resize(video_frame, dsize=(scaled_W, out_H), interpolation=cv2.INTER_CUBIC)

            # Ensure trajectory render is (traj_W, out_H)
            if rendered_bgr.shape[1] != traj_W or rendered_bgr.shape[0] != out_H:
                rendered_bgr = cv2.resize(rendered_bgr, dsize=(traj_W, out_H), interpolation=cv2.INTER_CUBIC)

            # Concatenate and write
            combined_image = np.hstack((video_frame_resized, rendered_bgr))
            video_writer.append_data(combined_image[...,::-1]) # convert to RGB

            num_frames_visualized += 1
            pbar.update()
            if num_frames_visualized >= max_frames:
                break

    cap.release()
    video_writer.close()

    if not offscreen:
        cv2.destroyAllWindows()
        renderer.destroy_window()

def save_trajectory_umi_format(out_csv_path, poses, times, is_lost):
    # output the poses as csv (matching the format in UMI pipeline) which is in quat_xyzw and with relative timestamps starting at 0
    pos_quat_xyzw = pose_4x4_to_quat_xyzw(poses)

    csv_data = OrderedDict({
        'frame_idx': np.arange(len(times)),
        'timestamp': [time - times[0] for time in times],
        'state': [2] * len(times),
        'is_lost': is_lost,
        'is_keyframe': [False] * len(times),
        'x': pos_quat_xyzw[:, 0],
        'y': pos_quat_xyzw[:, 1],
        'z': pos_quat_xyzw[:, 2],
        'q_x': pos_quat_xyzw[:, 3],
        'q_y': pos_quat_xyzw[:, 4],
        'q_z': pos_quat_xyzw[:, 5],
        'q_w': pos_quat_xyzw[:, 6],
    })
    df = pd.DataFrame(csv_data)

    formats = {'timestamp': '{:.6f}'}
    for key in ['x', 'y', 'z', 'q_x', 'q_y', 'q_z', 'q_w']:
        formats[key] = '{:.9f}'

    for col, f in formats.items():
        df[col] = df[col].map(lambda x: f.format(x))

    df.to_csv(out_csv_path, index=False)

if __name__ == "__main__":
    # Create linearly interpolated poses from 0 to 2pi rotation around z and 0 to 10 in position
    angles = np.linspace(0.1, 2*np.pi, 100)
    positions = np.linspace(0, 0.5, 100)
    poses = np.zeros((100, 6))  # [x,y,z, rx,ry,rz] format
    poses[:, 0] = positions  # Linear motion along x-axis
    poses[:, 5] = angles  # Rotate around z-axis
    
    # Convert to 4x4 pose matrices
    poses_4x4 = pose_6d_to_4x4(poses)

    vis_trajectories("tmp_test.mp4", poses_4x4)
