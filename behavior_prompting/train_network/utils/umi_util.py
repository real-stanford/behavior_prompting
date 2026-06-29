import os
import cv2
import numpy as np
import imageio
import tempfile
import shutil
from behavior_prompting.common.pose_util import pose10d_to_mat
from behavior_prompting.common.trajectory_util import vis_trajectories, vis_pose_sequence


# wrt obs key pairs for each side: (pos_key, rot_key)
_WRT_KEYS = {
    'left': (
        'gripper_left_eef_pos_wrt_gripper_right',
        'gripper_left_eef_rot_axis_angle_wrt_gripper_right',
    ),
    'right': (
        'gripper_right_eef_pos_wrt_gripper_left',
        'gripper_right_eef_rot_axis_angle_wrt_gripper_left',
    ),
}

# action_dim = 10 per robot: pos(3) + rot6d(6) + gripper(1)
_DIM_PER_ROBOT = 10


def _pad_to_width(img, target_w, pad_side='both'):
    """Pad image width to target_w with white space.

    pad_side: 'left' | 'right' | 'both' (centred).
    """
    h, w, c = img.shape
    total = target_w - w
    if total <= 0:
        return img
    if pad_side == 'left':
        return np.hstack([np.full((h, total, c), 255, dtype='uint8'), img])
    elif pad_side == 'right':
        return np.hstack([img, np.full((h, total, c), 255, dtype='uint8')])
    else:
        left  = total // 2
        right = total - left
        return np.hstack([np.full((h, left,  c), 255, dtype='uint8'),
                          img,
                          np.full((h, right, c), 255, dtype='uint8')])


def vis_prompt(prompt, output_path, traj_scale: int = 2):
    """Visualize RGB streams and trajectories from a UMI prompt.

    Per-column layout (left then right if bimanual):
      row 1 : main camera        (H × W, padded to col_w)
      row 2 : ultrawide camera   (H × W, padded to col_w)
      row 3 : action-chunk trajectory  (col_w × col_w, square)
      row 4 : wrt-other-arm pose       (col_w × col_w, square)

    col_w = H * traj_scale so all open3d panels render square.
    Camera images are padded with white space to match col_w.

    Row 3 is animated (one output frame per action-chunk pose).
    Row 4 is a frozen snapshot updated once per prompt timestep.

    Args:
        traj_scale: square panel size = H * traj_scale.
    """
    fps = 20
    temp_dir = tempfile.mkdtemp()

    try:
        obs = prompt['obs']

        # Detect camera layout; fall back to legacy camera0 naming.
        # Check both main and ultrawide so ultrawide-only configs are detected.
        has_left  = 'camera_left_main_rgb'  in obs or 'camera_left_ultrawide_rgb'  in obs
        has_right = 'camera_right_main_rgb' in obs or 'camera_right_ultrawide_rgb' in obs
        sides = []
        if has_left:  sides.append('left')
        if has_right: sides.append('right')
        if not sides: sides = ['camera0']

        # Load camera frames for each side → (T, H, W, C) uint8
        cam = {}
        T = None
        for side in sides:
            prefix = 'camera0' if side == 'camera0' else f'camera_{side}'
            main_key = f'{prefix}_main_rgb'
            uw_key   = f'{prefix}_ultrawide_rgb'
            # Fall back to ultrawide if main is absent (ultrawide-only configs).
            main = obs.get(main_key) if main_key in obs else obs.get(uw_key)
            if main is None:
                continue
            uw = obs.get(uw_key, main)
            if T is None:
                T = main.shape[0]
            cam[side] = {
                'main':      (main * 255).astype('uint8').transpose(0, 2, 3, 1),
                'ultrawide': (uw   * 255).astype('uint8').transpose(0, 2, 3, 1),
            }

        _, H, W, _ = cam[sides[0]]['main'].shape
        col_w = H * traj_scale  # square panel side; cameras padded to this width

        # Actions: (T, chunk_n_actions, action_dim)
        actions = prompt['action']
        _, chunk_n_actions, action_dim = actions.shape

        # Map each side to its action pose slice (pos+rot6d, gripper excluded).
        n_robots = action_dim // _DIM_PER_ROBOT
        if n_robots >= 2 and has_left and has_right:
            action_pose_slice = {
                'left':  slice(0,   9),
                'right': slice(10, 19),
            }
        else:
            action_pose_slice = {side: slice(0, min(9, action_dim)) for side in sides}

        # ── Pre-generate wrt pose-sequence videos (one per side, T frames each) ────
        # Each frame shows only the current relative pose (fresh render, no accumulation).
        wrt_frames = {}  # {side: list of (col_w, col_w, 3) frames}
        for side in sides:
            if side not in _WRT_KEYS:
                continue
            pos_key, rot_key = _WRT_KEYS[side]
            if pos_key not in obs or rot_key not in obs:
                continue

            pos = obs[pos_key]  # (T, 3)
            rot = obs[rot_key]  # (T, 6) rot6d
            poses = pose10d_to_mat(np.concatenate([pos, rot], axis=-1))  # (T, 4, 4)

            traj_path = os.path.join(temp_dir, f'wrt_{side}.mp4')
            # Blue = left arm, red = right arm regardless of which is subject/reference.
            # For 'left' side: subject=left(blue), reference=right(red) → swap defaults.
            # For 'right' side: subject=right(red), reference=left(blue) → keep defaults.
            if side == 'left':
                pose_color, base_color = (0.2, 0.2, 1.0), (1.0, 0.2, 0.2)
            else:
                pose_color, base_color = (1.0, 0.2, 0.2), (0.2, 0.2, 1.0)
            vis_pose_sequence(traj_path, poses, out_width=col_w, out_height=col_w, fps=fps,
                              base_color=base_color, pose_color=pose_color)

            reader = imageio.get_reader(traj_path)
            wrt_frames[side] = [frame for frame in reader]
            reader.close()

        # ── Main loop: one action-chunk trajectory video per prompt timestep ──────
        video_writer = imageio.get_writer(output_path, fps=fps, codec='libx264')

        for t in range(T):
            # Generate per-side action chunk trajectory (col_w × col_w, square).
            action_traj_frames = {}
            for side in sides:
                aslice = action_pose_slice.get(side)
                if aslice is None:
                    continue
                poses = pose10d_to_mat(actions[t, :, aslice])  # (chunk_n_actions, 4, 4)

                traj_path = os.path.join(temp_dir, f'act_{t}_{side}.mp4')
                vis_trajectories(traj_path, poses, out_width=col_w, out_height=col_w,
                                 fps=fps, include_base_frame=True)

                reader = imageio.get_reader(traj_path)
                action_traj_frames[side] = [frame for frame in reader]
                reader.close()

            n_out = max((len(v) for v in action_traj_frames.values()), default=1)

            # Frozen wrt snapshot for this timestep.
            wrt_snapshot = {
                side: wrt_frames[side][min(t, len(wrt_frames[side]) - 1)]
                for side in wrt_frames
            }

            for f in range(n_out):
                cols = []
                for si, side in enumerate(sides):
                    # Pad cameras on the outer edge only so adjacent columns have no gap.
                    if len(sides) == 1:
                        pad_side = 'both'
                    elif si == 0:
                        pad_side = 'left'
                    elif si == len(sides) - 1:
                        pad_side = 'right'
                    else:
                        pad_side = 'both'
                    main_padded = _pad_to_width(cam[side]['main'][t],      col_w, pad_side)
                    uw_padded   = _pad_to_width(cam[side]['ultrawide'][t], col_w, pad_side)

                    parts = [main_padded, uw_padded]

                    # Action-chunk trajectory (already col_w × col_w)
                    if side in action_traj_frames:
                        fi = min(f, len(action_traj_frames[side]) - 1)
                        parts.append(action_traj_frames[side][fi])

                    # Wrt pose snapshot (already col_w × col_w)
                    if side in wrt_snapshot:
                        parts.append(wrt_snapshot[side])

                    cols.append(np.vstack(parts))

                # Pad columns to equal height, then place side-by-side.
                max_h = max(c.shape[0] for c in cols)
                padded = []
                for c in cols:
                    if c.shape[0] < max_h:
                        pad = np.full((max_h - c.shape[0], c.shape[1], 3), 255, dtype='uint8')
                        c = np.vstack([c, pad])
                    padded.append(c)

                video_writer.append_data(np.hstack(padded))

        video_writer.close()

    finally:
        shutil.rmtree(temp_dir)
