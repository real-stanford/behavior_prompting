"""
RMQ-based action replay server for real-env bimanual iPhUMI deployment.

Demo poses from a UMI replay buffer are in an arbitrary world frame W from iPhone
camera tracking — NOT aligned to either robot base frame.  Delta actions between
consecutive demo frames are frame-invariant (the world frame cancels in
inv(demo[t]) @ demo[t+k]), so open-loop replay is correct as long as the robot starts
in the same initial inter-gripper configuration as the demo.

Workflow:
  1. Start this server.
  2. In the task, press 'o': the task queries 'query_initial_gripper_transform',
     receives the demo's initial left-wrt-right inter-gripper transform, and moves
     the left arm to the correct position (right arm stays fixed).
  3. Press 'c' in the task to start replay.

Topics served (compatible with real_env PolicyAgent):
  query_initial_gripper_transform → {} → {"transform_xyz_wxyz": list[7]}
  policy_config, policy_reset, policy_prompt, policy_inference, export_recorded_data

Action output keys (pos3+rot6d, delta relative to current demo step):
  action_{left,right}_eef_pos:            (K, 3)
  action_{left,right}_eef_rot_axis_angle: (K, 6)
  action_{left,right}_gripper_width:      (K, 1)
"""

import datetime
import sys
import time
import traceback

import click
import numpy as np
import robotmq
import zarr
from scipy.spatial.transform import Rotation

from behavior_prompting.common.pose_util import mat_to_rot6d


def _echo_exception() -> str:
    exc_type, exc_value, exc_tb = sys.exc_info()
    return "".join(traceback.format_exception(exc_type, exc_value, exc_tb))


def _aa_to_xyz_wxyz(pos: np.ndarray, aa: np.ndarray) -> np.ndarray:
    """(N, 3) xyz + (N, 3) axis-angle rotvec → (N, 7) xyz_wxyz."""
    q_xyzw = Rotation.from_rotvec(aa).as_quat()
    q_wxyz = q_xyzw[:, [3, 0, 1, 2]]
    return np.concatenate([pos, q_wxyz], axis=-1)


def _xyz_wxyz_to_pose_9d(pose_xyz_wxyz: np.ndarray) -> np.ndarray:
    """(..., 7) xyz_wxyz → (..., 9) pos3+rot6d."""
    pos = pose_xyz_wxyz[..., :3]
    wxyz = pose_xyz_wxyz[..., 3:]
    xyzw = np.concatenate([wxyz[..., 1:], wxyz[..., :1]], axis=-1)
    rotmat = Rotation.from_quat(xyzw).as_matrix()
    return np.concatenate([pos, mat_to_rot6d(rotmat)], axis=-1)


def _get_relative_pose(new_xyz_wxyz: np.ndarray, base_xyz_wxyz: np.ndarray) -> np.ndarray:
    """Both (7,) xyz_wxyz → (7,) inv(base) @ new."""
    def to_mat(p):
        m = np.eye(4)
        m[:3, :3] = Rotation.from_quat(p[[4, 5, 6, 3]]).as_matrix()
        m[:3, 3] = p[:3]
        return m
    rel = np.linalg.inv(to_mat(base_xyz_wxyz)) @ to_mat(new_xyz_wxyz)
    pos = rel[:3, 3]
    xyzw = Rotation.from_matrix(rel[:3, :3]).as_quat()
    wxyz = xyzw[[3, 0, 1, 2]]
    if wxyz[0] < 0:
        wxyz = -wxyz
    return np.concatenate([pos, wxyz])


class ActionReplayNodeRealEnv:
    def __init__(
        self,
        replay_buffer_path: str,
        task_name: str,
        task_index: int,
        server_endpoint: str,
        action_horizon: int,
        stride: int,
        sides: str = "both",
        start_trim: int = 0,
    ):
        rb = zarr.open(replay_buffer_path, "r")

        rb_task_names = list(rb["meta/task_names"][:])
        assert task_name in rb_task_names, (
            f"Task {task_name!r} not found. Available: {rb_task_names}"
        )
        task_idx = rb_task_names.index(task_name)

        task_data_ends = rb["meta/task_data_ends"][:]
        task_data_starts = np.concatenate([[0], task_data_ends[:-1]])
        task_start = int(task_data_starts[task_idx])
        task_end = int(task_data_ends[task_idx])

        episode_ends = rb["meta/episode_ends"][:]
        episode_starts = np.concatenate([[0], episode_ends[:-1]])

        task_episodes = [
            i for i in range(len(episode_ends))
            if int(episode_starts[i]) >= task_start and int(episode_ends[i]) <= task_end + 1
        ]
        if not task_episodes:
            task_episodes = list(range(len(episode_ends)))

        assert task_index < len(task_episodes), (
            f"task_index={task_index} out of range; "
            f"task {task_name!r} has {len(task_episodes)} episode(s)"
        )
        ep_idx = task_episodes[task_index]
        start_frame = int(episode_starts[ep_idx])
        end_frame = min(int(episode_ends[ep_idx]), task_end)
        episode_length = end_frame - start_frame

        assert 0 <= start_trim < episode_length, (
            f"start_trim={start_trim} must be in [0, {episode_length})"
        )
        start_frame += start_trim

        print(
            f"[replay] Task={task_name!r} idx={task_index} → "
            f"episode {ep_idx}, frames [{start_frame}, {end_frame}), "
            f"length={end_frame - start_frame}"
            + (f" (skipped first {start_trim} frames)" if start_trim else "")
        )

        left_pos = rb["data/gripper_left_eef_pos"][start_frame:end_frame]
        left_aa  = rb["data/gripper_left_eef_rot_axis_angle"][start_frame:end_frame]
        right_pos = rb["data/gripper_right_eef_pos"][start_frame:end_frame]
        right_aa  = rb["data/gripper_right_eef_rot_axis_angle"][start_frame:end_frame]

        self.demo_left_xyz_wxyz  = _aa_to_xyz_wxyz(left_pos,  left_aa)   # (N, 7) world frame W
        self.demo_right_xyz_wxyz = _aa_to_xyz_wxyz(right_pos, right_aa)  # (N, 7) world frame W
        self.demo_left_gripper   = rb["data/gripper_left_gripper_width"][start_frame:end_frame]
        self.demo_right_gripper  = rb["data/gripper_right_gripper_width"][start_frame:end_frame]
        self.demo_length = end_frame - start_frame

        # Initial inter-gripper transform: left wrt right at step 0, in world frame W.
        # Physical meaning: left gripper pose expressed in the right gripper's local frame.
        # The task uses this (with tx_left_right_base) to align the left arm before replay.
        self.demo_initial_intergrip_xyz_wxyz = _get_relative_pose(
            self.demo_left_xyz_wxyz[0], self.demo_right_xyz_wxyz[0]
        )

        self.action_horizon = action_horizon
        self.stride         = stride  # demo frames per deployment action step
        self.demo_step      = 0
        self.replay_sides: frozenset[int] = {
            "both": frozenset({0, 1}),
            "left":  frozenset({0}),
            "right": frozenset({1}),
        }[sides]

        self.server_endpoint = server_endpoint
        self.server = robotmq.RMQServer("policy_server", server_endpoint)
        for topic in (
            "policy_config",
            "policy_reset",
            "policy_prompt",
            "policy_inference",
            "export_recorded_data",
            "query_initial_gripper_transform",
        ):
            self.server.add_topic(topic, message_remaining_time_s=60)

    def _build_policy_config(self) -> dict:
        now = datetime.datetime.now()
        return {
            "workspace": {
                "proprio_length": 1, # doesn't matter since we aren't using obs
                "obs_down_sample_steps": 1, # doesn't matter since we aren't using obs
                "task": {"shape_meta": {"obs": {}}, "img_obs_horizon": 1},
            },
            "use_prompting": False,
            "policy_name": "action_replay",
            "run_name": "action_replay",
            "epoch": 0,
            "date_str": now.strftime("%Y-%m-%d"),
            "time_str": now.strftime("%H-%M-%S"),
        }

    def run_node(self):
        print(
            f"[replay] Serving on {self.server_endpoint} "
            f"(demo_length={self.demo_length}, "
            f"action_horizon={self.action_horizon}, stride={self.stride})"
        )
        print(
            f"[replay] Initial inter-gripper transform (left wrt right, xyz_wxyz): "
            f"{self.demo_initial_intergrip_xyz_wxyz}"
        )
        while True:
            data, topic = self.server.wait_for_request(timeout_s=0.1)
            if topic is None:
                continue

            if topic == "policy_config":
                try:
                    self.server.reply_request(
                        topic=topic, data=robotmq.serialize(self._build_policy_config())
                    )
                except Exception:
                    err = _echo_exception()
                    print(f"[replay] policy_config error:\n{err}")
                    self.server.reply_request(topic=topic, data=robotmq.serialize(f"ERROR: {err}"))

            elif topic == "policy_reset":
                self.demo_step = 0
                print("[replay] Reset → demo_step=0")
                self.server.reply_request(topic=topic, data=robotmq.serialize("OK"))

            elif topic == "policy_prompt":
                print("[replay] policy_prompt ignored (action replay has no prompting)")
                self.server.reply_request(topic=topic, data=robotmq.serialize("OK"))

            elif topic == "policy_inference":
                try:
                    obs = robotmq.deserialize(data)
                    start_t = time.monotonic()
                    actions = self._handle_inference(obs)
                    print(
                        f"[replay] inference: demo_step={self.demo_step}/{self.demo_length}, "
                        f"time={time.monotonic() - start_t:.3f}s"
                    )
                    self.server.reply_request(topic=topic, data=robotmq.serialize(actions))
                except Exception:
                    err = _echo_exception()
                    print(f"[replay] policy_inference error:\n{err}")
                    self.server.reply_request(topic=topic, data=robotmq.serialize(f"ERROR: {err}"))

            elif topic == "export_recorded_data":
                self.server.reply_request(topic=topic, data=robotmq.serialize(""))

            elif topic == "query_initial_gripper_transform":
                self.server.reply_request(
                    topic=topic,
                    data=robotmq.serialize(
                        {"transform_xyz_wxyz": self.demo_initial_intergrip_xyz_wxyz.tolist()}
                    ),
                )

    def _handle_inference(self, obs: dict) -> dict:
        """
        Return open-loop delta actions from the demo trajectory.

        At demo_step t, action[k] = inv(demo[t]) @ demo[t+k+1] for each arm.
        Because the world frame W cancels in this relative computation, the delta is
        equivalent to inv(robot[t]) @ robot[t+k+1] — correct regardless of where the
        robot base frames sit relative to W, as long as the initial inter-gripper
        transform was matched via the 'o'-key alignment.
        """
        K = self.action_horizon
        output: dict = {}
        for arm_idx, demo_xyz_wxyz, demo_gripper in [
            (0, self.demo_left_xyz_wxyz,  self.demo_left_gripper),
            (1, self.demo_right_xyz_wxyz, self.demo_right_gripper),
        ]:
            if arm_idx not in self.replay_sides:
                # Hold: identity relative pose (arm stays put), freeze gripper at current demo step.
                output[f"action{arm_idx}_eef_pos"]            = np.zeros((K, 3), dtype=np.float32)
                output[f"action{arm_idx}_eef_rot_axis_angle"] = np.tile(
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], (K, 1)
                ).astype(np.float32)
                output[f"action{arm_idx}_gripper_width"] = np.full(
                    (K, 1), demo_gripper[self.demo_step], dtype=np.float32
                )
                continue

            ref = demo_xyz_wxyz[self.demo_step]  # (7,) reference at current step

            target_indices = [
                min(self.demo_step + (k + 1) * self.stride, self.demo_length - 1)
                for k in range(self.action_horizon)
            ]
            targets = demo_xyz_wxyz[target_indices]  # (K, 7)

            rel_poses = np.array([
                _get_relative_pose(targets[k], ref)
                for k in range(len(targets))
            ], dtype=np.float64)  # (K, 7)

            pose9d = _xyz_wxyz_to_pose_9d(rel_poses)  # (K, 9)
            output[f"action{arm_idx}_eef_pos"]            = pose9d[:, :3].astype(np.float32)
            output[f"action{arm_idx}_eef_rot_axis_angle"] = pose9d[:, 3:].astype(np.float32)
            output[f"action{arm_idx}_gripper_width"]      = demo_gripper[target_indices].astype(np.float32)

        if self.demo_step >= self.demo_length - 1:
            print("[replay] WARNING: reached end of demo — holding last frame")
        else:
            self.demo_step = min(
                self.demo_step + self.action_horizon * self.stride, self.demo_length - 1
            )

        return output


@click.command()
@click.option("--replay-buffer-path", "-r", required=True, help="Path to .zarr.zip replay buffer")
@click.option("--task-name", "-t", required=True, help="Task name (must match meta/task_names)")
@click.option("--task-index", "-n", default=0, show_default=True, help="Episode index within the task")
@click.option("--endpoint", default="tcp://0.0.0.0:18765", show_default=True, help="RMQ server endpoint")
@click.option("--action-horizon", default=16, show_default=True, help="Actions per inference call and demo frames advanced per call")
@click.option(
    "--stride", default=1, show_default=True,
    help="Use this to replay the trajectory at a different rate. At 1 the trajectory executes every step of the 60Hz demonstration. If the robot is exexcuting at 10Hz this means that the action replay will take 6 times as long as the original demo.",
)
@click.option(
    "--sides", default="both", show_default=True,
    type=click.Choice(["both", "left", "right"]),
    help="Which arm(s) to replay. The other arm holds its current pose.",
)
@click.option(
    "--start-trim", default=0, show_default=True, type=int,
    help="Number of demo frames to skip from the start.",
)
def main(replay_buffer_path, task_name, task_index, endpoint, action_horizon, stride, sides, start_trim):
    node = ActionReplayNodeRealEnv(
        replay_buffer_path=replay_buffer_path,
        task_name=task_name,
        task_index=task_index,
        server_endpoint=endpoint,
        action_horizon=action_horizon,
        stride=stride,
        sides=sides,
        start_trim=start_trim,
    )
    node.run_node()


if __name__ == "__main__":
    main()
