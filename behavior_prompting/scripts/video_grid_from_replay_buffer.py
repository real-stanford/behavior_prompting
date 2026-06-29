#!/usr/bin/env python3
"""
Video Grid from Replay Buffer

Extracts video sequences from tasks in a replay buffer and creates a video grid.
"""

import os
import argparse
import math
import tempfile
import shutil
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import imageio
from moviepy import VideoFileClip, CompositeVideoClip, clips_array, TextClip, ImageClip, concatenate_videoclips
from moviepy.video.VideoClip import ColorClip
from tqdm import tqdm

from behavior_prompting.common.replay_buffer import ReplayBuffer
from behavior_prompting.common.imagecodecs_numcodecs import register_codecs

register_codecs()


def select_tasks(
    replay_buffer: ReplayBuffer,
    n: Optional[int],
    unique_tasks: bool,
    in_task_name: Optional[str] = None,
) -> List[int]:
    """
    Select which task indices to include.

    Args:
        replay_buffer: The replay buffer to select tasks from
        n: Maximum number of tasks to select (None = no limit)
        unique_tasks: If True, only include one instance per unique task name
        in_task_name: If provided, only include tasks whose names contain this string

    Returns:
        List of task indices to include
    """
    n_tasks = replay_buffer.n_tasks
    if n_tasks == 0:
        return []

    selected_indices = []
    seen_task_names = set()

    for task_idx in range(n_tasks):
        if n is not None and len(selected_indices) >= n:
            break

        task_name = replay_buffer.get_task_name(task_idx)

        if in_task_name is not None and in_task_name not in task_name:
            continue

        if unique_tasks:
            if task_name not in seen_task_names:
                selected_indices.append(task_idx)
                seen_task_names.add(task_name)
        else:
            selected_indices.append(task_idx)

    return selected_indices


def _frame_to_uint8_hwc(frame: np.ndarray) -> np.ndarray:
    """Convert a single frame to (H, W, C) uint8."""
    if not isinstance(frame, np.ndarray):
        return None
    if len(frame.shape) == 3:
        if frame.shape[0] < frame.shape[2] and frame.shape[0] <= 4:
            frame = np.transpose(frame, (1, 2, 0))
    elif len(frame.shape) == 2:
        frame = np.expand_dims(frame, axis=-1)
        frame = np.repeat(frame, 3, axis=-1)
    else:
        return None
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def extract_video_from_task(
    replay_buffer_path: str,
    task_idx: int,
    rgb_key: str,
    output_path: str,
    fps: float,
    flip_h: bool = False,
) -> Optional[str]:
    """
    Extract video sequence from a single task and save to file.
    Opens the replay buffer from path (for use in workers).

    Returns:
        output_path on success, None on failure
    """
    try:
        replay_buffer = ReplayBuffer.create_from_path(replay_buffer_path, mode="r")
        task_data = replay_buffer.get_task(task_idx, copy=True)
        if rgb_key not in task_data["data"]:
            return None
        rgb_data = task_data["data"][rgb_key]
        if len(rgb_data) == 0:
            return None

        out = imageio.get_writer(output_path, fps=fps, codec="libx264")
        try:
            for i in range(len(rgb_data)):
                frame = rgb_data[i]
                frame = _frame_to_uint8_hwc(np.asarray(frame))
                if frame is None:
                    continue
                if flip_h:
                    frame = frame[::-1, :, :]  # (H, W, C) flip H to match libero dataset orientation
                out.append_data(frame)
        finally:
            out.close()

        return output_path
    except Exception:
        return None


def extract_videos_from_tasks(
    replay_buffer: ReplayBuffer,
    replay_buffer_path: str,
    task_indices: List[int],
    rgb_key: str,
    fps: float,
    temp_dir: str,
    max_workers: int,
    task_names_for_overlay: Optional[List[str]] = None,
    flip_h: bool = False,
) -> List[Tuple[str, str]]:
    """
    Extract videos from selected tasks. Returns list of (video_path, task_name).
    """
    os.makedirs(temp_dir, exist_ok=True)
    results: List[Tuple[str, str]] = []
    task_names = (
        task_names_for_overlay
        if task_names_for_overlay is not None
        else [replay_buffer.get_task_name(i) for i in task_indices]
    )

    if max_workers <= 1:
        for idx, task_idx in enumerate(tqdm(task_indices, desc="Extracting videos", unit="task")):
            out_path = os.path.join(temp_dir, f"task_{task_idx:06d}.mp4")
            path = extract_video_from_task(
                replay_buffer_path, task_idx, rgb_key, out_path, fps, flip_h=flip_h
            )
            if path is not None:
                results.append((path, task_names[idx]))
            else:
                tqdm.write(f"Warning: Failed to extract video for task {task_idx}")
    else:
        # Collect (idx, path, task_name) then sort by idx so grid order is stable
        completed: List[Tuple[int, str, str]] = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            for idx, task_idx in enumerate(task_indices):
                out_path = os.path.join(temp_dir, f"task_{task_idx:06d}.mp4")
                future = executor.submit(
                    extract_video_from_task,
                    replay_buffer_path,
                    task_idx,
                    rgb_key,
                    out_path,
                    fps,
                    flip_h,
                )
                future_to_idx[future] = (idx, task_idx, out_path)
            
            # Overall progress bar for parallel extraction
            with tqdm(total=len(task_indices), desc="Extracting videos", unit="task") as pbar:
                for future in as_completed(future_to_idx):
                    idx, task_idx, out_path = future_to_idx[future]
                    try:
                        path = future.result()
                        if path is not None:
                            completed.append((idx, path, task_names[idx]))
                        else:
                            tqdm.write(f"Warning: Failed to extract video for task {task_idx}")
                    except Exception as e:
                        tqdm.write(f"Warning: Error extracting task {task_idx}: {e}")
                    finally:
                        pbar.update(1)
        completed.sort(key=lambda x: x[0])
        results = [(path, name) for _, path, name in completed]
    return results


def add_text_overlay(
    clip: VideoFileClip, text: str, fontsize: int = 12, position: str = "top", margin: int = 10
) -> CompositeVideoClip:
    """Add a text overlay to a video clip."""
    if position == "top":
        text_position = ("center", margin)
    elif position == "bottom":
        text_position = ("center", clip.h - margin - fontsize)
    else:
        text_position = ("center", "center")
    txt_clip = TextClip(
        text=text,
        font_size=fontsize,
        color="white",
        stroke_color="black",
        stroke_width=2,
        duration=clip.duration,
    ).with_position(text_position)
    return CompositeVideoClip([clip, txt_clip])


def calculate_grid_dimensions(num_videos: int, max_columns: Optional[int] = None) -> Tuple[int, int]:
    """Calculate grid dimensions (rows, cols)."""
    if max_columns is not None:
        cols = min(max_columns, num_videos)
        rows = math.ceil(num_videos / cols)
    else:
        cols = math.ceil(math.sqrt(num_videos))
        rows = math.ceil(num_videos / cols)
    return rows, cols


def create_video_grid(
    video_paths: List[str],
    output_path: str,
    fps: float,
    max_duration: Optional[float] = None,
    max_frames: Optional[int] = None,
    max_columns: int = 10,
    enable_text_overlay: bool = False,
    overlay_labels: Optional[List[str]] = None,
    loop_shorter: bool = False,
    output_resolution: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Create a grid video from the given video paths.
    Shorter clips are either frozen on last frame (default) or looped (if loop_shorter).
    """
    output_path = os.path.abspath(output_path)
    if overlay_labels is None:
        overlay_labels = [os.path.splitext(os.path.basename(p))[0] for p in video_paths]

    print(f"Loading {len(video_paths)} videos...")
    clips = []
    successful_paths = []
    successful_labels = []
    for path in video_paths:
        try:
            clip = VideoFileClip(path)
            clips.append(clip)
            successful_paths.append(path)
            successful_labels.append(overlay_labels[len(successful_paths)])
            print(f"Loaded: {os.path.basename(path)} ({clip.duration:.2f}s)")
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
            continue

    if not clips:
        raise ValueError("No valid video clips could be loaded")

    if max_frames is not None:
        max_duration = max_frames / fps
        print(f"Limiting to {max_frames} frames ({max_duration:.2f}s at {fps} fps)")

    longest_duration = max(c.duration for c in clips)
    target_duration = longest_duration if max_duration is None else min(max_duration, longest_duration)

    trimmed_clips = []
    for clip in clips:
        if clip.duration > target_duration:
            trimmed_clip = clip.with_duration(target_duration)
            trimmed_clips.append(trimmed_clip)
        elif clip.duration < target_duration:
            if loop_shorter:
                trimmed_clip = clip.loop(duration=target_duration)
            else:
                # Freeze on last frame: concatenate video with a clip of its last frame
                last_frame = clip.get_frame(max(0, clip.duration - 0.01))
                hold_duration = target_duration - clip.duration
                last_frame_clip = ImageClip(last_frame, duration=hold_duration)
                trimmed_clip = concatenate_videoclips([clip, last_frame_clip])
            trimmed_clips.append(trimmed_clip)
        else:
            trimmed_clips.append(clip)

    if enable_text_overlay:
        print("Adding text overlays...")
        overlayed_clips = []
        for clip, label in zip(trimmed_clips, successful_labels):
            overlayed_clips.append(add_text_overlay(clip, label))
    else:
        overlayed_clips = trimmed_clips

    first_clip = overlayed_clips[0]
    video_width, video_height = first_clip.w, first_clip.h
    if output_resolution is not None:
        out_w, out_h = output_resolution
        # Resize all clips to output resolution
        resized = []
        for c in overlayed_clips:
            resized.append(c.resized((out_w, out_h)))
        overlayed_clips = resized
        video_width, video_height = out_w, out_h

    rows, cols = calculate_grid_dimensions(len(overlayed_clips), max_columns)
    print(f"Creating {rows}x{cols} grid...")

    grid_clips = []
    clip_idx = 0
    for _ in range(rows):
        row_clips = []
        for _ in range(cols):
            if clip_idx < len(overlayed_clips):
                row_clips.append(overlayed_clips[clip_idx])
                clip_idx += 1
            else:
                padding_clip = ColorClip(
                    size=(video_width, video_height),
                    color=(255, 255, 255),
                    duration=target_duration,
                )
                row_clips.append(padding_clip)
        grid_clips.append(row_clips)

    print("Compositing video grid...")
    final_video = clips_array(grid_clips)
    print(f"Writing output to: {output_path}")
    # Use tqdm progress bar for output video encoding if proglog is available
    try:
        import proglog
        logger = proglog.TqdmProgressBarLogger(print_messages=False)
    except (ImportError, AttributeError):
        logger = "bar"  # moviepy default progress bar
    final_video.write_videofile(
        output_path,
        codec="libx264",
        audio=False,
        fps=fps,
        threads=os.cpu_count() or 1,
        logger=logger,
    )

    for clip in clips:
        clip.close()
    for clip in overlayed_clips:
        if hasattr(clip, "close"):
            clip.close()
    final_video.close()
    print(f"Grid video created successfully: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a video grid from tasks in a replay buffer"
    )
    parser.add_argument("replay_buffer", help="Path to the replay buffer zarr file")
    parser.add_argument("--rgb-key", required=True, help="Key name for RGB data in the replay buffer")
    parser.add_argument("--fps", type=float, required=True, help="Output frame rate (required)")
    parser.add_argument("-o", "--output", default="tmp_video_grid.mp4", help="Output video path")
    parser.add_argument(
        "-c", "--max-columns",
        type=int,
        default=10,
        help="Number of columns in the grid (default: 10)",
    )
    parser.add_argument(
        "--output-resolution",
        type=str,
        default=None,
        help="Output resolution as WIDTHxHEIGHT (e.g. 1920x1080)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Maximum number of tasks to include",
    )
    parser.add_argument(
        "--unique-tasks",
        action="store_true",
        default=True,
        help="Only show one instance per unique task name (default: True)",
    )
    parser.add_argument(
        "--all-demos",
        action="store_true",
        default=False,
        help="Show all demonstrations, not just one per task (overrides unique-tasks)",
    )
    parser.add_argument(
        "--in-task-name",
        type=str,
        default=None,
        help="Filter tasks whose names contain this string",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        default=False,
        help="Loop shorter videos instead of freezing on last frame",
    )
    parser.add_argument("-d", "--duration", type=float, help="Maximum duration of output video")
    parser.add_argument("-f", "--max-frames", type=int, help="Maximum number of frames to write")
    parser.add_argument(
        "--text-overlay",
        action="store_true",
        default=False,
        help="Add text overlays with task names",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of parallel workers for video extraction (default: CPU count)",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default=None,
        help="Temporary directory for intermediate videos (default: system temp)",
    )
    parser.add_argument(
        "--flip-h",
        action="store_true",
        default=False,
        help="Flip images along the height axis (for libero-style datasets stored upside down)",
    )
    args = parser.parse_args()

    unique_tasks = not args.all_demos and args.unique_tasks
    if args.all_demos:
        unique_tasks = False

    output_resolution = None
    if args.output_resolution is not None:
        parts = args.output_resolution.strip().lower().split("x")
        if len(parts) != 2:
            parser.error("--output-resolution must be WIDTHxHEIGHT (e.g. 1920x1080)")
        try:
            output_resolution = (int(parts[0]), int(parts[1]))
        except ValueError:
            parser.error("--output-resolution must be two integers")

    replay_buffer_path = os.path.abspath(os.path.expanduser(args.replay_buffer))
    if not os.path.exists(replay_buffer_path):
        print(f"Error: Replay buffer not found: {replay_buffer_path}")
        return 1

    print(f"Loading replay buffer from: {replay_buffer_path}")
    try:
        replay_buffer = ReplayBuffer.create_from_path(replay_buffer_path, mode="r")
    except Exception as e:
        print(f"Error loading replay buffer: {e}")
        return 1

    print(f"Replay buffer loaded: {replay_buffer.n_tasks} tasks, {replay_buffer.n_steps} steps")
    if args.rgb_key not in replay_buffer.data:
        print(f"Error: RGB key '{args.rgb_key}' not found. Available: {list(replay_buffer.data.keys())}")
        return 1

    task_indices = select_tasks(
        replay_buffer,
        n=args.max_tasks,
        unique_tasks=unique_tasks,
        in_task_name=args.in_task_name,
    )
    if not task_indices:
        filter_msg = f" matching '{args.in_task_name}'" if args.in_task_name else ""
        print(f"No tasks selected{filter_msg}.")
        return 1

    print(f"Selected {len(task_indices)} tasks for video grid")
    max_workers = args.max_workers if args.max_workers is not None else (os.cpu_count() or 1)
    max_workers = min(max_workers, len(task_indices))  # cap at number of tasks
    temp_dir = args.temp_dir
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="video_grid_replay_")
        cleanup_temp = True
    else:
        cleanup_temp = False

    try:
        task_names = [str(replay_buffer.get_task_name(i)) for i in task_indices]
        extracted = extract_videos_from_tasks(
            replay_buffer=replay_buffer,
            replay_buffer_path=replay_buffer_path,
            task_indices=task_indices,
            rgb_key=args.rgb_key,
            fps=args.fps,
            temp_dir=temp_dir,
            max_workers=max_workers,
            task_names_for_overlay=task_names,
            flip_h=args.flip_h,
        )
        if not extracted:
            print("No videos could be extracted from tasks.")
            return 1

        video_paths = [p for p, _ in extracted]
        overlay_labels = [label for _, label in extracted]

        create_video_grid(
            video_paths=video_paths,
            output_path=os.path.abspath(args.output),
            fps=args.fps,
            max_duration=args.duration,
            max_frames=args.max_frames,
            max_columns=args.max_columns,
            enable_text_overlay=args.text_overlay,
            overlay_labels=overlay_labels,
            loop_shorter=args.loop,
            output_resolution=output_resolution,
        )
    finally:
        if cleanup_temp and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    exit(main())
