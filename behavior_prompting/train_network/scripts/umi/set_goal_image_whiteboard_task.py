'''
For each task in an umi replay buffer, set the meta entry called 'goal_img_frame_idx' to the frame index of the goal image.
The goal image is the first frame encountered where SAM detects 4 red dots while iterating through the task backwards from the end.
For greater computational efficiency, user can set the batchsize and can only run sam on every nth frame. 
Create or overwrite the meta entry called 'goal_img_frame_idx' in the replay buffer.
For debugging, user can set the --debug flag to view the first <number> goal images.
The user can also pass a list of task names to only process the specified tasks and leave the other indexes unchanged.
'''

import argparse
import os
import numpy as np
from collections import defaultdict
from behavior_prompting.common.replay_buffer import ReplayBuffer
from behavior_prompting.common.imagecodecs_numcodecs import register_codecs
import math
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from tqdm import tqdm

# Register codecs for image decompression
register_codecs(verbose=False)


def _resolve_camera_key(replay_buffer):
    """Return the preferred camera key for reading images."""
    for key in ['camera_right_main_rgb', 'camera_left_main_rgb', 'camera_right_ultrawide_rgb', 'camera_left_ultrawide_rgb', 'camera0_main_rgb', 'camera0_ultrawide_rgb', 'image']:
        if key in replay_buffer.data:
            return key
    return None


def _build_candidates(replay_buffer, tasks_to_process, every_n_frames):
    """Build list of (task_idx, frame_idx) in backwards order per task.
    For each task, candidate frames are end_frame-1, end_frame-1-every_n, ... until start_frame.
    """
    candidates = []
    for task_idx in tasks_to_process:
        end_frame = int(replay_buffer.task_data_ends[task_idx])
        task_length = int(replay_buffer.task_lengths[task_idx])
        start_frame = end_frame - task_length
        # backwards from end: end_frame-1, end_frame-1-every_n, ...
        frames = list(range(end_frame - 1, start_frame - 1, -every_n_frames))
        for frame_idx in frames:
            candidates.append((task_idx, frame_idx))
    return candidates


def _load_image_at_frame(replay_buffer, camera_key, frame_idx):
    """Load a single image at frame_idx, respecting upsampling for the camera key."""
    if replay_buffer.is_key_upsampled(camera_key):
        storage_idx = replay_buffer.map_upsample_index(camera_key, frame_idx)
        img = replay_buffer[camera_key][storage_idx]
    else:
        img = replay_buffer[camera_key][frame_idx]
    return np.asarray(img)


def set_goal_image(
    replay_buffer_path: str,
    debug: int = None,
    every_n_frames: int = 1,
    batch_size: int = 32,
    task_names: list = None,
    min_red_dots: int = 4,
):
    """
    Set 'goal_img_frame_idx' meta entry per task to the first frame (scanning backwards from end)
    where the detector finds at least min_red_dots red dots.

    Args:
        replay_buffer_path: Path to the zarr replay buffer
        debug: Optional number of goal images to visualize (saved as goal_images_grid.png)
        every_n_frames: Run detector only on every Nth frame when scanning backwards (default: 1)
        batch_size: Batch size for DINO model inference (default: 32)
        task_names: If provided, only update indices for these task names; others unchanged
        min_red_dots: Minimum red dots required for a frame to be the goal (default: 4)
    """
    if not os.path.exists(replay_buffer_path):
        raise FileNotFoundError(f"Replay buffer file not found: {replay_buffer_path}")

    print(f"Loading replay buffer from: {replay_buffer_path}")
    replay_buffer = ReplayBuffer.create_from_path(replay_buffer_path, mode='r+')

    n_tasks = replay_buffer.n_tasks
    if n_tasks == 0:
        print("No tasks found in replay buffer. Nothing to do.")
        return

    print(f"Found {n_tasks} tasks")

    camera_key = _resolve_camera_key(replay_buffer)
    if camera_key is None:
        raise KeyError(
            "No known camera key (camera0_main_rgb, camera0_ultrawide_rgb, image) found. "
            f"Available keys: {list(replay_buffer.data.keys())}"
        )
    print(f"Using camera key: {camera_key}")

    # Task filter
    if task_names is not None and len(task_names) > 0:
        allowed = set(task_names)
        tasks_to_process = [i for i in range(n_tasks) if replay_buffer.task_names[i] in allowed]
        print(f"Processing only task names {allowed}: {len(tasks_to_process)} task(s)")
    else:
        tasks_to_process = list(range(n_tasks))

    if len(tasks_to_process) == 0:
        print("No tasks to process. Nothing to do.")
        return

    # Initialize goal_img_frame_indices
    meta_group = replay_buffer.meta
    if 'goal_img_frame_idx' in meta_group:
        goal_img_frame_indices = np.array(meta_group['goal_img_frame_idx'][:], dtype=np.int64)
        assert goal_img_frame_indices.shape[0] == n_tasks
    else:
        goal_img_frame_indices = np.array(
            [int(replay_buffer.task_data_ends[i]) - 1 for i in range(n_tasks)],
            dtype=np.int64,
        )

    # Build candidate (task_idx, frame_idx) list
    candidates = _build_candidates(replay_buffer, tasks_to_process, every_n_frames)
    if not candidates:
        print("No candidate frames. Writing meta and exiting.")
    else:
        print(f"Running detector on {len(candidates)} candidate frames (every_n_frames={every_n_frames}, min_red_dots={min_red_dots})...")

        model_id = "IDEA-Research/grounding-dino-tiny"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Grounding DINO on device: {device}")
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

        box_threshold = 0.35
        text_threshold = 0.25

        # Load all candidate images and convert to PIL
        goal_images_pil = []
        for (task_idx, frame_idx) in tqdm(candidates, desc="Loading candidate images"):
            try:
                arr = _load_image_at_frame(replay_buffer, camera_key, frame_idx)
                if arr.dtype != np.uint8:
                    if arr.max() <= 1.0:
                        arr = (arr * 255).astype(np.uint8)
                    else:
                        arr = arr.astype(np.uint8)
                pil_image = Image.fromarray(arr)
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")
                goal_images_pil.append(pil_image)
            except Exception as e:
                print(f"  Warning: Could not load image for task {task_idx} frame {frame_idx}: {e}")
                goal_images_pil.append(None)  # placeholder; we'll skip in inference or use -1

        # Batch by image size and run DINO
        dim_to_indices = defaultdict(list)
        for idx, img in enumerate(goal_images_pil):
            if img is not None:
                dim_to_indices[img.size].append(idx)
            # else: skip in results by not adding to dim_to_indices; we'll treat as 0 dots

        red_dot_counts = [0] * len(candidates)  # default 0 for failed loads
        pbar = tqdm(total=len([i for i, img in enumerate(goal_images_pil) if img is not None]), desc="Detecting red dots")

        for dim, indices in dim_to_indices.items():
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start : batch_start + batch_size]
                batch_images = [goal_images_pil[i] for i in batch_indices]
                text_prompts = ["red dot."] * len(batch_images)
                try:
                    inputs = processor(images=batch_images, text=text_prompts, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    target_sizes = [img.size[::-1] for img in batch_images]
                    try:
                        results = processor.post_process_grounded_object_detection(
                            outputs, inputs.input_ids, threshold=box_threshold, target_sizes=target_sizes
                        )
                    except TypeError:
                        try:
                            results = processor.post_process_grounded_object_detection(
                                outputs,
                                inputs.input_ids,
                                box_threshold=box_threshold,
                                text_threshold=text_threshold,
                                target_sizes=target_sizes,
                            )
                        except TypeError:
                            results = processor.post_process_grounded_object_detection(
                                outputs, inputs.input_ids, target_sizes=target_sizes
                            )
                    for i, result in enumerate(results):
                        orig_idx = batch_indices[i]
                        num_red_dots = 0
                        for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                            score_val = score.cpu().item() if hasattr(score, "cpu") else float(score)
                            if score_val >= box_threshold:
                                num_red_dots += 1
                        red_dot_counts[orig_idx] = num_red_dots
                        pbar.update(1)
                except Exception as e:
                    print(f"  Warning: Error processing batch: {e}")
                    for orig_idx in batch_indices:
                        red_dot_counts[orig_idx] = 0
                        pbar.update(1)
        pbar.close()

        # Per task: first (from end) frame with count >= min_red_dots
        # candidates are (task_idx, frame_idx); we walk backwards per task
        # Build per-task list of (frame_idx, count), then take max frame_idx with count >= min_red_dots
        task_to_best_frame = {}
        for (task_idx, frame_idx), count in zip(candidates, red_dot_counts):
            if count >= min_red_dots:
                if task_idx not in task_to_best_frame or frame_idx > task_to_best_frame[task_idx]:
                    task_to_best_frame[task_idx] = frame_idx

        for task_idx in tasks_to_process:
            if task_idx in task_to_best_frame:
                goal_img_frame_indices[task_idx] = task_to_best_frame[task_idx]
            else:
                goal_img_frame_indices[task_idx] = -1

    # Fallback: if all tasks with same task_name have -1, use last frame
    print("\nChecking for task names with all indices set to -1...")
    task_name_to_indices = defaultdict(list)
    for task_idx in range(n_tasks):
        task_name_to_indices[replay_buffer.task_names[task_idx]].append(task_idx)

    num_fallback_tasks = 0
    for task_name, indices_list in task_name_to_indices.items():
        if all(goal_img_frame_indices[idx] == -1 for idx in indices_list):
            for task_idx in indices_list:
                goal_img_frame_indices[task_idx] = int(replay_buffer.task_data_ends[task_idx]) - 1
                num_fallback_tasks += 1
            print(f"  Task name '{task_name}': All {len(indices_list)} task(s) had invalid indices. Falling back to last frame.")
    if num_fallback_tasks > 0:
        print(f"Applied fallback to {num_fallback_tasks} task(s).")
    else:
        print("No task names had all indices set to -1. No fallback needed.")

    # Write meta
    if 'goal_img_frame_idx' in meta_group:
        existing_array = meta_group['goal_img_frame_idx']
        if existing_array.compressor is not None:
            print("Warning: 'goal_img_frame_idx' already exists. Overwriting (removing compressor)...")
            del meta_group['goal_img_frame_idx']
            meta_group.zeros(
                name='goal_img_frame_idx',
                shape=goal_img_frame_indices.shape,
                dtype=np.int64,
                compressor=None,
                overwrite=False,
            )
            meta_group['goal_img_frame_idx'][:] = goal_img_frame_indices
        else:
            if existing_array.shape != goal_img_frame_indices.shape:
                existing_array.resize(goal_img_frame_indices.shape)
            existing_array[:] = goal_img_frame_indices
    else:
        meta_group.zeros(
            name='goal_img_frame_idx',
            shape=goal_img_frame_indices.shape,
            dtype=np.int64,
            compressor=None,
            overwrite=False,
        )
        meta_group['goal_img_frame_idx'][:] = goal_img_frame_indices

    print("Successfully wrote 'goal_img_frame_idx' to meta group.")

    # Debug visualization
    if debug is not None and debug > 0:
        num_images = min(debug, n_tasks)
        print(f"\nDebug: Visualizing first {num_images} goal images in a grid...")
        vis_camera_key = _resolve_camera_key(replay_buffer)
        if vis_camera_key is None:
            print("Warning: No known camera key. Skipping visualization.")
        else:
            goal_images = []
            for task_idx in range(num_images):
                goal_frame_idx = goal_img_frame_indices[task_idx]
                try:
                    arr = _load_image_at_frame(replay_buffer, vis_camera_key, goal_frame_idx)
                    if arr.ndim == 3 and arr.shape[-1] == 3:
                        rgb = arr.copy()
                    else:
                        rgb = arr.copy()
                    pil_image = Image.fromarray(rgb) if rgb.dtype == np.uint8 else Image.fromarray((rgb * 255).astype(np.uint8))
                    if pil_image.mode != "RGB":
                        pil_image = pil_image.convert("RGB")
                    goal_images.append(pil_image)
                except Exception as e:
                    print(f"  Warning: Could not load goal image for task {task_idx}: {e}")
            if goal_images:
                n_cols = math.ceil(math.sqrt(len(goal_images)))
                n_rows = math.ceil(len(goal_images) / n_cols)
                w, h = goal_images[0].size
                grid_image = Image.new("RGB", (n_cols * w, n_rows * h), color="black")
                for idx, img in enumerate(goal_images):
                    row, col = idx // n_cols, idx % n_cols
                    grid_image.paste(img, (col * w, row * h))
                output_path = os.path.join(os.path.dirname(replay_buffer_path), "goal_images_grid.png")
                grid_image.save(output_path)
                print(f"\nGrid image saved to: {output_path}")
                print(f"  Grid size: {n_rows}x{n_cols} ({len(goal_images)} images)")
            else:
                print("No images were successfully loaded.")


def main():
    parser = argparse.ArgumentParser(
        description="Set 'goal_img_frame_idx' per task to the first frame (backwards from end) "
                    "where the detector finds at least min_red_dots red dots."
    )
    parser.add_argument("--replay_buffer_path", type=str, required=True, help="Path to the zarr replay buffer")
    parser.add_argument(
        "--debug",
        type=int,
        default=None,
        help="Optional: Number of goal images to visualize (saved as goal_images_grid.png)",
    )
    parser.add_argument(
        "--every_n_frames",
        type=int,
        default=1,
        help="Run detector on every Nth frame when scanning backwards (default: 1)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for DINO model inference (default: 16)",
    )
    parser.add_argument(
        "--task_names",
        type=str,
        nargs="*",
        default=None,
        help="If provided, only update goal_img_frame_idx for these task names",
    )
    parser.add_argument(
        "--min_red_dots",
        type=int,
        default=4,
        help="Minimum red dots required for a frame to be the goal (default: 4)",
    )
    args = parser.parse_args()
    set_goal_image(
        args.replay_buffer_path,
        debug=args.debug,
        every_n_frames=args.every_n_frames,
        batch_size=args.batch_size,
        task_names=args.task_names,
        min_red_dots=args.min_red_dots,
    )
    print("\nScript completed successfully!")


if __name__ == "__main__":
    main()
