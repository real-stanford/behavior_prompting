#!/usr/bin/env python3
"""
Replay Buffer to Image Grid

This script takes a replay buffer as input, extracts RGB frames from tasks
at a specified proportion through each task, overlays task names, and creates
an image grid combining all frames into a single image file.
"""

import os
import argparse
import math
from typing import List, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm

from behavior_prompting.common.replay_buffer import ReplayBuffer
from behavior_prompting.common.imagecodecs_numcodecs import register_codecs
register_codecs()

def calculate_nrow(num_images: int, max_columns: int = None) -> int:
    """Calculate number of rows for torchvision make_grid (which expects nrow parameter)."""
    if max_columns is not None:
        return min(max_columns, num_images)
    else:
        # Try to make it as square as possible
        return math.ceil(math.sqrt(num_images))


def add_text_overlay(image: Image.Image, text: str, fontsize: int = 12, 
                     position: str = 'top', margin: int = 10) -> Image.Image:
    """
    Add a text overlay to an image.
    
    Args:
        image: The PIL Image to overlay text on
        text: The text to display
        fontsize: Font size for the text (default: 12)
        position: Position of the text ('top', 'bottom', 'center') (default: 'top')
        margin: Margin from the edge in pixels (default: 10)
    
    Returns:
        A PIL Image with the text overlay
    """
    # Create a copy to avoid modifying the original
    img_with_text = image.copy()
    draw = ImageDraw.Draw(img_with_text)
    
    img_width, img_height = img_with_text.size
    max_text_width = img_width - 2 * margin
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fontsize)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # Check if text fits, if not truncate with ellipsis or reduce font size
    if font:
        bbox = draw.textbbox((0, 0), text, font=font)
    else:
        bbox = draw.textbbox((0, 0), text)
    text_width = bbox[2] - bbox[0]
    
    # If text is too wide, try reducing font size first
    current_fontsize = fontsize
    current_font = font
    while text_width > max_text_width and current_fontsize > 8:
        current_fontsize = max(8, int(current_fontsize * 0.9))
        try:
            current_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", current_fontsize)
        except:
            try:
                current_font = ImageFont.load_default()
            except:
                current_font = None
        if current_font:
            bbox = draw.textbbox((0, 0), text, font=current_font)
        else:
            bbox = draw.textbbox((0, 0), text)
        text_width = bbox[2] - bbox[0]
    
    # If still too wide after font reduction, truncate with ellipsis
    if text_width > max_text_width:
        ellipsis = "..."
        if current_font:
            ellipsis_width = draw.textbbox((0, 0), ellipsis, font=current_font)[2] - draw.textbbox((0, 0), ellipsis, font=current_font)[0]
        else:
            ellipsis_width = draw.textbbox((0, 0), ellipsis)[2] - draw.textbbox((0, 0), ellipsis)[0]
        
        available_width = max_text_width - ellipsis_width
        truncated_text = text
        while True:
            if current_font:
                test_bbox = draw.textbbox((0, 0), truncated_text, font=current_font)
            else:
                test_bbox = draw.textbbox((0, 0), truncated_text)
            test_width = test_bbox[2] - test_bbox[0]
            if test_width <= available_width or len(truncated_text) <= 1:
                break
            truncated_text = truncated_text[:-1]
        text = truncated_text + ellipsis
        if current_font:
            bbox = draw.textbbox((0, 0), text, font=current_font)
        else:
            bbox = draw.textbbox((0, 0), text)
        text_width = bbox[2] - bbox[0]
    
    text_height = bbox[3] - bbox[1]
    
    # Calculate text position
    if position == 'top':
        x = (img_width - text_width) // 2
        y = margin
    elif position == 'bottom':
        x = (img_width - text_width) // 2
        y = img_height - text_height - margin
    else:  # center
        x = (img_width - text_width) // 2
        y = (img_height - text_height) // 2
    
    # Ensure text doesn't go outside image bounds
    x = max(0, min(x, img_width - text_width))
    y = max(0, min(y, img_height - text_height))
    
    # Draw text with stroke for better visibility (black stroke, white text)
    # Draw stroke (black outline) by drawing text in multiple positions
    stroke_width = 2
    for adj in range(-stroke_width, stroke_width + 1):
        for adj2 in range(-stroke_width, stroke_width + 1):
            if adj != 0 or adj2 != 0:
                if current_font:
                    draw.text((x + adj, y + adj2), text, font=current_font, fill='black')
                else:
                    draw.text((x + adj, y + adj2), text, fill='black')
    
    # Draw main text (white)
    if current_font:
        draw.text((x, y), text, font=current_font, fill='white')
    else:
        draw.text((x, y), text, fill='white')
    
    return img_with_text


def select_tasks(replay_buffer: ReplayBuffer, n: int, unique_tasks: bool, 
                in_task_name: Optional[str] = None) -> List[int]:
    """
    Select which task indices to include in the image grid.
    
    Args:
        replay_buffer: The replay buffer to select tasks from
        n: Maximum number of tasks to select
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
        if len(selected_indices) >= n:
            break
        
        task_name = replay_buffer.get_task_name(task_idx)
        
        # Filter by task name substring if specified
        if in_task_name is not None:
            if in_task_name not in task_name:
                continue
        
        if unique_tasks:
            if task_name not in seen_task_names:
                selected_indices.append(task_idx)
                seen_task_names.add(task_name)
        else:
            selected_indices.append(task_idx)
    
    return selected_indices


def extract_frame_from_task(replay_buffer: ReplayBuffer, task_idx: int, 
                            rgb_key: str, proportion: float) -> Optional[np.ndarray]:
    """
    Extract RGB frame at specified proportion through a task.
    
    Args:
        replay_buffer: The replay buffer containing the task
        task_idx: Index of the task
        rgb_key: Key name for RGB data in the replay buffer
        proportion: Proportion through the task (0.0 to 1.0)
    
    Returns:
        RGB frame as numpy array (H, W, C) in uint8 format, or None if extraction fails
    """
    try:
        # Get task data
        task_data = replay_buffer.get_task(task_idx, copy=True)
        
        # Check if RGB key exists
        if rgb_key not in task_data['data']:
            print(f"Warning: RGB key '{rgb_key}' not found in task {task_idx}")
            return None
        
        rgb_data = task_data['data'][rgb_key]
        
        # Handle empty or invalid task
        if len(rgb_data) == 0:
            print(f"Warning: Task {task_idx} has no RGB data")
            return None
        
        # Calculate frame index
        task_length = len(rgb_data)
        frame_idx = int((task_length - 1) * proportion)
        frame_idx = max(0, min(frame_idx, task_length - 1))  # Clamp to valid range
        
        # Extract frame
        frame = rgb_data[frame_idx]
        
        # Ensure frame is in correct format (H, W, C) and uint8
        if isinstance(frame, np.ndarray):
            # Handle different channel orders
            if len(frame.shape) == 3:
                # Check if it's (C, H, W) and convert to (H, W, C)
                if frame.shape[0] < frame.shape[2] and frame.shape[0] <= 4:
                    frame = np.transpose(frame, (1, 2, 0))
            elif len(frame.shape) == 2:
                # Grayscale, add channel dimension
                frame = np.expand_dims(frame, axis=-1)
                # Convert to RGB by repeating channels
                frame = np.repeat(frame, 3, axis=-1)
            
            # Ensure uint8 dtype
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    # Assume normalized [0, 1] and convert to [0, 255]
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
            
            return frame
        else:
            print(f"Warning: Frame from task {task_idx} is not a numpy array")
            return None
            
    except Exception as e:
        print(f"Error extracting frame from task {task_idx}: {e}")
        return None


def create_image_grid_from_tasks(replay_buffer: ReplayBuffer, task_indices: List[int],
                                rgb_key: str, proportion: float, output_path: str,
                                max_columns: int = None, padding_value: float = 1.0) -> None:
    """
    Create an image grid from selected tasks in the replay buffer.
    
    Args:
        replay_buffer: The replay buffer containing tasks
        task_indices: List of task indices to include
        rgb_key: Key name for RGB data
        proportion: Proportion through each task to extract frame
        output_path: Path to save the output image
        max_columns: Maximum number of columns in the grid (None for auto)
        padding_value: Padding value for empty grid positions (0.0-1.0)
    """
    if len(task_indices) == 0:
        raise ValueError("No tasks selected for image grid")
    
    print(f"Extracting frames from {len(task_indices)} tasks...")
    
    # Extract frames and convert to PIL Images with text overlays
    frame_tensors = []
    to_tensor = ToTensor()
    successful_tasks = []
    
    for task_idx in tqdm(task_indices, desc="Extracting frames", unit="task"):
        # Extract frame
        frame = extract_frame_from_task(replay_buffer, task_idx, rgb_key, proportion)
        
        if frame is not None:
            # Convert numpy array to PIL Image
            pil_frame = Image.fromarray(frame)
            
            # Get task name and add text overlay
            task_name = replay_buffer.get_task_name(task_idx)
            pil_frame = add_text_overlay(pil_frame, task_name, fontsize=16, position='top')
            
            # Convert to tensor
            tensor_frame = to_tensor(pil_frame)
            frame_tensors.append(tensor_frame)
            successful_tasks.append(task_idx)
        else:
            tqdm.write(f"Failed to extract frame from task {task_idx}")
    
    if not frame_tensors:
        raise ValueError("No valid frames could be extracted from tasks")
    
    # Stack all tensors
    frames_batch = torch.stack(frame_tensors)
    
    # Calculate nrow (number of images per row)
    nrow = calculate_nrow(len(frame_tensors), max_columns)
    print(f"Creating image grid with {nrow} columns...")
    
    # Create the grid using torchvision
    grid_tensor = make_grid(frames_batch, nrow=nrow, padding=2, pad_value=padding_value)
    
    # Convert back to PIL Image and save
    to_pil = ToPILImage()
    grid_image = to_pil(grid_tensor)
    
    print(f"Saving image grid to: {output_path}")
    grid_image.save(output_path, quality=95, optimize=True)
    
    print(f"Image grid created successfully: {output_path}")
    print(f"Grid dimensions: {grid_image.size[0]}x{grid_image.size[1]} pixels")
    print(f"Successfully processed {len(successful_tasks)}/{len(task_indices)} tasks")


def main():
    parser = argparse.ArgumentParser(
        description="Create an image grid from RGB frames extracted from tasks in a replay buffer"
    )
    parser.add_argument(
        "replay_buffer",
        help="Path to the replay buffer zarr file"
    )
    parser.add_argument(
        "--rgb-key",
        required=True,
        help="Key name for RGB data in the replay buffer"
    )
    parser.add_argument(
        "--proportion-in-task",
        type=float,
        default=0.9,
        help="Proportion through each task to extract frame (0.0-1.0, default: 0.9)"
    )
    parser.add_argument(
        "-n",
        type=int,
        default=10,
        help="Number of tasks to include in the grid (default: 10)"
    )
    parser.add_argument(
        "--unique-tasks",
        action="store_true",
        default=False,
        help="If set, only include one instance per unique task name"
    )
    parser.add_argument(
        "--in-task-name",
        type=str,
        default=None,
        help="Filter tasks to only include those whose names contain this string"
    )
    parser.add_argument(
        "-o", "--output",
        default="tmp_replay_buffer_to_image_grid",
        help="Output folder path (default: tmp_replay_buffer_to_image_grid)"
    )
    parser.add_argument(
        "--max-columns",
        type=int,
        help="Maximum number of columns in the grid (default: auto-calculated)"
    )
    parser.add_argument(
        "--padding-value",
        type=float,
        default=1.0,
        help="Padding value for empty grid positions (0.0-1.0, default: 1.0 for white)"
    )
    
    args = parser.parse_args()
    
    # Validate proportion
    if not 0.0 <= args.proportion_in_task <= 1.0:
        parser.error("--proportion-in-task must be between 0.0 and 1.0")
    
    # Validate n
    if args.n <= 0:
        parser.error("-n must be greater than 0")
    
    # Load replay buffer
    print(f"Loading replay buffer from: {args.replay_buffer}")
    try:
        replay_buffer = ReplayBuffer.create_from_path(args.replay_buffer, mode='r')
    except Exception as e:
        print(f"Error loading replay buffer: {e}")
        return 1
    
    print(f"Replay buffer loaded: {replay_buffer.n_tasks} tasks, {replay_buffer.n_steps} steps")
    
    # Check if RGB key exists
    if args.rgb_key not in replay_buffer.data:
        print(f"Error: RGB key '{args.rgb_key}' not found in replay buffer")
        print(f"Available keys: {list(replay_buffer.data.keys())}")
        return 1
    
    # Select tasks
    task_indices = select_tasks(replay_buffer, args.n, args.unique_tasks, args.in_task_name)
    
    if len(task_indices) == 0:
        filter_msg = f" matching filter '{args.in_task_name}'" if args.in_task_name else ""
        print(f"No tasks selected{filter_msg}. Replay buffer may be empty or no matching tasks found.")
        return 1
    
    filter_msg = f" (filtered by '{args.in_task_name}')" if args.in_task_name else ""
    print(f"Selected {len(task_indices)} tasks for image grid{filter_msg}")
    
    # Create output directory
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "image_grid.png")
    
    # Create image grid
    try:
        create_image_grid_from_tasks(
            replay_buffer=replay_buffer,
            task_indices=task_indices,
            rgb_key=args.rgb_key,
            proportion=args.proportion_in_task,
            output_path=output_path,
            max_columns=args.max_columns,
            padding_value=args.padding_value
        )
    except Exception as e:
        print(f"Error creating image grid: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
