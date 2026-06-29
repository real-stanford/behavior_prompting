#!/usr/bin/env python3
"""
Image Grid from Videos

This script takes a folder as input, finds all MP4 files in the folder,
extracts the last frame from each video, and creates an image grid that 
combines all frames into a single image file.
"""

import os
import sys
import argparse
import math
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoFileClip
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor, ToPILImage


def find_mp4_files(folder_path: str) -> List[str]:
    """Find all MP4 files in the given folder."""
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    mp4_files = list(folder.glob("*.mp4"))
    if not mp4_files:
        raise ValueError(f"No MP4 files found in folder: {folder_path}")
    
    return [str(f) for f in sorted(mp4_files)]


def calculate_nrow(num_images: int, max_columns: int = None) -> int:
    """Calculate number of rows for torchvision make_grid (which expects nrow parameter)."""
    if max_columns is not None:
        return min(max_columns, num_images)
    else:
        # Try to make it as square as possible
        return math.ceil(math.sqrt(num_images))


def extract_last_frame(video_path: str) -> np.ndarray:
    """Extract the last frame from a video file."""
    try:
        clip = VideoFileClip(video_path)
        # Get the last frame (at duration - a small epsilon to ensure we get a valid frame)
        last_frame = clip.get_frame(clip.duration - 0.01)
        clip.close()
        return last_frame
    except Exception as e:
        print(f"Warning: Could not extract frame from {video_path}: {e}")
        return None


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
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fontsize)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # Get text bounding box to calculate position
    if font:
        bbox = draw.textbbox((0, 0), text, font=font)
    else:
        bbox = draw.textbbox((0, 0), text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Calculate text position
    img_width, img_height = img_with_text.size
    if position == 'top':
        x = (img_width - text_width) // 2
        y = margin
    elif position == 'bottom':
        x = (img_width - text_width) // 2
        y = img_height - text_height - margin
    else:  # center
        x = (img_width - text_width) // 2
        y = (img_height - text_height) // 2
    
    # Draw text with stroke for better visibility (black stroke, white text)
    # Draw stroke (black outline) by drawing text in multiple positions
    stroke_width = 2
    for adj in range(-stroke_width, stroke_width + 1):
        for adj2 in range(-stroke_width, stroke_width + 1):
            if adj != 0 or adj2 != 0:
                if font:
                    draw.text((x + adj, y + adj2), text, font=font, fill='black')
                else:
                    draw.text((x + adj, y + adj2), text, fill='black')
    
    # Draw main text (white)
    if font:
        draw.text((x, y), text, font=font, fill='white')
    else:
        draw.text((x, y), text, fill='white')
    
    return img_with_text


def _process_single_grid(grid_args: Tuple[int, List[str], str, dict]) -> Tuple[int, str]:
    """
    Helper function to process a single image grid. Used for parallel processing.
    
    Args:
        grid_args: Tuple containing (grid_idx, batch_videos, output_path, kwargs)
    
    Returns:
        Tuple of (grid_idx, output_path) on success
    """
    grid_idx, batch_videos, output_path, kwargs = grid_args
    try:
        create_image_grid(
            video_paths=batch_videos,
            output_path=output_path,
            **kwargs
        )
        return grid_idx, output_path
    except Exception as e:
        print(f"Error processing grid {grid_idx + 1}: {e}")
        raise


def create_image_grids(video_paths: List[str], output_folder: str,
                      max_columns: int = None, max_videos: int = None,
                      max_grids: int = None, padding_value: float = 1.0,
                      enable_text_overlay: bool = False, max_workers: int = None) -> None:
    """
    Create multiple image grids to cover all videos, splitting them into batches.
    
    Args:
        video_paths: List of paths to MP4 files
        output_folder: Path to the output folder where grid images will be saved
        max_columns: Maximum number of columns in the grid (if None, auto-calculated)
        max_videos: Maximum number of videos per grid (if None, processes all videos in one grid)
        max_grids: Maximum number of grids to create (if None, creates as many as needed)
        padding_value: Padding value for empty grid positions (0.0-1.0, default: 1.0 for white)
        enable_text_overlay: Whether to add text overlays with video names (default: False)
        max_workers: Maximum number of parallel workers (if None, uses os.cpu_count())
    """
    if max_videos is None or max_videos <= 0:
        raise ValueError("max_videos must be specified and greater than 0 for create_image_grids")
    
    output_folder = os.path.abspath(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    total_videos = len(video_paths)
    num_grids = math.ceil(total_videos / max_videos)
    
    # Limit number of grids if max_grids is specified
    if max_grids is not None and max_grids > 0:
        num_grids = min(num_grids, max_grids)
        total_videos_to_process = num_grids * max_videos
        if total_videos_to_process < total_videos:
            print(f"Limiting to {num_grids} grid(s), processing first {total_videos_to_process} videos out of {total_videos} total videos")
    
    if max_workers is None:
        max_workers = os.cpu_count() or 1
    
    print(f"Creating {num_grids} image grid(s) to cover {min(total_videos, num_grids * max_videos)} videos (max {max_videos} videos per grid)...")
    print(f"Using {max_workers} parallel worker(s)...")
    
    # Prepare arguments for each grid
    grid_args_list = []
    for grid_idx in range(num_grids):
        start_idx = grid_idx * max_videos
        end_idx = min(start_idx + max_videos, total_videos)
        batch_videos = video_paths[start_idx:end_idx]
        
        # Create filename with index range: image_grid_1-100.png, image_grid_101-200.png, etc.
        output_filename = f"image_grid_{start_idx+1}-{end_idx}.png"
        output_path = os.path.join(output_folder, output_filename)
        
        kwargs = {
            'max_columns': max_columns,
            'max_videos': None,  # Don't limit since we've already batched
            'padding_value': padding_value,
            'enable_text_overlay': enable_text_overlay
        }
        
        grid_args_list.append((grid_idx, batch_videos, output_path, kwargs))
    
    # Process grids in parallel
    if max_workers > 1 and num_grids > 1:
        # Use parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_grid = {
                executor.submit(_process_single_grid, grid_args): grid_args[0]
                for grid_args in grid_args_list
            }
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_grid):
                grid_idx = future_to_grid[future]
                try:
                    result_grid_idx, output_path = future.result()
                    completed += 1
                    print(f"Completed grid {result_grid_idx + 1}/{num_grids} ({completed}/{num_grids}): {os.path.basename(output_path)}")
                except Exception as e:
                    print(f"Grid {grid_idx + 1} failed with error: {e}")
                    raise
    else:
        # Sequential processing (single worker or single grid)
        for grid_idx, batch_videos, output_path, kwargs in grid_args_list:
            print(f"\n--- Creating grid {grid_idx + 1}/{num_grids} (videos {grid_idx * max_videos}-{min((grid_idx + 1) * max_videos, total_videos) - 1}) ---")
            create_image_grid(
                video_paths=batch_videos,
                output_path=output_path,
                **kwargs
            )
    
    print(f"\nAll {num_grids} image grid(s) created successfully in: {output_folder}")


def create_image_grid(video_paths: List[str], output_path: str, 
                     max_columns: int = None, max_videos: int = None,
                     padding_value: float = 1.0, enable_text_overlay: bool = False) -> None:
    """
    Create an image grid from the last frames of the given video paths.
    
    Args:
        video_paths: List of paths to MP4 files
        output_path: Path for the output image
        max_columns: Maximum number of columns in the grid (if None, auto-calculated)
        max_videos: Maximum number of videos to process (if None, processes all videos)
        padding_value: Padding value for empty grid positions (0.0-1.0, default: 1.0 for white)
        enable_text_overlay: Whether to add text overlays with video names (default: False)
    """
    output_path = os.path.abspath(output_path)
    
    # Limit the number of videos if max_videos is specified
    if max_videos is not None and max_videos > 0:
        video_paths = video_paths[:max_videos]
        print(f"Processing only the first {len(video_paths)} videos (limited by max_videos={max_videos})")
    
    print(f"Extracting last frames from {len(video_paths)} videos...")
    
    # Extract last frames from all videos and convert to tensors
    frame_tensors = []
    successful_paths = []
    to_tensor = ToTensor()
    
    for path in video_paths:
        frame = extract_last_frame(path)
        if frame is not None:
            # Convert numpy array to PIL Image
            pil_frame = Image.fromarray(frame.astype(np.uint8))
            
            # Add text overlay if requested
            if enable_text_overlay:
                video_name = os.path.basename(path)
                # Remove extension for cleaner display
                video_name = os.path.splitext(video_name)[0]
                pil_frame = add_text_overlay(pil_frame, video_name)
            
            # Convert to tensor
            tensor_frame = to_tensor(pil_frame)
            frame_tensors.append(tensor_frame)
            successful_paths.append(path)
            print(f"Extracted frame from: {os.path.basename(path)}")
    
    if not frame_tensors:
        raise ValueError("No valid frames could be extracted from videos")
    
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


def main():
    parser = argparse.ArgumentParser(
        description="Create an image grid from the last frames of all MP4 files in a folder"
    )
    parser.add_argument(
        "folder",
        help="Path to the folder containing MP4 files"
    )
    parser.add_argument(
        "-o", "--output",
        default="tmp_image_grid.png",
        help="Output image file path (single grid mode) or output folder path (multiple grids mode) (default: tmp_image_grid.png)"
    )
    parser.add_argument(
        "-c", "--max-columns",
        type=int,
        help="Maximum number of columns in the grid (default: auto-calculated)"
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        help="Maximum number of videos to process (if None, processes all videos)"
    )
    parser.add_argument(
        "--max-grids",
        type=int,
        help="Maximum number of grids to create (only used with --multiple-grids, default: creates as many as needed)"
    )
    parser.add_argument(
        "--multiple-grids",
        action="store_true",
        default=False,
        help="Generate multiple image grids to cover all videos (default: False). When enabled, output must be a folder and max-videos is required."
    )
    parser.add_argument(
        "--padding-value",
        type=float,
        default=1.0,
        help="Padding value for empty grid positions (0.0-1.0, default: 1.0 for white)"
    )
    parser.add_argument(
        "--text-overlay",
        action="store_true",
        default=False,
        help="Add text overlays with video names on top of each frame (default: False)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers for processing multiple grids (default: number of CPU cores). Set to 1 for sequential processing."
    )
    
    args = parser.parse_args()
    
    # Find MP4 files
    video_paths = find_mp4_files(args.folder)
    print(f"Found {len(video_paths)} MP4 files in {args.folder}")
    
    # Validate arguments for multiple grids mode
    if args.multiple_grids:
        if args.max_videos is None or args.max_videos <= 0:
            parser.error("--max-videos is required when --multiple-grids is enabled")
        # Create image grids
        create_image_grids(
            video_paths=video_paths,
            output_folder=args.output,
            max_columns=args.max_columns,
            max_videos=args.max_videos,
            max_grids=args.max_grids,
            padding_value=args.padding_value,
            enable_text_overlay=args.text_overlay,
            max_workers=args.max_workers
        )
    else:
        # Create single image grid
        create_image_grid(
            video_paths=video_paths,
            output_path=args.output,
            max_columns=args.max_columns,
            max_videos=args.max_videos,
            padding_value=args.padding_value,
            enable_text_overlay=args.text_overlay
        )


if __name__ == "__main__":
    main()
