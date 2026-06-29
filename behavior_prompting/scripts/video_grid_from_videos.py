#!/usr/bin/env python3
"""
Video Grid Creator

This script takes a folder as input, finds all MP4 files in the folder and subdirectories,
and creates a grid video that combines all videos into a single MP4 file.
"""

import os
import argparse
import math
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from moviepy import VideoFileClip, CompositeVideoClip, clips_array, TextClip, ImageClip, concatenate_videoclips


def find_mp4_files(folder_path: str) -> List[str]:
    """Find all MP4 files in the given folder and subdirectories."""
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    mp4_files = list(folder.rglob("*.mp4"))
    if not mp4_files:
        raise ValueError(f"No MP4 files found in folder: {folder_path}")
    
    return [str(f) for f in sorted(mp4_files)]


def add_text_overlay(clip: VideoFileClip, text: str, fontsize: int = 12, 
                     position: str = 'top', margin: int = 10) -> CompositeVideoClip:
    """
    Add a text overlay to a video clip.
    
    Args:
        clip: The video clip to overlay text on
        text: The text to display
        fontsize: Font size for the text (default: 24)
        position: Position of the text ('top', 'bottom', 'center') (default: 'top')
        margin: Margin from the edge in pixels (default: 10)
    
    Returns:
        A CompositeVideoClip with the text overlay
    """
    # Calculate text position
    if position == 'top':
        text_position = ('center', margin)
    elif position == 'bottom':
        text_position = ('center', clip.h - margin - fontsize)
    else:  # center
        text_position = ('center', 'center')
    
    # Create text clip with stroke for better visibility
    txt_clip = TextClip(
        text=text,
        font_size=fontsize,
        # font='Arial-Bold',
        color='white',
        stroke_color='black',
        stroke_width=2,
        duration=clip.duration
    ).with_position(text_position)
    
    # Composite: video and text (video first, then text on top)
    return CompositeVideoClip([clip, txt_clip])


def calculate_grid_dimensions(num_videos: int, max_columns: int = None) -> Tuple[int, int]:
    """Calculate optimal grid dimensions for the given number of videos."""
    if max_columns is not None:
        # Use specified max columns
        cols = min(max_columns, num_videos)
        rows = math.ceil(num_videos / cols)
    else:
        # Try to make it as square as possible
        cols = math.ceil(math.sqrt(num_videos))
        rows = math.ceil(num_videos / cols)
    return rows, cols


def _process_single_grid(grid_args: Tuple[int, List[str], str, dict]) -> Tuple[int, str]:
    """
    Helper function to process a single video grid. Used for parallel processing.
    
    Args:
        grid_args: Tuple containing (grid_idx, batch_videos, output_path, kwargs)
    
    Returns:
        Tuple of (grid_idx, output_path) on success
    """
    grid_idx, batch_videos, output_path, kwargs = grid_args
    try:
        create_video_grid(
            video_paths=batch_videos,
            output_path=output_path,
            **kwargs
        )
        return grid_idx, output_path
    except Exception as e:
        print(f"Error processing grid {grid_idx + 1}: {e}")
        raise


def create_video_grids(video_paths: List[str], output_folder: str,
                      max_duration: float = None, max_frames: int = None,
                      max_columns: int = None, max_videos: int = None,
                      max_grids: int = None, enable_text_overlay: bool = False,
                      max_workers: int = None) -> None:
    """
    Create multiple video grids to cover all videos, splitting them into batches.
    
    Args:
        video_paths: List of paths to MP4 files
        output_folder: Path to the output folder where grid videos will be saved
        max_duration: Maximum duration of each output video (if None, uses longest input video)
        max_frames: Maximum number of frames to write (if None, writes all frames)
        max_columns: Maximum number of columns in the grid (if None, auto-calculated)
        max_videos: Maximum number of videos per grid (if None, processes all videos in one grid)
        max_grids: Maximum number of grids to create (if None, creates as many as needed)
        enable_text_overlay: Whether to add text overlays with video names (default: False)
        max_workers: Maximum number of parallel workers (if None, uses os.cpu_count())
    """
    if max_videos is None or max_videos <= 0:
        raise ValueError("max_videos must be specified and greater than 0 for create_video_grids")
    
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
    
    print(f"Creating {num_grids} video grid(s) to cover {min(total_videos, num_grids * max_videos)} videos (max {max_videos} videos per grid)...")
    print(f"Using {max_workers} parallel worker(s)...")
    
    # Prepare arguments for each grid
    grid_args_list = []
    for grid_idx in range(num_grids):
        start_idx = grid_idx * max_videos
        end_idx = min(start_idx + max_videos, total_videos)
        batch_videos = video_paths[start_idx:end_idx]
        
        # Create filename with index range: video_grid_0-9.mp4, video_grid_10-19.mp4, etc.
        output_filename = f"video_grid_{start_idx+1}-{end_idx}.mp4"
        output_path = os.path.join(output_folder, output_filename)
        
        kwargs = {
            'max_duration': max_duration,
            'max_frames': max_frames,
            'max_columns': max_columns,
            'max_videos': None,  # Don't limit since we've already batched
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
            create_video_grid(
                video_paths=batch_videos,
                output_path=output_path,
                **kwargs
            )
    
    print(f"\nAll {num_grids} video grid(s) created successfully in: {output_folder}")


def create_video_grid(video_paths: List[str], output_path: str, 
                     max_duration: float = None, max_frames: int = None,
                     max_columns: int = None, max_videos: int = None,
                     enable_text_overlay: bool = False) -> None:
    """
    Create a grid video from the given video paths.
    
    Args:
        video_paths: List of paths to MP4 files
        output_path: Path for the output grid video
        max_duration: Maximum duration of the output video (if None, uses longest input video)
        max_frames: Maximum number of frames to write (if None, writes all frames)
        max_columns: Maximum number of columns in the grid (if None, auto-calculated)
        max_videos: Maximum number of videos to process (if None, processes all videos)
        enable_text_overlay: Whether to add text overlays with video names (default: False)
    """
    output_path = os.path.abspath(output_path)
    
    # Limit the number of videos if max_videos is specified
    if max_videos is not None and max_videos > 0:
        video_paths = video_paths[:max_videos]
        print(f"Processing only the first {len(video_paths)} videos (limited by max_videos={max_videos})")
    
    print(f"Loading {len(video_paths)} videos...")
    
    # Load all video clips and track which paths succeeded
    clips = []
    successful_paths = []
    for path in video_paths:
        try:
            clip = VideoFileClip(path)
            clips.append(clip)
            successful_paths.append(path)
            print(f"Loaded: {os.path.basename(path)} ({clip.duration:.2f}s)")
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
            continue
    
    if not clips:
        raise ValueError("No valid video clips could be loaded")
    
    # If max_frames is specified, calculate duration from fps
    if max_frames is not None:
        # Get fps from the first clip
        fps = clips[0].fps
        max_duration = max_frames / fps
        print(f"Limiting to {max_frames} frames ({max_duration:.2f}s at {fps} fps)")
    
    # Determine output duration
    longest_video_duration = max(clip.duration for clip in clips)
    if max_duration is None:
        max_duration = longest_video_duration
    else:
        # Cap max_duration to the longest video duration if it exceeds it
        max_duration = min(max_duration, longest_video_duration)
    
    # If we have a max_duration, create new clips with that duration
    if max_duration is not None:
        print(f"Trimming all clips to {max_duration:.2f} seconds...")
        trimmed_clips = []
        for clip in clips:
            # Create a new clip with the limited duration
            if clip.duration > max_duration:
                # Use the first portion of the clip
                trimmed_clip = clip.with_duration(max_duration)
            else:
                # If clip is shorter than max_duration, freeze on the last frame
                if clip.duration < max_duration:
                    last_frame = clip.get_frame(max(0, clip.duration - 0.01))
                    hold_duration = max_duration - clip.duration
                    last_frame_clip = ImageClip(last_frame, duration=hold_duration)
                    trimmed_clip = concatenate_videoclips([clip, last_frame_clip])
                else:
                    trimmed_clip = clip
            trimmed_clips.append(trimmed_clip)
    else:
        trimmed_clips = clips
    
    # Add text overlays with video names if requested
    if enable_text_overlay:
        print("Adding text overlays with video names...")
        overlayed_clips = []
        for clip, path in zip(trimmed_clips, successful_paths):
            video_name = os.path.basename(path)
            # Remove extension for cleaner display
            video_name = os.path.splitext(video_name)[0]
            overlayed_clip = add_text_overlay(clip, video_name)
            overlayed_clips.append(overlayed_clip)
    else:
        overlayed_clips = trimmed_clips
    
    # Get the size of the first video to calculate positions
    first_clip = overlayed_clips[0]
    video_width = first_clip.w
    video_height = first_clip.h
    
    # Calculate grid dimensions
    rows, cols = calculate_grid_dimensions(len(overlayed_clips), max_columns)
    print(f"Creating {rows}x{cols} grid...")
    
    # Create a rectangular grid, filling empty positions with white/black
    grid_clips = []
    clip_idx = 0
    
    for i in range(rows):
        row_clips = []
        for j in range(cols):
            if clip_idx < len(overlayed_clips):
                row_clips.append(overlayed_clips[clip_idx])
                clip_idx += 1
            else:
                # Create a white/black clip for empty positions
                from moviepy.video.VideoClip import ColorClip
                # Use white color (255, 255, 255) or black (0, 0, 0)
                padding_clip = ColorClip(size=(video_width, video_height), color=(255, 255, 255), duration=max_duration)
                row_clips.append(padding_clip)
        grid_clips.append(row_clips)
    
    # Create the composite video using clips_array
    print("Compositing video grid...")
    final_video = clips_array(grid_clips)
    
    # Write the output file
    print(f"Writing output to: {output_path}")
    final_video.write_videofile(
        output_path,
        codec='libx264',
        audio_codec='aac',
        temp_audiofile='temp-audio.m4a',
        remove_temp=True,
        fps=clips[0].fps,  # Ensure consistent FPS for output
        threads=os.cpu_count()  # Use multiple threads for faster encoding
    )
    
    # Clean up
    for clip in clips:
        clip.close()
    for clip in overlayed_clips:
        if hasattr(clip, 'close'):
            clip.close()
    final_video.close()
    
    print(f"Grid video created successfully: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a grid video from all MP4 files in a folder"
    )
    parser.add_argument(
        "folder",
        help="Path to the folder containing MP4 files (searches recursively in subdirectories)"
    )
    parser.add_argument(
        "-o", "--output",
        default="tmp_video_grid.mp4",
        help="Output file path (single grid mode) or output folder path (multiple grids mode) (default: tmp_video_grid.mp4)"
    )
    parser.add_argument(
        "-d", "--duration",
        type=float,
        help="Maximum duration of output video (default: longest input video)"
    )
    # TODO: right now if duration is specified it will do that many seconds even if the video would have ended sooner
    parser.add_argument(
        "-f", "--max-frames",
        type=int,
        help="Maximum number of frames to write (overrides duration if specified)"
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
        help="Generate multiple video grids to cover all videos (default: False). When enabled, output must be a folder and max-videos is required."
    )
    parser.add_argument(
        "--text-overlay",
        action="store_true",
        default=False,
        help="Add text overlays with video names on top of each video (default: False)"
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
        # Create video grids
        create_video_grids(
            video_paths=video_paths,
            output_folder=args.output,
            max_duration=args.duration,
            max_frames=args.max_frames,
            max_columns=args.max_columns,
            max_videos=args.max_videos,
            max_grids=args.max_grids,
            enable_text_overlay=args.text_overlay,
            max_workers=args.max_workers
        )
    else:
        # Create single video grid
        create_video_grid(
            video_paths=video_paths,
            output_path=args.output,
            max_duration=args.duration,
            max_frames=args.max_frames,
            max_columns=args.max_columns,
            max_videos=args.max_videos,
            enable_text_overlay=args.text_overlay
        )


if __name__ == "__main__":
    main()
