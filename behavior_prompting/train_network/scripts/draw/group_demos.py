import multiprocessing
import os
import shutil
import argparse
import re
from typing import List, Optional, Set
import av
import numpy as np
import zarr
from tqdm import tqdm
import concurrent.futures
from behavior_prompting.common.replay_buffer import ReplayBuffer
from behavior_prompting.common.replay_buffer_util import print_replay_buffer_draw
from behavior_prompting.common.imagecodecs_numcodecs import register_codecs, JpegXl

# Import grid creation functions
from behavior_prompting.scripts.video_grid_from_videos import create_video_grid
from behavior_prompting.scripts.image_grid_from_videos import create_image_grid

def parse_task_patterns(patterns: str) -> Set[str]:
    """
    Parse task patterns and expand them to a set of task names.
    
    Examples:
    - "A,B,C" -> {"draw_A", "draw_B", "draw_C"}
    - "A,B-P,Q" -> {"draw_A", "draw_B", "draw_C", ..., "draw_P", "draw_Q"}
    - "circle,square,triangle" -> {"draw_circle", "draw_square", "draw_triangle"}
    """
    task_names = set()
    
    # Split by comma
    parts = [part.strip() for part in patterns.split(',')]
    
    for part in parts:
        if '-' in part:
            # Handle range like "B-P"
            start, end = part.split('-')
            start = start.strip()
            end = end.strip()
            
            # Check if it's a letter range
            if len(start) == 1 and len(end) == 1 and start.isalpha() and end.isalpha():
                # Preserve the case of the original letters
                start_ord = ord(start)
                end_ord = ord(end)
                
                # Ensure start_ord <= end_ord for the range
                if start_ord > end_ord:
                    start_ord, end_ord = end_ord, start_ord
                
                for i in range(start_ord, end_ord + 1):
                    letter = chr(i)
                    task_names.add(f"draw_{letter}")
            else:
                # Not a letter range, treat as literal
                task_names.add(f"draw_{start}")
                task_names.add(f"draw_{end}")
        else:
            # Single task name
            task_names.add(f"draw_{part}")
    
    return task_names


def extract_task_name_from_dataset(base_name: str) -> str:
    """
    Extract task name from dataset base name using the new naming convention.
    
    Args:
        base_name: Dataset name without .zarr suffix (e.g., 'draw_a_lower', 'draw_A_upper')
    
    Returns:
        Task name (e.g., 'draw_a', 'draw_A') or None if format doesn't match
    """
    if base_name.startswith('draw_') and (base_name.endswith('_lower') or base_name.endswith('_upper')):
        # Extract the pattern between 'draw_' and '_lower' or '_upper'
        if base_name.endswith('_lower'):
            pattern = base_name[5:-6]  # Remove 'draw_' prefix and '_lower' suffix
        else:  # ends with '_upper'
            pattern = base_name[5:-6]  # Remove 'draw_' prefix and '_upper' suffix
        return f"draw_{pattern}"
    return None


def find_matching_datasets(input_dirs: List[str], target_task_names: Set[str] = None) -> List[str]:
    """
    Find all zarr datasets in the input directories that match the target task names.
    If target_task_names is None, include all zarr files.
    Returns list of full paths to matching datasets.
    """
    matching_datasets = []
    
    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            print(f"Warning: Input directory {input_dir} does not exist")
            continue
        
        # Look for .zarr directories
        for item in os.listdir(input_dir):
            item_path = os.path.join(input_dir, item)
            if os.path.isdir(item_path) and item.endswith('.zarr'):
                # Extract task name from directory name
                base_name = item[:-5]  # Remove '.zarr' suffix
                task_name = extract_task_name_from_dataset(base_name)
                
                if task_name is not None:
                    if target_task_names is None or task_name in target_task_names:
                        matching_datasets.append(item_path)
                        print(f"Found matching dataset: {item} -> {task_name} (from {input_dir})")
    
    return matching_datasets


def load_replay_buffer(dataset_path: str) -> ReplayBuffer:
    """Load a replay buffer from a zarr dataset path."""
    replay_buffer = ReplayBuffer.create_from_path(dataset_path, mode='r')
    assert replay_buffer.n_episodes > 0, f"Replay buffer {os.path.basename(dataset_path)} has no episodes"
    return replay_buffer


def merge_replay_buffers(datasets: List[str], output_path: str, num_workers: int, image_chunk_size: int = 1, use_zip_store: bool = True):
    """
    Merge multiple replay buffers into a single zarr.zip file or zarr directory.
    """
    if not datasets:
        print("No datasets to merge")
        return
    
    print(f"\nMerging {len(datasets)} datasets into {output_path}")

    if os.path.exists(output_path):
        print(f"Output path {output_path} already exists. Deleting...")
        if use_zip_store:
            os.remove(output_path)
        else:
            shutil.rmtree(output_path)

    compressors = {
        'agent_pos': None,        # No compression for position data
        'pen_down': None,         # No compression for binary data
        'action': None,           # No compression for action data
    }
    
    # Create new replay buffer
    # TODO: probably should not do this in memory, instead write directly to disk
    out_replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())
    
    total_episodes = 0
    video_buffer_start = 0
    vid_args = []
    
    # Add episodes from each replay buffer
    for i, dataset_path in enumerate(tqdm(datasets, desc="Processing datasets", unit="dataset")):
        replay_buffer = load_replay_buffer(dataset_path)

        task_names = replay_buffer.task_names[:]
        for task_name in task_names:
            assert task_names[0] == task_name, f"Task names are not the same: {task_names[0]} != {task_name}"

        # Process episodes with progress bar
        for episode_idx in tqdm(range(replay_buffer.n_episodes), 
                               desc=f"Dataset {i+1}/{len(datasets)}: {os.path.basename(dataset_path)}", 
                               leave=False):
            # Get episode data
            episode_data = replay_buffer.get_episode(episode_idx, data_keys=['agent_pos', 'pen_down', 'action'], labels_keys=['boundary_angle'])
            
            # Add to merged buffer
            # we use compression here because we want to reduce the size of the replay buffer and don't care about compression time right now
            out_replay_buffer.add_episode(**episode_data, compressors=compressors)
            
            total_episodes += 1
        
        replay_buffer_len = len(replay_buffer.data['agent_pos'])

        vid_args.append({
            'replay_buffer': replay_buffer,
            'episode_idx': episode_idx,
            'buffer_start': video_buffer_start,
            'buffer_end': video_buffer_start + replay_buffer_len,
            'type': 'image'
        })

        vid_args.append({
            'replay_buffer': replay_buffer,
            'episode_idx': episode_idx,
            'buffer_start': video_buffer_start,
            'buffer_end': video_buffer_start + replay_buffer_len,
            'type': 'drawing_image'
        })

        video_buffer_start += replay_buffer_len
    
    print(f"Successfully added {total_episodes} episodes to replay buffer. We have not yet added videos.")

    assert video_buffer_start == out_replay_buffer.data['agent_pos'].shape[0], f"Video buffer length of {video_buffer_start} should match replay buffer length which is {out_replay_buffer.data['agent_pos'].shape[0]}"

    """ Add videos to replay buffer """
    out_res_image = (224, 224)
    out_res_drawing_image = (512, 512)
    
    print(f"{total_episodes} episodes used in total ({len(vid_args)} videos)!")

    # image arrays
    _ = out_replay_buffer.data.require_dataset(
        name='image',
        shape=(video_buffer_start,) + out_res_image + (3,),
        chunks=(image_chunk_size,) + out_res_image + (3,),
        compressor=JpegXl(level=99, numthreads=1),
        dtype=np.uint8
    )
    _ = out_replay_buffer.labels.require_dataset(
        name='drawing_image',
        shape=(video_buffer_start,) + out_res_drawing_image + (3,),
        chunks=(image_chunk_size,) + out_res_drawing_image + (3,),
        compressor=JpegXl(level=99, numthreads=1),
        dtype=np.uint8
    )

    def video_to_zarr(out_replay_buffer, vid_metadata):
        vid_type = vid_metadata['type']

        if vid_type == 'image':
            out_img_array = out_replay_buffer.data['image']
            in_img_array = vid_metadata['replay_buffer'].data['image']
        elif vid_type == 'drawing_image':
            out_img_array = out_replay_buffer.labels['drawing_image']
            in_img_array = vid_metadata['replay_buffer'].labels['drawing_image']

        buffer_start = vid_metadata['buffer_start']
        buffer_end = vid_metadata['buffer_end']

        buffer_idx = buffer_start
        for frame in tqdm(in_img_array, leave=False):
            # compress image
            out_img_array[buffer_idx] = frame
            buffer_idx += 1
        assert buffer_idx == buffer_end, f"Buffer index {buffer_idx} != {buffer_end}"
                    
    with tqdm(total=len(vid_args)) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for vid_metadata in vid_args:
                if len(futures) >= num_workers:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    pbar.update(len(completed))

                futures.add(executor.submit(video_to_zarr, 
                    out_replay_buffer, vid_metadata))

            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))
    
    # Create zip file using zarr.ZipStore
    if use_zip_store:
        with zarr.ZipStore(output_path) as zip_store:
            out_replay_buffer.save_to_store(store=zip_store)
    else:
        # If not using ZipStore, save to a directory
        out_replay_buffer.save_to_store(store=zarr.DirectoryStore(output_path))

    print_replay_buffer_draw(output_path, vis_video=False, vis_drawing_image=False)

    print(f"Successfully created {output_path}")

def group_demos(input_dirs: List[str], 
                task_patterns: Optional[str] = None, 
                output: Optional[str] = None, 
                num_workers: Optional[int] = None,
                image_chunk_size: int = 1,
                use_zip_store: bool = True,
                create_grids: bool = True,
                vis_max_videos: int = 100,
                vis_max_duration: float = 60.0):
    """
    Group multiple zarr datasets from demo_draw.py across one or more directories 
    into a single zarr.zip file or zarr directory using zarr.ZipStore or zarr.DirectoryStore.
    
    Args:
        input_dirs: List of directories containing the zarr datasets from demo_draw.py
        task_patterns: Comma-separated task patterns to include. Examples: "A,B,C" or "A,B-P,Q" or "circle,square". 
                      If not provided, all zarr files in input_dirs will be included.
        output: Output zarr.zip file path or zarr directory path. If not provided, will use input_dir + .zarr.zip for single directory,
                or require explicit path for multiple directories.
        num_workers: Number of workers to use for video processing. If -1, uses CPU count.
        image_chunk_size: Chunk size for zarr arrays. Default is 1.
        use_zip_store: Whether to use ZipStore (True) or DirectoryStore (False). Default is True.
        create_grids: Whether to create image and video grids from the generated videos. Default is True.
        vis_max_videos: Maximum number of videos to use for visualization grids (default: 100).
        vis_max_duration: Maximum duration in seconds for video grid (default: 60.0).
    
    Returns:
        str: Path to the created output file or directory
    """
    register_codecs()
    
    # Convert to absolute paths
    input_dirs = [os.path.abspath(d) for d in input_dirs]

    if num_workers is None or num_workers == -1:
        num_workers = multiprocessing.cpu_count()
    
    # Set output path: if not provided, infer from single input directory
    if output is None:
        if len(input_dirs) == 1:
            # Single directory: use its name
            input_dir_name = os.path.basename(input_dirs[0])
            assert not input_dir_name.endswith('.zarr.zip'), f"Input directory name {input_dir_name} should not end with .zarr.zip"
            assert not input_dir_name.endswith('.zarr'), f"Input directory name {input_dir_name} should not end with .zarr"
            if use_zip_store:
                output = os.path.join(os.path.dirname(input_dirs[0]), f"{input_dir_name}.zarr.zip")
            else:
                output = os.path.join(os.path.dirname(input_dirs[0]), f"{input_dir_name}.zarr")
        else:
            # Multiple directories: require explicit output path
            raise ValueError("When multiple input directories are provided, an explicit output path must be specified")
    else:
        output = os.path.abspath(output)
        if use_zip_store:
            assert output.endswith('.zarr.zip'), f"When using ZipStore, output path must end with .zarr.zip: {output}"
        else:
            assert output.endswith('.zarr'), f"When using DirectoryStore, output path must end with .zarr: {output}"
    
    # Parse task patterns
    if task_patterns is None:
        print("No task patterns provided. Including all zarr files in input directory.")
        target_task_names = None
    else:
        print(f"Parsing task patterns: {task_patterns}")
        target_task_names = parse_task_patterns(task_patterns)
        print(f"Target task names: {sorted(target_task_names)}")
    
    # Find matching datasets
    matching_datasets = find_matching_datasets(input_dirs, target_task_names)
    
    if not matching_datasets:
        print("No matching datasets found!")
        print(f"Available datasets in input directories:")
        for input_dir in input_dirs:
            if os.path.exists(input_dir):
                for item in os.listdir(input_dir):
                    if os.path.isdir(os.path.join(input_dir, item)) and item.endswith('.zarr'):
                        base_name = item[:-5]  # Remove '.zarr' suffix
                        task_name = extract_task_name_from_dataset(base_name)
                        
                        if task_name:
                            print(f"  - {item} -> {task_name} (from {input_dir})")
                        else:
                            print(f"  - {item} -> (unknown format) (from {input_dir})")
        raise ValueError("No matching datasets found")
    
    print(f"\nFound {len(matching_datasets)} matching datasets:")
    matching_datasets.sort(key=lambda x: os.path.basename(x))
    for dataset in matching_datasets:
        print(f"  - {os.path.basename(dataset)}")
    
    # Merge datasets
    merge_replay_buffers(matching_datasets, output, num_workers, image_chunk_size, use_zip_store)
    
    # Optionally create visualization grids from the generated videos
    if create_grids:
        print(f"\nCreating visualization grids from grouped demos...")
        
        # Determine the directory containing video files
        # For zip files, videos are in the input directories
        # For directory output, videos would be in the output directory (if they exist)
        video_search_dirs = input_dirs if use_zip_store else [output]
        
        # Get list of MP4 files from the search directories
        video_files = []
        for search_dir in video_search_dirs:
            if os.path.exists(search_dir):
                for f in os.listdir(search_dir):
                    if f.endswith('.mp4') and not f.endswith('video_grid.mp4'):
                        video_files.append(os.path.join(search_dir, f))
        
        if not video_files:
            print("Warning: No MP4 files found for grid creation")
        else:
            video_paths = sorted(video_files)
            print(f"Found {len(video_paths)} video files for visualization")
            
            # Determine output directory for grids (same directory as the grouped output)
            vis_output_base_path = output.replace('.zarr.zip', '') if use_zip_store else output.replace('.zarr', '')
            grid_output_path = vis_output_base_path + '_image_grid.png'
            
            # Create image grid from video last frames
            print(f"Creating image grid: {grid_output_path}")
            create_image_grid(
                video_paths=video_paths,
                output_path=grid_output_path,
                max_videos=vis_max_videos
            )
            
            # Create video grid
            grid_output_path = vis_output_base_path + '_video_grid.mp4'
            print(f"Creating video grid: {grid_output_path}")
            create_video_grid(
                video_paths=video_paths,
                output_path=grid_output_path,
                max_duration=vis_max_duration,
                max_videos=vis_max_videos
            )
            
            print(f"Successfully created visualization grids:")
            print(f"  Image grid: {grid_output_path}")
            print(f"  Video grid: {grid_output_path}")
    
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Group multiple zarr datasets from demo_draw.py across one or more directories into a single zarr.zip file using zarr.ZipStore"
    )
    parser.add_argument(
        '-i', '--input_dirs',
        nargs='+',
        help='One or more directories containing the zarr datasets from demo_draw.py'
    )
    parser.add_argument(
        '-t', '--task_patterns',
        default=None,
        help='Comma-separated task patterns to include. Examples: "A,B,C" or "A,B-P,Q" or "circle,square". If not provided, all zarr files in input_dirs will be included.'
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output zarr.zip file path (if not provided, will use input_dir + .zarr.zip)'
    )
    parser.add_argument(
        '-n', '--num_workers',
        type=int,
        default=None,
        help='Number of workers to use for video processing'
    )
    parser.add_argument(
        '-c', '--image_chunk_size',
        type=int,
        default=1,
        help='Chunk size for the image zarr arrays (default: 1)'
    )
    parser.add_argument(
        '--no-zip-store',
        action='store_true',
        default=False,
        help='Use DirectoryStore instead of ZipStore for output (default: False, uses ZipStore)'
    )
    parser.add_argument(
        '--no-create-grids',
        action='store_true',
        default=False,
        help='Disable creation of image and video grids from the generated videos (default: False, grids are created)'
    )
    parser.add_argument(
        '--vis-max-videos',
        type=int,
        default=100,
        help='Maximum number of videos to use for visualization grids (default: 100)'
    )
    parser.add_argument(
        '--vis-max-duration',
        type=float,
        default=60.0,
        help='Maximum duration in seconds for video grid (default: 60.0)'
    )
    
    args = parser.parse_args()

    # Call the core functionality
    output_path = group_demos(
        input_dirs=args.input_dirs,
        task_patterns=args.task_patterns,
        output=args.output,
        num_workers=args.num_workers,
        image_chunk_size=args.image_chunk_size,
        use_zip_store=not args.no_zip_store,
        create_grids=not args.no_create_grids,
        vis_max_videos=args.vis_max_videos,
        vis_max_duration=args.vis_max_duration
    )
    print(f"Successfully created {output_path}")


if __name__ == "__main__":
    main()
