"""
Concatenate replay buffers into a new replay buffer.

This script reads episodes from multiple source replay buffers and concatenates
them into a new replay buffer. The source buffers are not modified (read-only).
"""

import os
import sys
import argparse
import numpy as np
import shutil
import zarr
from zarr.storage import ZipStore, DirectoryStore
from tqdm import tqdm
import concurrent.futures

from behavior_prompting.common.replay_buffer import ReplayBuffer
from behavior_prompting.common.replay_buffer import get_optimal_chunks, rechunk_recompress_array
from behavior_prompting.common.replay_buffer_util import print_replay_buffer_summary
from behavior_prompting.common.imagecodecs_numcodecs import register_codecs
register_codecs()


def load_replay_buffer(path, load_inputs_in_memory=False):
    """Load replay buffer from path in read-only mode."""
    if load_inputs_in_memory:
        replay_buffer = ReplayBuffer.copy_from_path(path)
    else:
        replay_buffer = ReplayBuffer.create_from_path(path, mode='r')
    return replay_buffer


def remove_labels_from_tasks(tasks):
    """Remove labels from all tasks in the list."""
    for task in tasks:
        task["labels"] = {}
    return tasks


def append_suffix_to_tasks(tasks, suffix):
    """Append suffix to task names in the list."""
    if not suffix:
        return tasks
    updated_tasks = []
    for task in tasks:
        task_copy = dict(task)
        task_copy["name"] = f"{task_copy['name']}{suffix}"
        updated_tasks.append(task_copy)
    return updated_tasks


def resolve_chunks_and_compressors(source_buffers, buffer_paths, use_buffer_index=None):
    """Resolve chunks and compressors from all source buffers.
    
    Inspects all input buffers to determine chunk sizes and compressors for each
    entry in data, labels, and meta groups. Errors if any buffers have conflicting
    choices for the same entry (unless use_buffer_index is specified).
    
    Args:
        source_buffers: List of ReplayBuffer objects
        buffer_paths: List of paths to source buffers (for error messages)
        use_buffer_index: Optional 0-indexed buffer index to use as source for chunks/compressors.
                         If specified, conflicts are ignored and this buffer's settings are used.
    
    Returns:
        tuple: (resolved_chunks, resolved_compressors) where each is a dict
            with keys like 'data/key', 'labels/key', 'meta/key'
    
    Raises:
        ValueError: If any buffers have conflicting chunks or compressors (unless use_buffer_index is set)
        IndexError: If use_buffer_index is out of range
    """
    import numcodecs
    
    # If use_buffer_index is specified, only use that buffer
    if use_buffer_index is not None:
        if use_buffer_index < 0 or use_buffer_index >= len(source_buffers):
            raise IndexError(
                f"Buffer index {use_buffer_index} is out of range. "
                f"Valid indices are 0-{len(source_buffers)-1} (1-{len(source_buffers)} when 1-indexed)."
            )
        
        # Only collect from the specified buffer
        source_buffer = source_buffers[use_buffer_index]
        buffer_path = buffer_paths[use_buffer_index]
        
        resolved_chunks = {}
        resolved_compressors = {}
        
        # Only process zarr buffers (numpy buffers don't have chunks/compressors)
        if source_buffer.backend == 'zarr':
            # Collect from data group
            for key, arr in source_buffer.data.items():
                if isinstance(arr, zarr.Array):
                    resolved_chunks[f'data/{key}'] = arr.chunks
                    resolved_compressors[f'data/{key}'] = arr.compressor
            
            # Collect from labels group
            for key, arr in source_buffer.labels.items():
                if isinstance(arr, zarr.Array):
                    resolved_chunks[f'labels/{key}'] = arr.chunks
                    resolved_compressors[f'labels/{key}'] = arr.compressor
            
            # Collect from meta group
            for key, arr in source_buffer.meta.items():
                if isinstance(arr, zarr.Array):
                    resolved_chunks[f'meta/{key}'] = arr.chunks
                    resolved_compressors[f'meta/{key}'] = arr.compressor
        
        return resolved_chunks, resolved_compressors
    
    # Structure: {group_key: {buffer_idx: (chunks, compressor)}}
    all_settings = {}
    
    # Helper to get chunks and compressor from an array
    def get_array_settings(arr, buffer_idx, group_name, entry_key):
        """Get chunks and compressor from an array, handling both zarr and numpy."""
        if isinstance(arr, zarr.Array):
            chunks = arr.chunks
            compressor = arr.compressor
        elif isinstance(arr, np.ndarray):
            # Numpy arrays don't have chunks/compressor - skip them
            return None
        else:
            # Unknown type - skip
            return None
        
        group_key = f"{group_name}/{entry_key}"
        if group_key not in all_settings:
            all_settings[group_key] = {}
        
        all_settings[group_key][buffer_idx] = (chunks, compressor)
        return (chunks, compressor)
    
    # Collect settings from all buffers
    for buffer_idx, (buffer, buffer_path) in enumerate(zip(source_buffers, buffer_paths)):
        # Only process zarr buffers (numpy buffers don't have chunks/compressors)
        if buffer.backend != 'zarr':
            continue
        
        # Collect from data group
        for key, arr in buffer.data.items():
            get_array_settings(arr, buffer_idx, 'data', key)
        
        # Collect from labels group
        for key, arr in buffer.labels.items():
            get_array_settings(arr, buffer_idx, 'labels', key)
        
        # Collect from meta group
        for key, arr in buffer.meta.items():
            get_array_settings(arr, buffer_idx, 'meta', key)
    
    # Check for conflicts and resolve
    resolved_chunks = {}
    resolved_compressors = {}
    conflicts = []
    
    for group_key, buffer_settings in all_settings.items():
        if len(buffer_settings) == 0:
            continue
        
        # Get all unique (chunks, compressor) pairs
        unique_settings = {}
        for buffer_idx, (chunks, compressor) in buffer_settings.items():
            # Create a comparable representation of compressor
            # For numcodecs compressors, compare codec_id and config
            if compressor is None:
                comp_key = None
            elif hasattr(compressor, 'codec_id') and hasattr(compressor, 'config'):
                comp_key = (compressor.codec_id, tuple(sorted(compressor.config.items())))
            else:
                # Fallback: use string representation
                comp_key = str(compressor)
            
            setting_key = (chunks, comp_key)
            if setting_key not in unique_settings:
                unique_settings[setting_key] = []
            unique_settings[setting_key].append(buffer_idx)
        
        # Check for conflicts
        if len(unique_settings) > 1:
            # Conflict found
            conflict_details = []
            for (chunks, comp_key), buffer_indices in unique_settings.items():
                buffer_names = [buffer_paths[idx] for idx in buffer_indices]
                if comp_key is None:
                    comp_str = "None (uncompressed)"
                elif isinstance(comp_key, tuple) and len(comp_key) == 2:
                    comp_str = f"{comp_key[0]} with config {comp_key[1]}"
                else:
                    comp_str = str(comp_key)
                conflict_details.append(
                    f"  Buffers {buffer_indices} ({', '.join(buffer_names)}): "
                    f"chunks={chunks}, compressor={comp_str}"
                )
            conflicts.append(
                f"Conflict for {group_key}:\n" + "\n".join(conflict_details)
            )
        else:
            # No conflict - use the single setting
            (chunks, comp_key), buffer_indices = next(iter(unique_settings.items()))
            # Get the actual compressor object from first buffer
            first_buffer_idx = buffer_indices[0]
            _, actual_compressor = buffer_settings[first_buffer_idx]
            
            resolved_chunks[group_key] = chunks
            resolved_compressors[group_key] = actual_compressor
    
    # Raise error if conflicts found
    if conflicts:
        error_msg = "Conflicting chunks or compressors found across input buffers:\n\n"
        error_msg += "\n\n".join(conflicts)
        raise ValueError(error_msg)
    
    return resolved_chunks, resolved_compressors


def process_episodes_from_buffer(source_buffer, target_buffer, max_episodes=None, ignore_labels=False, buffer_name="", allowed_task_names=None, task_suffix=None, skip_task_names=None, rgb_keys=None, resolved_chunks=None, resolved_compressors=None, overall_progress=None):
    """Process episodes from source buffer and add them to target buffer.
    
    Args:
        source_buffer: Source replay buffer
        target_buffer: Target replay buffer to add episodes to
        max_episodes: Maximum number of episodes to process (None for all)
        ignore_labels: Whether to remove labels from tasks
        buffer_name: Name of the buffer for logging
        allowed_task_names: List of task names to include (None for all tasks)
        task_suffix: Suffix to append to task names (None for no change)
        skip_task_names: List of task names to skip (None for no skipping). Task names should already include suffixes.
        rgb_keys: List of RGB keys to exclude during episode addition (None for no exclusion)
        resolved_chunks: Dict of resolved chunks with keys like 'data/key', 'labels/key', 'meta/key'
        resolved_compressors: Dict of resolved compressors with keys like 'data/key', 'labels/key', 'meta/key'
        overall_progress: Optional tqdm progress bar to update for overall progress
    
    Returns:
        tuple: (episodes_added, episode_mappings) where episode_mappings is a list of dicts with mapping info
    """
    n_episodes = source_buffer.n_episodes
    
    if n_episodes == 0:
        print(f"Warning: {buffer_name} has no episodes, skipping")
        return 0, []
    
    # Determine which keys to exclude when getting episodes
    rgb_keys_set = set(rgb_keys) if rgb_keys is not None else set()
    all_data_keys = set(source_buffer.data.keys())
    non_rgb_keys = list(all_data_keys - rgb_keys_set) if rgb_keys_set else None
    
    # Efficiently find episode indices that contain allowed task names
    episode_indices_to_process = None
    if allowed_task_names is not None:
        print(f"  Filtering for task names: {allowed_task_names}")
        # Get all task names
        task_names = source_buffer.task_names[:]
        # Find task indices that match allowed names
        matching_task_indices = np.where(np.isin(task_names, allowed_task_names))[0]
        
        if len(matching_task_indices) == 0:
            print(f"  No matching tasks found, skipping all episodes")
            return 0, []
        
        # Get episode indices for matching tasks
        task_to_episode = source_buffer.get_task_to_episode_idxs()
        matching_episode_indices = task_to_episode[matching_task_indices]
        # Get unique episode indices
        episode_indices_to_process = np.unique(matching_episode_indices).tolist()
        print(f"  Found {len(episode_indices_to_process)} episode(s) with matching tasks")
    else:
        # Process all episodes
        episode_indices_to_process = list(range(n_episodes))
    
    # Apply max_episodes limit
    if max_episodes is not None:
        episode_indices_to_process = episode_indices_to_process[:max_episodes]
    
    num_to_process = len(episode_indices_to_process)
    print(f"Processing {num_to_process} episode(s) from {buffer_name}...")
    
    episodes_added = 0
    episode_mappings = []
    
    # Get source buffer episode boundaries for tracking
    source_episode_ends = source_buffer.episode_ends[:]
    
    for i, episode_idx in enumerate(episode_indices_to_process):
        print(f"  Processing episode {i + 1}/{num_to_process} (episode index {episode_idx})...")
        
        # Get source episode boundaries
        source_start_idx = 0 if episode_idx == 0 else source_episode_ends[episode_idx - 1]
        source_end_idx = source_episode_ends[episode_idx]
        
        # Get episode data (excluding RGB keys if specified)
        if rgb_keys_set:
            episode = source_buffer.get_episode(episode_idx, copy=True, data_keys=non_rgb_keys)
        else:
            episode = source_buffer.get_episode(episode_idx, copy=True)
        
        # Extract episode components
        data = episode['data']
        tasks = episode['tasks']
        episode_name = episode['episode_name']
        upsample_indexing_values = episode['upsample_indexing_values']
        upsample_indexing_lengths = episode['upsample_indexing_lengths']
        downsample_indexing_values = episode['downsample_indexing_values']
        
        # Remove labels if requested
        if ignore_labels:
            tasks = remove_labels_from_tasks(tasks)
        
        # Append suffix to task names if requested
        if task_suffix:
            tasks = append_suffix_to_tasks(tasks, task_suffix)

        # Skip episode if it contains any task in skip_task_names
        if skip_task_names is not None:
            episode_task_names = [task["name"] for task in tasks]
            if any(task_name in skip_task_names for task_name in episode_task_names):
                print(f"  Skipping episode (contains task in skip list)")
                if overall_progress is not None:
                    overall_progress.update(1)
                continue

        # Get target episode index (before adding, so it's the current n_episodes)
        target_episode_idx = target_buffer.n_episodes
        target_start_idx = 0 if target_episode_idx == 0 else target_buffer.episode_ends[target_episode_idx - 1]

        # Prepare chunks and compressors for add_episode
        # Convert from 'data/key' and 'labels/key' format to just 'key'
        # Note: add_episode uses the same dict for both data and labels, keyed by entry name
        episode_chunks = {}
        episode_compressors = {}
        
        if resolved_chunks is not None:
            for full_key, chunks_val in resolved_chunks.items():
                if full_key.startswith('data/'):
                    key = full_key[5:]  # Remove 'data/' prefix
                    if key not in rgb_keys_set:  # Only include non-RGB keys
                        episode_chunks[key] = chunks_val
                elif full_key.startswith('labels/'):
                    key = full_key[7:]  # Remove 'labels/' prefix
                    episode_chunks[key] = chunks_val  # Labels use same dict, keyed by label name
        
        if resolved_compressors is not None:
            for full_key, compressor_val in resolved_compressors.items():
                if full_key.startswith('data/'):
                    key = full_key[5:]  # Remove 'data/' prefix
                    if key not in rgb_keys_set:  # Only include non-RGB keys
                        episode_compressors[key] = compressor_val
                elif full_key.startswith('labels/'):
                    key = full_key[7:]  # Remove 'labels/' prefix
                    episode_compressors[key] = compressor_val  # Labels use same dict, keyed by label name

        # Add episode to target buffer (without RGB keys if specified)
        target_buffer.add_episode(
            data=data,
            tasks=tasks,
            episode_name=episode_name,
            upsample_indexing_values=upsample_indexing_values,
            upsample_indexing_lengths=upsample_indexing_lengths,
            downsample_indexing_values=downsample_indexing_values,
            chunks=episode_chunks if episode_chunks else None,
            compressors=episode_compressors if episode_compressors else None
        )
        
        # Get target episode end after adding
        target_end_idx = target_buffer.episode_ends[target_episode_idx]
        
        # Track episode mapping for RGB copying (only if RGB keys are specified)
        if rgb_keys_set:
            # Calculate upsampled ranges for each RGB key
            upsampled_ranges = {}
            for rgb_key in rgb_keys_set:
                if source_buffer.is_key_upsampled(rgb_key):
                    # Get source upsampled episode boundaries
                    episode_ends_key = f'episode_ends_{rgb_key}'
                    source_upsampled_ends = source_buffer.meta[episode_ends_key][:]
                    source_upsampled_start = 0 if episode_idx == 0 else source_upsampled_ends[episode_idx - 1]
                    source_upsampled_end = source_upsampled_ends[episode_idx]
                    
                    # Get target upsampled episode boundaries
                    target_upsampled_ends = target_buffer.meta[episode_ends_key][:]
                    target_upsampled_start = 0 if target_episode_idx == 0 else target_upsampled_ends[target_episode_idx - 1]
                    target_upsampled_end = target_upsampled_ends[target_episode_idx]
                    
                    upsampled_ranges[rgb_key] = {
                        'source_start': int(source_upsampled_start),
                        'source_end': int(source_upsampled_end),
                        'target_start': int(target_upsampled_start),
                        'target_end': int(target_upsampled_end)
                    }
            
            episode_mapping = {
                'source_buffer': source_buffer,
                'source_episode_idx': episode_idx,
                'target_episode_idx': target_episode_idx,
                'source_start_idx': int(source_start_idx),
                'source_end_idx': int(source_end_idx),
                'target_start_idx': int(target_start_idx),
                'target_end_idx': int(target_end_idx),
                'upsampled_ranges': upsampled_ranges
            }
            episode_mappings.append(episode_mapping)
        
        episodes_added += 1
        if overall_progress is not None:
            overall_progress.update(1)
    
    print(f"  Added {episodes_added} episode(s)")
    return episodes_added, episode_mappings


def copy_rgb_worker(mapping, rgb_key, source_buffer, target_buffer):
    """Worker function to copy RGB data from source to target buffer.
    
    Args:
        mapping: Episode mapping dict with source/target indices
        rgb_key: RGB key to copy
        source_buffer: Source replay buffer
        target_buffer: Target replay buffer
    """
    source_arr = source_buffer.data[rgb_key]
    target_arr = target_buffer.data[rgb_key]
    
    # Check if upsampled
    if source_buffer.is_key_upsampled(rgb_key):
        # Use upsampled ranges from mapping
        ranges = mapping['upsampled_ranges'][rgb_key]
        source_slice = source_arr[ranges['source_start']:ranges['source_end']]
        target_arr[ranges['target_start']:ranges['target_end']] = source_slice
    else:
        # Use regular ranges
        source_slice = source_arr[mapping['source_start_idx']:mapping['source_end_idx']]
        target_arr[mapping['target_start_idx']:mapping['target_end_idx']] = source_slice


def copy_rgb_data_parallel(episode_mappings, rgb_keys, target_buffer, resolved_chunks=None, resolved_compressors=None, num_workers=None):
    """Copy RGB data from source buffers to target buffer in parallel.
    
    Args:
        episode_mappings: List of episode mapping dicts
        rgb_keys: List of RGB keys to copy
        target_buffer: Target replay buffer
        resolved_chunks: Dict of resolved chunks for RGB keys
        resolved_compressors: Dict of resolved compressors for RGB keys
        num_workers: Number of worker threads (None for default)
    """
    if num_workers is None:
        import multiprocessing
        num_workers = multiprocessing.cpu_count()
    
    if len(episode_mappings) == 0 or len(rgb_keys) == 0:
        return
    
    # Validate chunk sizes BEFORE creating arrays (required for thread-safe parallel copying)
    # Get a source buffer to get array shapes and properties
    source_buffer = episode_mappings[0]['source_buffer']
    
    print(f"\nValidating chunk sizes for thread-safe parallel copying...")
    for rgb_key in rgb_keys:
        # Check resolved_chunks if it exists (this is what we'll use when creating arrays)
        chunks_key = f'data/{rgb_key}'
        if resolved_chunks is not None and chunks_key in resolved_chunks:
            resolved_chunk_size = resolved_chunks[chunks_key][0]
            assert resolved_chunk_size == 1, (
                f"Resolved chunks for '{rgb_key}' has chunk size {resolved_chunk_size} in time dimension, "
                f"but parallel copying requires chunk size 1 for thread safety. "
                f"Please rechunk the source buffers or use sequential copying (omit --rgb-keys)."
            )
        else:
            # If resolved_chunks doesn't have this key, we'll use source array chunks as fallback
            # Check source array chunk size in that case
            source_arr = source_buffer.data[rgb_key]
            if isinstance(source_arr, zarr.Array):
                source_chunk_size = source_arr.chunks[0]
                assert source_chunk_size == 1, (
                    f"Source array '{rgb_key}' has chunk size {source_chunk_size} in time dimension, "
                    f"and will be used as fallback for target array creation. "
                    f"Parallel copying requires chunk size 1 for thread safety. "
                    f"Please rechunk the source buffer or use sequential copying (omit --rgb-keys)."
                )
        
        # Check target buffer if array already exists
        if rgb_key in target_buffer.data:
            target_arr = target_buffer.data[rgb_key]
            if isinstance(target_arr, zarr.Array):
                target_chunk_size = target_arr.chunks[0]
                assert target_chunk_size == 1, (
                    f"Target array '{rgb_key}' has chunk size {target_chunk_size} in time dimension, "
                    f"but parallel copying requires chunk size 1 for thread safety. "
                    f"Please rechunk the target buffer or use sequential copying (omit --rgb-keys)."
                )
    
    # Now create arrays (we know chunk sizes are valid)
    print(f"\nEnsuring RGB arrays exist in target buffer...")
    for rgb_key in rgb_keys:
        if rgb_key not in target_buffer.data:
            # Get shape and dtype from source buffer
            source_arr = source_buffer.data[rgb_key]
            image_shape = source_arr.shape[1:]  # (H, W, C) or similar
            dtype = source_arr.dtype
            
            # Calculate total length needed in target buffer
            if source_buffer.is_key_upsampled(rgb_key):
                # For upsampled keys, use the episode_ends_{key} metadata
                episode_ends_key = f'episode_ends_{rgb_key}'
                if episode_ends_key in target_buffer.meta:
                    total_length = int(target_buffer.meta[episode_ends_key][-1]) if len(target_buffer.meta[episode_ends_key]) > 0 else 0
                else:
                    # Fallback: calculate from episode mappings
                    total_length = 0
                    for mapping in episode_mappings:
                        if rgb_key in mapping['upsampled_ranges']:
                            total_length = max(total_length, mapping['upsampled_ranges'][rgb_key]['target_end'])
            else:
                # For regular keys, use episode_ends
                total_length = int(target_buffer.episode_ends[-1]) if len(target_buffer.episode_ends) > 0 else 0
            
            # Get chunks and compressor for this key
            chunks = None
            compressor = None
            if resolved_chunks is not None:
                chunks_key = f'data/{rgb_key}'
                if chunks_key in resolved_chunks:
                    chunks = resolved_chunks[chunks_key]
            if resolved_compressors is not None:
                compressor_key = f'data/{rgb_key}'
                if compressor_key in resolved_compressors:
                    compressor = resolved_compressors[compressor_key]
            
            # Use source array's chunks/compressor as fallback
            if chunks is None:
                if isinstance(source_arr, zarr.Array):
                    chunks = source_arr.chunks
            if compressor is None:
                if isinstance(source_arr, zarr.Array):
                    compressor = source_arr.compressor
            
            # Create the array in target buffer
            target_buffer.data.require_dataset(
                name=rgb_key,
                shape=(total_length,) + image_shape,
                chunks=chunks,
                compressor=compressor,
                dtype=dtype
            )
            print(f"  Created array for {rgb_key} with shape {(total_length,) + image_shape}")
    
    # Create list of (mapping, rgb_key) tuples for all copy operations
    copy_tasks = []
    for mapping in episode_mappings:
        for rgb_key in rgb_keys:
            copy_tasks.append((mapping, rgb_key))
    
    print(f"\nCopying {len(copy_tasks)} RGB data segments in parallel using {num_workers} workers...")
    
    with tqdm(total=len(copy_tasks), desc="Copying RGB data", unit="segment") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for mapping, rgb_key in copy_tasks:
                if len(futures) >= num_workers:
                    # Limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    pbar.update(len(completed))
                
                futures.add(executor.submit(copy_rgb_worker, 
                    mapping, rgb_key, mapping['source_buffer'], target_buffer))
            
            # Wait for remaining tasks
            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))
    
    print(f"Completed copying RGB data for {len(episode_mappings)} episodes")


def main():
    parser = argparse.ArgumentParser(
        description='Concatenate replay buffers into a new replay buffer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Concatenate all episodes from three buffers
  python concat_replay_buffers.py \\
    --buffer buffer1.zarr --buffer buffer2.zarr --buffer buffer3.zarr \\
    -o output.zarr

  # Limit episodes for only the last buffer
  python concat_replay_buffers.py \\
    --buffer buffer1.zarr --buffer buffer2.zarr \\
    --buffer buffer3.zarr --max-episodes 10 \\
    -o output.zarr

  # Filter by task names for specific buffers
  python concat_replay_buffers.py \\
    --buffer buffer1.zarr --tasks task1 task2 \\
    --buffer buffer2.zarr \\
    --buffer buffer3.zarr --tasks task3 \\
    -o output.zarr

  # Append a task name suffix for a specific buffer
  python concat_replay_buffers.py \\
    --buffer buffer1.zarr --task-suffix _set1 \\
    --buffer buffer2.zarr \\
    -o output.zarr

  # Provide global defaults for any buffers without per-buffer overrides
  python concat_replay_buffers.py \\
    --default-max-episodes 5 --default-tasks taskA taskB \\
    --buffer buffer1.zarr --buffer buffer2.zarr \\
    -o output.zarr

  # Use chunks and compressors from a specific buffer
  python concat_replay_buffers.py \\
    --buffer buffer1.zarr \\
    --buffer buffer2.zarr --chunks-and-compressors-from-buffer \\
    --buffer buffer3.zarr \\
    -o output.zarr

  # Use chunks and compressors from a reference buffer (no episodes added from it)
  python concat_replay_buffers.py \\
    --buffer buffer1.zarr \\
    --buffer buffer2.zarr \\
    --reference-chunks-and-compressors-from-buffer reference.zarr \\
    -o output.zarr
        """
    )

    parser.add_argument('--output', '-o', type=str, required=True, 
                       help='Output path for concatenated replay buffer (.zarr or .zarr.zip)')
    parser.add_argument('--ignore-labels', '-i', action='store_true',
                       help='Remove labels from episodes before adding to output buffer')
    parser.add_argument('--keep-inputs-in-memory', action='store_true',
                       help='Load source buffers into memory (faster but uses more memory)')
    parser.add_argument('--default-max-episodes', type=int, default=None,
                       help='Default max episodes for buffers without per-buffer override (default: all)')
    parser.add_argument('--default-tasks', type=str, nargs='+', default=None,
                       help='Default task filter for buffers without per-buffer override (default: all tasks)')
    parser.add_argument('--skip-task-names', type=str, nargs='+', default=None,
                       help='Task names to skip (episodes containing these tasks will not be added). Task names should include any suffixes that will be applied.')
    parser.add_argument('--rgb-keys', type=str, nargs='+', default=None,
                       help='RGB data keys to exclude during episode addition and copy in parallel afterward. If not specified, all keys are included during episode addition (original behavior).')
    parser.add_argument('--num-workers', type=int, default=None,
                       help='Number of worker threads for parallel RGB copying. Defaults to number of CPU cores if not specified.')
    parser.add_argument('--rechunk-sample-meta', action='store_true',
                       help='Rechunk meta attributes for up/downsample indices using optimal chunk size')
    parser.add_argument('--reference-chunks-and-compressors-from-buffer', type=str, default=None,
                       help='Path to a replay buffer to use ONLY for chunks and compressors (no episodes will be added from this buffer)')
    
    args, remaining_args = parser.parse_known_args()

    def parse_buffer_args(args_list):
        buffers = []
        i = 0
        while i < len(args_list):
            arg = args_list[i]
            if arg == '--buffer':
                if i + 1 >= len(args_list):
                    parser.error("--buffer requires a path argument")
                buffer_path = args_list[i + 1]
                buffers.append({
                    "path": buffer_path,
                    "max_episodes": None,
                    "tasks": None,
                    "task_suffix": None,
                    "use_for_chunks": False
                })
                i += 2
                continue
            if arg in ('--max-episodes', '-m'):
                if not buffers:
                    parser.error("--max-episodes must follow a --buffer")
                if i + 1 >= len(args_list):
                    parser.error("--max-episodes requires an integer argument")
                try:
                    max_eps = int(args_list[i + 1])
                except ValueError:
                    parser.error(f"Invalid --max-episodes value: {args_list[i + 1]}")
                buffers[-1]["max_episodes"] = max_eps
                i += 2
                continue
            if arg in ('--tasks', '-t'):
                if not buffers:
                    parser.error("--tasks must follow a --buffer")
                i += 1
                tasks = []
                while i < len(args_list) and not args_list[i].startswith('-'):
                    tasks.append(args_list[i])
                    i += 1
                if not tasks:
                    parser.error("--tasks requires at least one task name")
                buffers[-1]["tasks"] = tasks
                continue
            if arg == '--task-suffix':
                if not buffers:
                    parser.error("--task-suffix must follow a --buffer")
                if i + 1 >= len(args_list):
                    parser.error("--task-suffix requires a string argument")
                buffers[-1]["task_suffix"] = args_list[i + 1]
                i += 2
                continue
            if arg == '--chunks-and-compressors-from-buffer':
                if not buffers:
                    parser.error("--chunks-and-compressors-from-buffer must follow a --buffer")
                buffers[-1]["use_for_chunks"] = True
                i += 1
                continue
            parser.error(f"Unrecognized argument: {arg}")
        return buffers

    buffer_configs = parse_buffer_args(remaining_args)
    
    if len(buffer_configs) == 0:
        parser.error("At least one --buffer is required")

    for idx, buffer_cfg in enumerate(buffer_configs, start=1):
        if not os.path.exists(buffer_cfg["path"]):
            raise FileNotFoundError(f"Buffer {idx} not found: {buffer_cfg['path']}")
        if buffer_cfg["max_episodes"] is None:
            buffer_cfg["max_episodes"] = args.default_max_episodes
        if buffer_cfg["tasks"] is None:
            buffer_cfg["tasks"] = args.default_tasks
    
    # Validate output path
    if not (args.output.endswith('.zarr') or args.output.endswith('.zarr.zip')):
        raise ValueError(f"Output path must end with .zarr or .zarr.zip, got: {args.output}")
    
    # Determine if we need to create a zipstore and the working path
    output_is_zip = args.output.endswith('.zarr.zip')
    if output_is_zip:
        # Remove .zip extension for the working path
        working_output_path = args.output[:-4]  # Remove '.zip'
    else:
        working_output_path = args.output
    
    # Check if output path already exists
    if os.path.exists(args.output):
        response = input(f"Output path {args.output} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        # If overwriting, remove existing output
        if os.path.isdir(args.output):
            shutil.rmtree(args.output)
        else:
            os.remove(args.output)
    
    # Also check if working path exists (in case we're creating a zip from existing dir)
    if output_is_zip and os.path.exists(working_output_path):
        response = input(f"Working path {working_output_path} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        if os.path.isdir(working_output_path):
            shutil.rmtree(working_output_path)
        else:
            os.remove(working_output_path)
    
    print("=" * 60)
    print("Concatenating Replay Buffers")
    print("=" * 60)
    for idx, buffer_cfg in enumerate(buffer_configs, start=1):
        print(f"Buffer {idx}: {buffer_cfg['path']}")
    print(f"Output: {args.output}")
    if output_is_zip:
        print(f"Working path: {working_output_path}")
    for idx, buffer_cfg in enumerate(buffer_configs, start=1):
        if buffer_cfg["max_episodes"] is not None:
            print(f"Max episodes from buffer {idx}: {buffer_cfg['max_episodes']}")
        if buffer_cfg["tasks"] is not None:
            print(f"Task filter for buffer {idx}: {buffer_cfg['tasks']}")
        if buffer_cfg["task_suffix"] is not None:
            print(f"Task suffix for buffer {idx}: {buffer_cfg['task_suffix']}")
    if args.ignore_labels:
        print("Labels will be removed from episodes")
    if args.skip_task_names is not None:
        print(f"Skipping episodes containing task names: {args.skip_task_names}")
    print("=" * 60)
    
    # Load source buffers
    print("\nLoading source buffers...")
    source_buffers = []
    for buffer_cfg in buffer_configs:
        source_buffers.append(load_replay_buffer(buffer_cfg["path"], load_inputs_in_memory=args.keep_inputs_in_memory))

    for idx, source_buffer in enumerate(source_buffers, start=1):
        print(f"Buffer {idx}: {source_buffer.n_episodes} episode(s)")

    # Validate rgb-keys exist in all source buffers
    if args.rgb_keys is not None:
        rgb_keys_set = set(args.rgb_keys)
        for idx, source_buffer in enumerate(source_buffers, start=1):
            available_keys = set(source_buffer.data.keys())
            missing_keys = rgb_keys_set - available_keys
            if missing_keys:
                raise ValueError(
                    f"Buffer {idx} is missing the following RGB keys specified in --rgb-keys: {sorted(missing_keys)}. "
                    f"Available keys: {sorted(available_keys)}"
                )
        print(f"\nRGB keys to be copied in parallel: {args.rgb_keys}")

    # Print task counts for each buffer and total (after applying suffixes, excluding skipped tasks)
    print("\nTask counts:")
    skip_task_names_set = set(args.skip_task_names) if args.skip_task_names is not None else set()
    all_task_names = set()
    all_task_names_with_skipped = set()  # For validation - includes skipped tasks
    for idx, (source_buffer, buffer_cfg) in enumerate(zip(source_buffers, buffer_configs), start=1):
        task_names = source_buffer.task_names[:]
        # Apply task suffix if specified
        task_suffix = buffer_cfg.get("task_suffix")
        if task_suffix:
            task_names = [f"{name}{task_suffix}" for name in task_names]
        unique_tasks = set(task_names)
        # Collect all task names (including skipped) for validation
        all_task_names_with_skipped.update(unique_tasks)
        # Exclude skipped tasks from count
        unique_tasks = unique_tasks - skip_task_names_set
        task_count = len(unique_tasks)
        print(f"  Buffer {idx}: {task_count} unique task(s)")
        all_task_names.update(unique_tasks)
    total_unique_tasks = len(all_task_names)
    print(f"  Total unique tasks across all buffers: {total_unique_tasks}")

    # Validate skip-task-names exist
    if args.skip_task_names is not None:
        skip_task_names_set = set(args.skip_task_names)
        missing_tasks = skip_task_names_set - all_task_names_with_skipped
        if missing_tasks:
            raise ValueError(
                f"The following task names specified in --skip-task-names were not found in any buffer "
                f"(after applying suffixes): {sorted(missing_tasks)}"
            )

    # Validate buffers are not empty
    if all(source_buffer.n_episodes == 0 for source_buffer in source_buffers):
        raise ValueError("All source buffers are empty")
    
    # Resolve chunks and compressors from all input buffers
    print("\nResolving chunks and compressors from input buffers...")
    buffer_paths = [cfg["path"] for cfg in buffer_configs]
    
    # Handle --reference-chunks-and-compressors-from-buffer flag (separate buffer for chunks/compressors only)
    reference_buffer = None
    reference_buffer_path = None
    if args.reference_chunks_and_compressors_from_buffer:
        reference_buffer_path = args.reference_chunks_and_compressors_from_buffer
        if not os.path.exists(reference_buffer_path):
            raise FileNotFoundError(f"Reference buffer not found: {reference_buffer_path}")
        print(f"  Loading reference buffer for chunks/compressors: {reference_buffer_path}")
        reference_buffer = load_replay_buffer(reference_buffer_path, load_inputs_in_memory=args.keep_inputs_in_memory)
        print(f"  Reference buffer has {reference_buffer.n_episodes} episode(s) (will not be added to output)")
    
    # Handle --chunks-and-compressors-from-buffer flag (find which buffer has it set)
    use_buffer_index = None
    chunks_from_buffers = [i for i, cfg in enumerate(buffer_configs) if cfg.get("use_for_chunks", False)]
    
    if reference_buffer is not None and len(chunks_from_buffers) > 0:
        raise ValueError(
            "Cannot use both --reference-chunks-and-compressors-from-buffer and "
            "--chunks-and-compressors-from-buffer. Use only one."
        )
    
    if len(chunks_from_buffers) > 1:
        raise ValueError(
            f"Multiple buffers have --chunks-and-compressors-from-buffer specified. "
            f"Only one buffer can be used as the source for chunks and compressors."
        )
    elif len(chunks_from_buffers) == 1:
        use_buffer_index = chunks_from_buffers[0]
        print(f"  Using chunks and compressors from buffer {use_buffer_index + 1} ({buffer_paths[use_buffer_index]})")
    
    try:
        if reference_buffer is not None:
            # Extract chunks and compressors from the reference buffer only
            resolved_chunks, resolved_compressors = resolve_chunks_and_compressors(
                [reference_buffer], [reference_buffer_path], use_buffer_index=0
            )
            print(f"  Extracted chunks and compressors for {len(resolved_chunks)} entries from reference buffer")
        else:
            # Resolve from source buffers as before
            resolved_chunks, resolved_compressors = resolve_chunks_and_compressors(
                source_buffers, buffer_paths, use_buffer_index=use_buffer_index
            )
            if use_buffer_index is not None:
                print(f"  Extracted chunks and compressors for {len(resolved_chunks)} entries from buffer {use_buffer_index + 1}")
            else:
                print(f"  Resolved chunks and compressors for {len(resolved_chunks)} entries")
    except (ValueError, IndexError) as e:
        print(f"Error during chunk/compressor resolution:")
        print(str(e))
        raise
    
    # Create new empty replay buffer on disk (no in-memory buffering)
    print("\nCreating new replay buffer...")
    new_buffer = ReplayBuffer.create_from_path(
        os.path.expanduser(working_output_path),
        mode='w'  # create/overwrite on disk
    )
    
    # Calculate total episodes to process (upper bound, before skip filtering)
    total_episodes_to_process = 0
    for source_buffer, buffer_cfg in zip(source_buffers, buffer_configs):
        n_episodes = source_buffer.n_episodes
        if n_episodes == 0:
            continue
        
        # Calculate episodes considering allowed_task_names filter
        if buffer_cfg["tasks"] is not None:
            task_names = source_buffer.task_names[:]
            matching_task_indices = np.where(np.isin(task_names, buffer_cfg["tasks"]))[0]
            if len(matching_task_indices) == 0:
                continue
            task_to_episode = source_buffer.get_task_to_episode_idxs()
            matching_episode_indices = task_to_episode[matching_task_indices]
            episode_count = len(np.unique(matching_episode_indices))
        else:
            episode_count = n_episodes
        
        # Apply max_episodes limit
        if buffer_cfg["max_episodes"] is not None:
            episode_count = min(episode_count, buffer_cfg["max_episodes"])
        
        total_episodes_to_process += episode_count
    
    # Create overall progress bar
    overall_progress = tqdm(
        total=total_episodes_to_process,
        desc="Overall progress",
        unit="episode",
        position=0,
        leave=True
    )
    
    buffer_episode_counts = []
    all_episode_mappings = []
    try:
        for idx, (source_buffer, buffer_cfg) in enumerate(zip(source_buffers, buffer_configs), start=1):
            print("\n" + "=" * 60)
            num_episodes, episode_mappings = process_episodes_from_buffer(
                source_buffer,
                new_buffer,
                max_episodes=buffer_cfg["max_episodes"],
                ignore_labels=args.ignore_labels,
                buffer_name=f"Buffer {idx}",
                allowed_task_names=buffer_cfg["tasks"],
                task_suffix=buffer_cfg["task_suffix"],
                skip_task_names=args.skip_task_names,
                rgb_keys=args.rgb_keys,
                resolved_chunks=resolved_chunks,
                resolved_compressors=resolved_compressors,
                overall_progress=overall_progress
            )
            buffer_episode_counts.append(num_episodes)
            all_episode_mappings.extend(episode_mappings)
    finally:
        overall_progress.close()
    
    # Copy RGB data in parallel if rgb_keys were specified
    if args.rgb_keys and len(all_episode_mappings) > 0:
        print("\n" + "=" * 60)
        copy_rgb_data_parallel(all_episode_mappings, args.rgb_keys, new_buffer, 
                              resolved_chunks=resolved_chunks, 
                              resolved_compressors=resolved_compressors,
                              num_workers=args.num_workers)
    
    # Optionally rechunk sample-related meta attributes for better performance
    if args.rechunk_sample_meta and new_buffer.backend == 'zarr':
        print("\n" + "=" * 60)
        print("Rechunking sample-related meta attributes using optimal chunks...")
        meta_group = new_buffer.meta
        for key, arr in meta_group.items():
            if not isinstance(arr, zarr.Array):
                continue
            if not (key.startswith('upsample_index_') or key.startswith('downsample_index')):
                continue
            if arr.size == 0:
                continue
            # Compute optimal chunks for this meta array
            optimal_chunks = get_optimal_chunks(arr.shape, arr.dtype, target_chunk_bytes=2e4)
            if optimal_chunks != arr.chunks:
                print(f"  Rechunking meta['{key}']: {arr.chunks} -> {optimal_chunks}")
                rechunk_recompress_array(meta_group, key, chunks=optimal_chunks)
            else:
                print(f"  Skipping meta['{key}']: already uses optimal chunks {arr.chunks}")
    
    # Output buffer is written directly to disk via create_from_path above.
    print("\n" + "=" * 60)
    print(f"Concatenated buffer written to {os.path.abspath(working_output_path)}")
    
    # If output should be a zipstore, convert the directory store to zipstore
    if output_is_zip:
        print("\n" + "=" * 60)
        print("Converting to ZipStore format...")
        print(f"  Source: {working_output_path}")
        print(f"  Target: {args.output}")
        
        # Open the directory store
        dir_store = DirectoryStore(working_output_path)
        
        # Create the zipstore
        zip_store = ZipStore(args.output, mode='w')
        
        try:
            # Copy the entire zarr structure from directory store to zipstore
            zarr.copy_store(source=dir_store, dest=zip_store, source_path='/', dest_path='/')
            print("  Successfully converted to ZipStore")
        finally:
            zip_store.close()
            dir_store.close()
        
        # Delete the directory store
        print(f"  Removing temporary directory: {working_output_path}")
        shutil.rmtree(working_output_path)
    
    # Reload buffer from final output path for summary if it's a zip
    if output_is_zip:
        final_buffer = ReplayBuffer.create_from_path(args.output, mode='r')
        final_episode_count = final_buffer.n_episodes
    else:
        final_buffer = new_buffer
        final_episode_count = new_buffer.n_episodes
    
    print("\n" + "=" * 60)
    print("Concatenation complete!")
    print(f"  Total episodes in output: {final_episode_count}")
    for idx, num_episodes in enumerate(buffer_episode_counts, start=1):
        print(f"  Episodes from buffer {idx}: {num_episodes}")
    print(f"  Output saved to: {args.output}")
    print("=" * 60)

    print_replay_buffer_summary(final_buffer)
    print(f"Saved to: {os.path.abspath(args.output)}")


if __name__ == '__main__':
    main()

