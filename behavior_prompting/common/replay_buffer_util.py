"""Utility to print out the contents of a replay buffer for debugging."""

from datetime import datetime
import pathlib
import shutil
import time
from typing import Optional

from filelock import FileLock
import zarr
from behavior_prompting.common.replay_buffer import ReplayBuffer
import cv2
import os
import numpy as np
from tqdm import tqdm
import imageio

def print_replay_buffer_summary(replay_buffer):
    print(replay_buffer)
    if hasattr(replay_buffer, "task_names"):
        task_names = replay_buffer.task_names[:]
        # Count occurrences of each task name
        unique_tasks, counts = np.unique(task_names, return_counts=True)
        # Sort by task name
        sorted_indices = np.argsort(unique_tasks)
        unique_tasks = unique_tasks[sorted_indices]
        counts = counts[sorted_indices]
        print(f'Task names ({len(unique_tasks)} unique):')
        for task_name, count in zip(unique_tasks, counts):
            print(f'  {task_name}: {count}')

    def _print_group_details(group_name, group):
        if group is None or not hasattr(group, "items"):
            return
        for key, array in group.items():
            shape = getattr(array, "shape", None)
            chunks = getattr(array, "chunks", None)
            compressor = getattr(array, "compressor", None)
            print(
                f'{group_name}/{key}: shape={shape}, '
                f'chunks={chunks}, compressor={compressor}'
            )

    data_group = getattr(replay_buffer, "data", None)
    labels_group = getattr(replay_buffer, "labels", None)
    meta_group = getattr(replay_buffer, "meta", None)
    _print_group_details("data", data_group)
    _print_group_details("labels", labels_group)
    _print_group_details("meta", meta_group)

def load_replay_buffer_lmdb(dataset_path, cache_dir):
    assert dataset_path.endswith('.zip'), 'dataset_path must be a zip file'
    
    mod_time = os.path.getmtime(dataset_path)
    stamp = datetime.fromtimestamp(mod_time).isoformat()
    stem_name = os.path.basename(dataset_path).split('.')[0]
    cache_name = '_'.join([stem_name, stamp])
    cache_dir = pathlib.Path(os.path.expanduser(cache_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir.joinpath(cache_name + '.zarr.mdb')
    lock_path = cache_dir.joinpath(cache_name + '.lock')
    
    # load cached file
    print('Acquiring lock on cache.')
    with FileLock(lock_path):
        # cache does not exist
        if not cache_path.exists():
            try:
                with zarr.LMDBStore(str(cache_path),     
                    writemap=True, metasync=False, sync=False, map_async=True, lock=False
                    ) as lmdb_store:
                    with zarr.ZipStore(dataset_path, mode='r') as zip_store:
                        print(f"Copying data to {str(cache_path)}")
                        ReplayBuffer.copy_from_store(
                            src_store=zip_store,
                            store=lmdb_store
                        )
                print("Cache written to disk!")
            except Exception as e:
                shutil.rmtree(cache_path)
                raise e
    
    # open read-only lmdb store
    store = zarr.LMDBStore(str(cache_path), readonly=True, lock=False)
    replay_buffer = ReplayBuffer.create_from_group(
        group=zarr.group(store)
    )
    
    return replay_buffer

def print_replay_buffer_libero(path, vis_frame: bool = False, vis_video: bool = False, one_video_per_task: bool = True, load_buffer_into_memory: bool = False):
    def fix_rgb_image(rgb_image):
        rgb_image = np.rot90(rgb_image, k=2, axes=(0, 1)).copy()
        rgb_image = np.flip(rgb_image, axis=1).copy()
        return rgb_image

    if load_buffer_into_memory:
        with zarr.ZipStore(path, mode='r') as zip_store:
            replay_buffer = ReplayBuffer.copy_from_store(
                src_store=zip_store, 
                store=zarr.MemoryStore()
            )
    else:
        replay_buffer = ReplayBuffer.create_from_path(path)

    print_replay_buffer_summary(replay_buffer)

    out_dir = os.path.dirname(path)
    dataset_name = os.path.basename(path).split('.')[0]
    if vis_frame:
        # Get a random sample index
        sample_index = np.random.randint(0, replay_buffer['agentview_rgb'].shape[0])
        
        # Get and fix both camera views
        agentview_rgb = fix_rgb_image(replay_buffer['agentview_rgb'][sample_index][...,::-1])
        eye_in_hand_rgb = fix_rgb_image(replay_buffer['eye_in_hand_rgb'][sample_index][...,::-1])
        
        # Hstack the images
        combined_image = np.hstack((agentview_rgb, eye_in_hand_rgb))
        
        # Save the combined image
        out_path = os.path.join(out_dir, 'tmp_vis', dataset_name, f'combined_camera_views.png')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, combined_image)
        print(f'Saved combined camera views to {out_path}')

    if vis_video:
        # video of combined camera views
        if one_video_per_task:
            unique_task_names = np.unique(replay_buffer.task_names[:])
            for task_name in unique_task_names:
                vis_task_name = task_name.replace(' ', '_')
                video_path = os.path.join(out_dir, 'tmp_vis', dataset_name, f'{vis_task_name}_combined_camera_views.mp4')
                os.makedirs(os.path.dirname(video_path), exist_ok=True)
                out = imageio.get_writer(video_path, fps=20, codec='libx264')
                task_index = np.where(replay_buffer.task_names[:] == task_name)[0][0]
                end_frame_idx = replay_buffer.task_data_ends[task_index]
                start_frame_idx = end_frame_idx - replay_buffer.task_lengths[task_index]
                for i in tqdm(range(start_frame_idx, end_frame_idx), desc=f'Writing combined camera views video for task \"{task_name}\"'):
                    # Get and fix both camera views
                    agentview_frame = fix_rgb_image(replay_buffer['agentview_rgb'][i][...,::-1])
                    eye_in_hand_frame = fix_rgb_image(replay_buffer['eye_in_hand_rgb'][i][...,::-1])
                    combined_frame = np.hstack((agentview_frame, eye_in_hand_frame))
                    out.append_data(combined_frame[...,::-1]) # convert to RGB
                out.close()
                print(f'Saved combined camera views video for task \"{task_name}\" to {video_path}')
        else:
            video_path = os.path.join(out_dir, 'tmp_vis', dataset_name, f'combined_camera_views.mp4')
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            out = imageio.get_writer(video_path, fps=20, codec='libx264')
            for i in tqdm(range(replay_buffer['agentview_rgb'].shape[0]), desc='Writing combined camera views video'):
                # Get and fix both camera views
                agentview_frame = fix_rgb_image(replay_buffer['agentview_rgb'][i][...,::-1])
                eye_in_hand_frame = fix_rgb_image(replay_buffer['eye_in_hand_rgb'][i][...,::-1])
                
                # Combine frames horizontally
                combined_frame = np.hstack((agentview_frame, eye_in_hand_frame))
                out.append_data(combined_frame[...,::-1]) # convert to RGB

            out.close()
        print(f'Saved combined camera views video to {video_path}')

def print_replay_buffer_umi(path, vis_frame: bool = False, vis_video: bool = False, load_buffer_into_memory: bool = False, task_for_video: Optional[str] = None):
    if load_buffer_into_memory:
        replay_buffer = ReplayBuffer.copy_from_path(path, store=zarr.MemoryStore())
    else:
        replay_buffer = ReplayBuffer.create_from_path(path)

    print_replay_buffer_summary(replay_buffer)

    out_dir = os.path.dirname(path)
    if vis_frame:
        # single frame of main camera
        sample_index = np.random.randint(0, replay_buffer['camera0_main_rgb'].shape[0])
        main_rgb = replay_buffer['camera0_main_rgb'][sample_index][...,::-1]
        out_path = os.path.join(out_dir, 'tmp_main_rgb.png')
        cv2.imwrite(out_path, main_rgb)
        print(f'Saved main RGB image to {out_path}')

        # single frame of ultrawide camera
        ultrawide_sample_index = replay_buffer.map_upsample_index('camera0_ultrawide_rgb', sample_index)
        ultrawide_rgb = replay_buffer['camera0_ultrawide_rgb'][ultrawide_sample_index][...,::-1]
        out_path = os.path.join(out_dir, 'tmp_ultrawide_rgb.png')
        cv2.imwrite(out_path, ultrawide_rgb)
        print(f'Saved ultrawide RGB image to {out_path}')

    if vis_video:
        # Get frame indices to include if task_for_video is specified
        main_frame_indices = None
        ultrawide_frame_indices = None
        if task_for_video is not None:
            task_names = replay_buffer.task_names[:]
            matching_task_indices = np.where(task_names == task_for_video)[0]
            if len(matching_task_indices) == 0:
                print(f'Warning: No tasks found with name "{task_for_video}". Creating empty videos.')
                main_frame_indices = []
                ultrawide_frame_indices = []
            else:
                # Collect all frame indices for matching tasks
                main_frame_indices_list = []
                ultrawide_frame_indices_list = []
                for task_idx in matching_task_indices:
                    end_frame_idx = replay_buffer.task_data_ends[task_idx]
                    start_frame_idx = end_frame_idx - replay_buffer.task_lengths[task_idx]
                    # Add all frames from this task
                    main_frame_indices_list.extend(range(start_frame_idx, end_frame_idx))
                    # For ultrawide, we need to map the main camera indices to ultrawide indices
                    for main_idx in range(start_frame_idx, end_frame_idx):
                        ultrawide_idx = replay_buffer.map_upsample_index('camera0_ultrawide_rgb', main_idx)
                        ultrawide_frame_indices_list.append(ultrawide_idx)
                main_frame_indices = sorted(set(main_frame_indices_list))
                ultrawide_frame_indices = sorted(set(ultrawide_frame_indices_list))
                print(f'Found {len(matching_task_indices)} task(s) with name "{task_for_video}"')
                print(f'Including {len(main_frame_indices)} main camera frames and {len(ultrawide_frame_indices)} ultrawide camera frames in videos')
        
        # video of main camera
        video_path = os.path.join(out_dir, 'tmp_main_video.mp4')
        frame_width = replay_buffer['camera0_main_rgb'].shape[2]
        frame_height = replay_buffer['camera0_main_rgb'].shape[1]
        out = imageio.get_writer(video_path, fps=60, codec='libx264')

        if main_frame_indices is not None:
            # Filter to only specified frames
            frame_range = main_frame_indices
        else:
            # Include all frames
            frame_range = range(replay_buffer['camera0_main_rgb'].shape[0])
        
        for i in tqdm(frame_range, desc='Writing main video'):
            frame = replay_buffer['camera0_main_rgb'][i]
            out.append_data(frame) # convert to RGB

        out.close()
        print(f'Saved main video to {video_path}')

        # video of ultrawide camera
        video_path = os.path.join(out_dir, 'tmp_ultrawide_video.mp4')
        frame_width = replay_buffer['camera0_ultrawide_rgb'].shape[2]
        frame_height = replay_buffer['camera0_ultrawide_rgb'].shape[1]
        out = imageio.get_writer(video_path, fps=10, codec='libx264')

        if ultrawide_frame_indices is not None:
            # Filter to only specified frames
            frame_range = ultrawide_frame_indices
        else:
            # Include all frames
            frame_range = range(replay_buffer['camera0_ultrawide_rgb'].shape[0])

        for i in tqdm(frame_range, desc='Writing ultrawide video'):
            frame = replay_buffer['camera0_ultrawide_rgb'][i]
            out.append_data(frame)

        out.close()
        print(f'Saved ultrawide video to {video_path}')

def print_replay_buffer_draw(path, replay_buffer: Optional[ReplayBuffer] = None, print_summary: bool = True, vis_video: bool = False, vis_drawing_image: bool = False, enable_print: bool = True, load_buffer_into_memory: bool = False, task_for_video: Optional[str] = None):
    # Handle both path string and replay buffer object
    start_time = time.time()
    if replay_buffer is None:
        # If path is provided, load the replay buffer
        if load_buffer_into_memory:
            replay_buffer = ReplayBuffer.copy_from_path(path, store=zarr.MemoryStore())
        else:
            replay_buffer = ReplayBuffer.create_from_path(path)
    end_time = time.time()
    if enable_print:
        print(f'Time taken to load replay buffer: {end_time - start_time} seconds')
    out_dir = os.path.dirname(path)
    if print_summary and enable_print:
        print_replay_buffer_summary(replay_buffer)
        print(f'Episode names: {replay_buffer.episode_names[:]}')
        print(f'Max task length: {max(replay_buffer.meta.task_lengths[:])}')

    if vis_video:
        rgb_data = replay_buffer['image']

        # Get frame indices to include if task_for_video is specified
        frame_indices = None
        if task_for_video is not None:
            task_names = replay_buffer.task_names[:]
            matching_task_indices = np.where(task_names == task_for_video)[0]
            if len(matching_task_indices) == 0:
                if enable_print:
                    print(f'Warning: No tasks found with name "{task_for_video}". Creating empty video.')
                frame_indices = []
            else:
                # Collect all frame indices for matching tasks
                frame_indices_list = []
                for task_idx in matching_task_indices:
                    end_frame_idx = replay_buffer.task_data_ends[task_idx]
                    start_frame_idx = end_frame_idx - replay_buffer.task_lengths[task_idx]
                    # Add all frames from this task
                    frame_indices_list.extend(range(start_frame_idx, end_frame_idx))
                frame_indices = sorted(set(frame_indices_list))
                if enable_print:
                    print(f'Found {len(matching_task_indices)} task(s) with name "{task_for_video}"')
                    print(f'Including {len(frame_indices)} frames in video')

        # Create video writer for rgb data with 224x224 resolution
        video_name = os.path.basename(path).split('.')[0]
        video_path = os.path.join(out_dir, f'tmp_{video_name}_draw_video.mp4')
        frame_width = 224
        frame_height = 224
        out = imageio.get_writer(video_path, fps=10, codec='libx264')

        # Write each frame to video
        if frame_indices is not None:
            # Filter to only specified frames
            frame_range = frame_indices
        else:
            # Include all frames
            frame_range = range(rgb_data.shape[0])
        
        iterator = tqdm(frame_range, desc='Writing draw video') if enable_print else frame_range
        for i in iterator:
            frame = rgb_data[i][...,::-1] # Convert RGB to BGR
            # Resize frame to 224x224
            frame = cv2.resize(frame, (frame_width, frame_height))
            out.append_data(frame[...,::-1]) # convert to RGB

        out.close()
        if enable_print:
            print(f'Saved draw video to {video_path}')

    if vis_drawing_image:
        # Check if drawing_image exists in replay buffer            
        drawing_data = replay_buffer.labels['drawing_image']

        # Get frame indices to include if task_for_video is specified
        frame_indices = None
        if task_for_video is not None:
            task_names = replay_buffer.task_names[:]
            matching_task_indices = np.where(task_names == task_for_video)[0]
            if len(matching_task_indices) == 0:
                if enable_print:
                    print(f'Warning: No tasks found with name "{task_for_video}". Creating empty drawing image video.')
                frame_indices = []
            else:
                # Collect all frame indices for matching tasks
                frame_indices_list = []
                for task_idx in matching_task_indices:
                    end_frame_idx = replay_buffer.task_data_ends[task_idx]
                    start_frame_idx = end_frame_idx - replay_buffer.task_lengths[task_idx]
                    # Add all frames from this task
                    frame_indices_list.extend(range(start_frame_idx, end_frame_idx))
                frame_indices = sorted(set(frame_indices_list))
                if enable_print:
                    print(f'Found {len(matching_task_indices)} task(s) with name "{task_for_video}"')
                    print(f'Including {len(frame_indices)} frames in drawing image video')

        # Create video writer for drawing image with 512x512 resolution
        video_name = os.path.basename(path).split('.')[0]
        video_path = os.path.join(out_dir, f'tmp_{video_name}_drawing_image_video.mp4')
        frame_width = 512
        frame_height = 512
        out = imageio.get_writer(video_path, fps=10, codec='libx264')

        # Write each frame to video
        if frame_indices is not None:
            # Filter to only specified frames
            frame_range = frame_indices
        else:
            # Include all frames
            frame_range = range(drawing_data.shape[0])
        
        iterator = tqdm(frame_range, desc='Writing drawing image video') if enable_print else frame_range
        for i in iterator:
            frame = drawing_data[i][...,::-1] # Convert RGB to BGR
            # Ensure frame is 512x512
            if frame.shape[:2] != (frame_height, frame_width):
                frame = cv2.resize(frame, (frame_width, frame_height))
            out.append_data(frame[...,::-1])

        out.close()
        if enable_print:
            print(f'Saved drawing image video to {video_path}')
