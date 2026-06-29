import shutil
import numpy as np
import click
from behavior_prompting.common.replay_buffer import ReplayBuffer
from behavior_prompting.common.replay_buffer_util import print_replay_buffer_draw
from behavior_prompting.train_network.env.draw.draw_env import DrawEnv
import pygame
import os
import glob
import zarr

from behavior_prompting.train_network.utils.draw_util import get_target_drawing_image_first_demonstration

def task_name_to_dataset_name(task_name):
    """Convert task name to a valid filename by replacing spaces with underscores."""
    safe_task_name = task_name.replace(' ', '_')
    ending = safe_task_name[len('draw_'):]
    if ending.upper() == ending:
        safe_task_name += '_upper'
    elif ending.lower() == ending:
        safe_task_name += '_lower'
    else:
        assert False, f"Task name {task_name} is not valid"
    return safe_task_name + '.zarr'

def get_task_dataset_path(output_dir, task_name):
    """Convert task name to a valid filename by replacing spaces with underscores."""
    return os.path.join(output_dir, task_name_to_dataset_name(task_name))

def get_or_create_replay_buffer_for_task(output_dir, task_name, in_memory, ignore_empty=False):
    """Get or create a replay buffer for a specific task."""
    dataset_path = get_task_dataset_path(output_dir, task_name)
    
    if os.path.exists(dataset_path):
        if in_memory:
            replay_buffer = ReplayBuffer.copy_from_path(dataset_path, store=zarr.MemoryStore())
        else:
            replay_buffer = ReplayBuffer.create_from_path(dataset_path, mode='r+')
        episode_count = replay_buffer.n_episodes
        assert ignore_empty or episode_count > 0, f"Task: '{task_name}' has no episodes"
    else:
        if in_memory:
            replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())
        else:
            replay_buffer = ReplayBuffer.create_from_path(dataset_path, mode='w')
    
    return replay_buffer

def get_all_task_names(output_dir, new_task_name=None):
    """Get all task names from existing datasets in the output directory."""
    if not os.path.exists(output_dir):
        return []
    
    task_names = []
    for dataset_path in glob.glob(os.path.join(output_dir, "*.zarr")):
        replay_buffer = ReplayBuffer.create_from_path(dataset_path, mode='r')
        if replay_buffer.n_episodes > 0:
            # Get the task name from the first episode
            cur_task_name = replay_buffer.task_names[0]
            task_names.append(cur_task_name)

    if new_task_name:
        assert new_task_name not in task_names, f"Task name {new_task_name} already exists"
        task_names.append(new_task_name)
    
    return sorted(task_names)

def cleanup_empty_tasks(output_dir):
    """Cleanup empty tasks from the output directory."""
    for dataset_path in glob.glob(os.path.join(output_dir, "*.zarr")):
        replay_buffer = ReplayBuffer.create_from_path(dataset_path, mode='r')
        if replay_buffer.n_episodes == 0:
            print(f'Warning: {dataset_path} has no episodes, thus deleting it')
            shutil.rmtree(dataset_path)

def get_total_episodes(output_dir):
    """Get total number of episodes across all tasks."""
    total_episodes = 0
    for cur_task_name in get_all_task_names(output_dir):
        replay_buffer = get_or_create_replay_buffer_for_task(output_dir, cur_task_name, in_memory=False)
        total_episodes += replay_buffer.n_episodes
    return total_episodes

@click.command()
@click.option('-o', '--output', required=True)
@click.option('-hz', '--control_hz', default=10, type=int)
@click.option('--boundary-angle', type=float, default=None, help='Boundary rotation angle in radians. If not specified, will be random within the range -π/4 to π/4.')
@click.option('--persistent/--no-persistent', default=True, help='If set to --no-persistent, the replay buffer file will be deleted at the end of the script.')
def main_click(output, control_hz, boundary_angle, persistent):
    """
    Collect demonstration for the Draw task using behavior_prompting replay buffer format.
    
    Usage: python demo_draw.py -o datasets/draw/demo
           python demo_draw.py -o datasets/draw/demo --boundary-angle 0.5  # Set specific boundary angle
    
    This script is compatible with both Linux and MacOS.
    The script creates a folder at the specified output path and creates separate zarr datasets
    for each task within that folder. Task names with spaces are converted to underscores
    in the dataset filenames. Image data is compressed using JPEG-XL for efficient storage.
    
    Task Naming Convention:
    - Task names must be all uppercase or all lowercase (e.g., "CIRCLE" or "circle")
    - "draw " is automatically prepended to all task names (e.g., "CIRCLE" becomes "draw CIRCLE")
    - Task names are case-sensitive and must be unique
    
    Example folder structure:
    datasets/draw/demo/
    ├── draw_circle.zarr/
    ├── draw_square.zarr/
    └── draw_triangle.zarr/
    
    Hover mouse close to the blue circle to start.
    Draw within the boundary area. 
    The episode will automatically terminate when you press "D" for done.
    
    Controls:
    - Hover mouse near blue circle to control agent
    - Left mouse button to put pen down/up
    - Press "Ctrl+Q" to exit
    - Press "Ctrl+W" to drop an episode
    - Press "Ctrl+N" to create a new task
    - Press "Ctrl+S" to print task summary
    - Press "Ctrl+G" to go to a specific task (prompts for task name)
    - Press "Ctrl+V" to generate videos for all datasets
    - Press "R" to retry current episode (only when drawing has started)
    - Press "D" when done drawing (only when drawing has started)
    - Press left/right arrow keys to navigate to previous/next task (before drawing starts)
    
    Note: Task navigation controls (arrow keys, Ctrl+G) only work before drawing starts.
    Once you begin drawing, only "R" (retry) and "D" (done) are available.
    
    Boundary Feature:
    The boundary is a rotatable square (350x350) centered at (256,256).
    Use --boundary-angle to set a specific rotation (0 = parallel to window edges, π/4 = 45° rotation).
    If not specified, a random angle between -π/4 and π/4 will be used for each episode. The rotation convention is that 0 is upright and a positive angle is a clockwise rotation.

    You should use group_demos.py to group demos into tasks after you are done collecting demos to create a replay buffer that is suitable for training.
    """
    collect_demos(output, control_hz=control_hz, boundary_angle=boundary_angle, persistent=persistent, single_demo=False)

def collect_demos(output, control_hz=10, boundary_angle=None, persistent=True, single_demo=False, env=None, in_memory_replay_buffer=False):
    if in_memory_replay_buffer:
        assert single_demo, "in_memory_replay_buffer is only supported for single demo"
    
    # create output directory if it doesn't exist
    os.makedirs(output, exist_ok=True)
    cleanup_empty_tasks(output)
    
    def print_task_summary(new_task_name=None):
        """Print a summary of all tasks and their demo counts."""
        msg = f"======== Task Summary for {output} ========"
        print(f'\n{msg}')
        all_task_names = get_all_task_names(output, new_task_name)
        if all_task_names:
            total_demos = 0
            for cur_task_name in all_task_names:
                replay_buffer = get_or_create_replay_buffer_for_task(output, cur_task_name, in_memory=False, ignore_empty=True)
                demo_count = replay_buffer.n_episodes
                total_demos += demo_count
                print(f'  "{cur_task_name}": {demo_count} demos')
            print(f'\nTotal: {len(all_task_names)} tasks, {total_demos} demos')
        else:
            print('  No tasks found')
        print('=' * len(msg) + '\n')
    
    # Print initial task summary
    print_task_summary()

    # create draw env with drawing enabled
    if env is None:
        env = DrawEnv(
            boundary_angle=boundary_angle,
            render_mode='human'
        )
    agent = env.teleop_agent()
    clock = pygame.time.Clock()

    def input_task_name():
        nonlocal is_new_task, task_index, last_task_index, obs, img, current_replay_buffer, new_task_name, display_current_task_name, display_all_task_names

        is_new_task = True
        env.set_target_drawing(None)
        obs = env.reset(no_rotation=True)
        img = (np.transpose(obs['image'], (1,2,0)) * 255).astype(np.uint8)
        new_task_name = None if not single_demo else 'draw single demo'
        existing_task_names = get_all_task_names(output)
        
        while not new_task_name:
            new_task_name = input('Enter new task name ("draw " will be prepended automatically): ').strip()

            if new_task_name == '':
                print('Task name cannot be empty')
                new_task_name = None
                continue
            
            if new_task_name == '/' or new_task_name == '\\':
                print('Task name cannot be "/" or "\\"') # otherwise it would interfere with path separator
                new_task_name = None
                continue

            if new_task_name.upper() != new_task_name and new_task_name.lower() != new_task_name:
                # have to check this because MacOS is case-preserving but case-insensitive by default so we add _upper or _lower to the name
                print('Task name must be all uppercase or all lowercase')
                new_task_name = None
                continue

            new_task_name = f'draw {new_task_name}'
            if new_task_name in existing_task_names:
                print(f'Task name "{new_task_name}" already exists')
                new_task_name = None
                continue

        # Create replay buffer for the new task
        current_replay_buffer = get_or_create_replay_buffer_for_task(output, new_task_name, in_memory=in_memory_replay_buffer)
        existing_task_names = get_all_task_names(output, new_task_name)
        task_index = existing_task_names.index(new_task_name)
        last_task_index = task_index # so we don't cause an update

        display_current_task_name = new_task_name
        display_all_task_names = existing_task_names

    is_new_task = True
    task_index = 0
    reward = 0
    current_replay_buffer = None
    new_task_name = None

    def get_cur_task_name():
        return new_task_name if is_new_task else get_all_task_names(output)[task_index]
    
    if get_all_task_names(output):
        assert not single_demo, "single demo is not supported when there are existing tasks"
        # Start with the first existing task
        is_new_task = False
        new_task_name = None
        task_index = 0
        task_name = get_cur_task_name()
        current_replay_buffer = get_or_create_replay_buffer_for_task(output, task_name, in_memory=in_memory_replay_buffer)
        target_drawing, boundary_angle = get_target_drawing_image_first_demonstration(current_replay_buffer, task_name)
        env.set_target_drawing(target_drawing, boundary_angle)
    else:
        input_task_name()
    
    # episode-level while loop
    episode_idx = get_total_episodes(output)
    display_current_task_name = get_cur_task_name()
    display_all_task_names = get_all_task_names(output)
    while True:
        episode_data = {
            'image': [],
            'agent_pos': [], 
            'pen_down': [],
            'action': []
        }
        episode_labels = {
            'boundary_angle': [],
            'drawing_image': []
        }
        
        # reset env and get observations
        # set seed for env
        if not single_demo:
            env.seed(episode_idx)
        obs = env.reset(no_rotation=is_new_task)
        img = (np.transpose(obs['image'], (1,2,0)) * 255).astype(np.uint8)
        
        # loop state
        retry = False
        done = False
        force_ep_done = False
        step_idx = 0
        display_current_task_name = get_cur_task_name()
        display_all_task_names = get_all_task_names(output, new_task_name)
        # step-level while loop
        while not done:
            episode_count = current_replay_buffer.n_episodes if current_replay_buffer else 0
            pygame.display.set_caption(f'{display_current_task_name} ({episode_count} demos) | {len(display_all_task_names)} tasks | Step: {step_idx} | Reward: {reward:.2f}')
            
            # process keypress events
            last_task_index = task_index
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r and step_idx > 0:
                        # press "R" to retry current episode
                        retry = True
                    elif event.key == pygame.K_q and pygame.key.get_mods() & pygame.KMOD_CTRL and step_idx == 0:
                        is_new_task = False
                        new_task_name = None

                        cleanup_empty_tasks(output)

                        print_task_summary(new_task_name)

                        # At the end of the script, delete the replay buffer if persistent is False
                        if not persistent:
                            try:
                                if input(f'Press Y to delete the replay buffer directory {output}: ') == 'Y':
                                    shutil.rmtree(output)
                                    print(f"Replay buffer directory '{output}' deleted as persistent=False.")
                            except Exception as e:
                                print(f"Failed to delete replay buffer directory '{output}': {e}")
                        exit(0)
                    elif event.key == pygame.K_q and not (pygame.key.get_mods() & pygame.KMOD_CTRL):
                        if single_demo:
                            return None
                    elif event.key == pygame.K_v and pygame.key.get_mods() & pygame.KMOD_CTRL and step_idx == 0:
                        # Delete all mp4 files in the output directory
                        mp4_files = glob.glob(os.path.join(output, "*.mp4"))
                        for mp4_file in mp4_files:
                            try:
                                os.remove(mp4_file)
                            except Exception as e:
                                print(f"Failed to delete mp4 file {mp4_file}: {e}")
                        
                        # Generate videos for all datasets
                        for cur_task_name in get_all_task_names(output):
                            dataset_path = get_task_dataset_path(output, cur_task_name)
                            print(f"\n--- Dataset for task '{cur_task_name}' ---")
                            print_replay_buffer_draw(dataset_path, print_summary=False, vis_video=True)
                    elif event.key == pygame.K_w and pygame.key.get_mods() & pygame.KMOD_CTRL and step_idx == 0:
                        # press "Ctrl+W" to drop episode
                        if episode_idx > 0: # if we have collected at least one demo
                            if is_new_task:
                                # case where we are creating a new task just stop creating the new task
                                # we also need to delete the replay buffer for the new task
                                path_to_delete = get_task_dataset_path(output, new_task_name)
                                shutil.rmtree(path_to_delete)
                                print(f'Deleted replay buffer for task: "{new_task_name}" since you never collected any demos for it')

                                task_index = 0 # go back to the first task
                                last_task_index = -1
                                is_new_task = False
                                new_task_name = None
                                retry = True
                                print('Stopping new task creation but did not drop episode since you never collected any demos for it')
                            else:
                                # case where we are not creating a new task
                                if current_replay_buffer.n_episodes == 1:
                                    # delete replay buffer
                                    task_name_to_delete = get_cur_task_name()
                                    path_to_delete = get_task_dataset_path(output, task_name_to_delete)
                                    shutil.rmtree(path_to_delete)
                                    print(f'Deleted replay buffer for task: "{task_name_to_delete}" since it had no more demos')

                                    # update task names
                                    task_index = 0 # go back to the first task
                                    last_task_index = -1
                                else:
                                    current_replay_buffer.drop_episode()

                                episode_idx -= 1
                                print(f'Dropped episode {episode_idx}')
                                retry = True
                            
                            if episode_idx == 0:
                                # this is the case where we deleted the only episode in the task
                                input_task_name()
                        else:
                            print('No episode to drop')
                    elif event.key == pygame.K_n and pygame.key.get_mods() & pygame.KMOD_CTRL and step_idx == 0:
                        input_task_name()
                    elif event.key == pygame.K_s and pygame.key.get_mods() & pygame.KMOD_CTRL and step_idx == 0:
                        # press "Ctrl+S" to print all task names and demo counts
                        print_task_summary(new_task_name)
                    elif event.key == pygame.K_g and pygame.key.get_mods() & pygame.KMOD_CTRL and step_idx == 0:
                        # press "Ctrl+G" to go to a specific task
                        if is_new_task:
                            print('Cannot switch tasks when creating a new task')
                            continue
                        
                        task_names = get_all_task_names(output)
                        if len(task_names) == 0:
                            print('No tasks available to switch to')
                            continue
                        
                        print(f'Available tasks: {task_names}')
                        target_task = input('Enter task name to switch to ("draw " will be prepended automatically): ').strip()

                        target_task = f'draw {target_task}'
                        if target_task in task_names:
                            task_index = task_names.index(target_task)
                            print(f'Switched to task: "{target_task}"')
                        else:
                            print(f'Task "{target_task}" not found. Available tasks: {task_names}')
                            continue
                    elif event.key == pygame.K_d and step_idx > 0:
                        # press "D" when done drawing
                        force_ep_done = True
                    elif event.key == pygame.K_LEFT and step_idx == 0:
                        if is_new_task:
                            print('Cannot go to previous task when creating a new task')
                            continue
                        # press left arrow to go to previous task
                        task_names = get_all_task_names(output, new_task_name)
                        if len(task_names) > 0:
                            task_index = max(0, task_index - 1)
                    elif event.key == pygame.K_RIGHT and step_idx == 0:
                        if is_new_task:
                            print('Cannot go to next task when creating a new task')
                            continue
                        # press right arrow to go to next task
                        task_names = get_all_task_names(output, new_task_name)
                        if len(task_names) > 0:
                            task_index = min(len(task_names) - 1, task_index + 1)

            if task_index != last_task_index:
                if task_index < len(get_all_task_names(output)):
                    # Switch to existing task
                    task_name = get_cur_task_name()
                    current_replay_buffer = get_or_create_replay_buffer_for_task(output, task_name, in_memory=in_memory_replay_buffer)
                    target_drawing, boundary_angle = get_target_drawing_image_first_demonstration(current_replay_buffer, task_name)
                    env.set_target_drawing(target_drawing, boundary_angle)
                    is_new_task = False
                    new_task_name = None
                else:
                    # Create new task
                    # input_task_name()
                    assert False, 'should not reach here'
                
                display_current_task_name = get_cur_task_name()
                display_all_task_names = get_all_task_names(output, new_task_name)

            # handle control flow
            if retry:
                break
            
            # get action from mouse
            # None if mouse is not close to the agent
            act = agent.act(obs)
            if act is not None:
                # teleop started - record step
                
                # Image: (H, W, C) uint8 -> normalize to [0,1] and keep as uint8*255 for storage
                img_data = img.astype(np.uint8)
                
                # Agent position: (x, y) coordinates
                agent_pos_data = np.array(obs['agent_pos'], dtype=np.float32)
                
                # Pen down state: binary indicator
                pen_down_data = np.array(obs['pen_down'], dtype=np.float32)
                
                # Action: (x, y, pen_down) where pen_down is binary
                action_data = np.array(act, dtype=np.float32)
                
                # Store data for this step
                episode_data['image'].append(img_data)
                episode_data['agent_pos'].append(agent_pos_data)
                episode_data['pen_down'].append(pen_down_data)
                episode_data['action'].append(action_data)
                
                # store labels for this step
                episode_labels['boundary_angle'].append(np.array(env.boundary_angle, dtype=np.float32))
                episode_labels['drawing_image'].append(env.get_drawing_image())
                
                step_idx += 1
                
            # step env and render
            obs, reward, done, info = env.step(act)
            img = (np.transpose(obs['image'], (1,2,0)) * 255).astype(np.uint8)
            done = force_ep_done or done
            
            # regulate control frequency
            clock.tick(control_hz)
            
        if not retry and len(episode_data['image']) > 0:
            # Convert lists to numpy arrays
            for key in episode_data:
                episode_data[key] = np.stack(episode_data[key])
            
            # Create a single task that spans the entire episode
            cur_task_name = get_cur_task_name()
            task_length = len(episode_data['image'])
            task_data = [{
                "name": cur_task_name,
                "start_idx": 0, 
                "end_idx": task_length,
                "labels": episode_labels
            }]
            
            # validate that all task_names in the replay buffer have the same task name
            assert np.all(current_replay_buffer.task_names[:] == cur_task_name), f"All episodes in the replay buffer must have the same task name as the current task: {cur_task_name}"

            # note we do not use image compression here becuase we want to save the data really fast to make the UI not lag
            current_replay_buffer.add_episode(
                data=episode_data,
                tasks=task_data,
                episode_name=f"episode_{episode_idx:04d}"
            )
            
            print(f'Saved episode {episode_idx} with {len(episode_data["image"])} steps for task: "{cur_task_name}"')
            episode_idx += 1

            if is_new_task:
                # Update the target image to be from this task
                target_drawing, boundary_angle = get_target_drawing_image_first_demonstration(current_replay_buffer, get_cur_task_name())
                env.set_target_drawing(target_drawing, boundary_angle)
                is_new_task = False
                new_task_name = None

            if single_demo:
                return current_replay_buffer
        else:
            print(f'Retrying episode {episode_idx}')

if __name__ == "__main__":
    main_click()
