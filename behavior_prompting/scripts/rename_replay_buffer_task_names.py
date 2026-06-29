#!/usr/bin/env python3
"""
Script to rename task_names in a replay buffer.

Usage:
    python rename_replay_buffer_task_names.py <replay_buffer_path> <new_task_name> [old_task_name]
    
    If old_task_name is provided, only tasks with that name will be renamed.
    If old_task_name is not provided, all tasks will be renamed to the new task name.
"""

import sys
import os
import argparse
import numpy as np
from behavior_prompting.common.replay_buffer import ReplayBuffer


def rename_task_names(replay_buffer_path: str, new_task_name: str, old_task_name: str = None):
    """
    Load a replay buffer and rename task_names to the provided task name.
    
    Args:
        replay_buffer_path: Path to the replay buffer file
        new_task_name: New task name to assign to tasks
        old_task_name: Optional old task name. If provided, only tasks with this name will be renamed.
                      If None, all tasks will be renamed.
    """
    # Check if file exists
    if not os.path.exists(replay_buffer_path):
        raise FileNotFoundError(f"Replay buffer file not found: {replay_buffer_path}")
    
    print(f"Loading replay buffer from: {replay_buffer_path}")
    
    # Load the replay buffer in write mode
    replay_buffer = ReplayBuffer.create_from_path(replay_buffer_path, mode='r+')
    
    # Get current task names
    current_task_names = replay_buffer.task_names[:]
    n_tasks = len(current_task_names)
    
    print(f"Found {n_tasks} tasks with current names: {current_task_names}")
    
    if n_tasks == 0:
        print("No tasks found in replay buffer. Nothing to rename.")
        return
    
    # Create new task names array
    new_task_names = current_task_names.copy()
    
    if old_task_name is not None:
        # Only rename tasks that match the old task name
        rename_mask = current_task_names == old_task_name
        rename_count = np.sum(rename_mask)
        
        if rename_count == 0:
            print(f"No tasks found with name '{old_task_name}'. Nothing to rename.")
            return
        
        new_task_names[rename_mask] = new_task_name
        print(f"Renamed {rename_count} tasks from '{old_task_name}' to '{new_task_name}'")
    else:
        # Rename all tasks
        new_task_names[:] = new_task_name
        print(f"Renamed all {n_tasks} tasks to: {new_task_name}")
    
    # Replace the task_names in the meta group
    meta_group = replay_buffer.meta
    task_names_array = meta_group['task_names']
    
    # Clear the existing array and resize it
    task_names_array.resize((n_tasks,))
    
    # Assign the new task names
    task_names_array[:] = new_task_names
    
    # The changes are automatically saved since we're using 'r+' mode
    print("Changes saved to replay buffer.")

    print(f"New task names: {replay_buffer.task_names[:]}")


def main():
    parser = argparse.ArgumentParser(
        description="Rename task_names in a replay buffer to a specified task name"
    )
    parser.add_argument(
        "replay_buffer_path",
        help="Path to the replay buffer file"
    )
    parser.add_argument(
        "new_task_name",
        help="New task name to assign to tasks"
    )
    parser.add_argument(
        "old_task_name",
        nargs='?',
        default=None,
        help="Optional old task name. If provided, only tasks with this name will be renamed. "
             "If not provided, all tasks will be renamed."
    )
    
    args = parser.parse_args()
    
    rename_task_names(args.replay_buffer_path, args.new_task_name, args.old_task_name)
    print("Task renaming completed successfully!")

if __name__ == "__main__":
    main()

