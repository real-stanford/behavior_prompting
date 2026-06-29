"""
Links dataset results from a run directory (from generate_demonstrations.py) 
into the libero datasets folder. If existing symlinks/directories are present, they are deleted first.
Also provides functionality to delete cache entries (rollout_cache and train_cache).

Default behavior:
- Extra splits (ending with _extra): Copied to libero datasets folder
- View splits (ending with _view): Copied to libero datasets folder, with internal symlinks
  updated to point to source splits in libero datasets folder

With --symlink flag:
- Extra splits: Symlinked to libero datasets folder
- View splits: Copied to libero datasets folder, with internal symlinks updated to point to
  source splits in libero datasets folder (which may be symlinks themselves)
"""

import argparse
import os
import shutil
import yaml
from libero.libero import get_libero_path


def load_config_cache_paths(config_path, train_network_base):
    """
    Load cache paths from libero_defaults.yaml config file.
    Cache paths are resolved relative to the train_network folder base.
    
    Args:
        config_path: Path to the libero_defaults.yaml config file
        train_network_base: Base path of the train_network folder
        
    Returns:
        tuple: (rollout_cache_dir, train_cache_dir) as absolute paths
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    rollout_cache_dir_rel = config.get('env_runner', {}).get('cache_dir', 'cache/libero/rollout_cache')
    train_cache_dir_rel = config.get('dataset', {}).get('cache_dir', 'cache/libero/train_cache')
    
    # Resolve relative to train_network base
    rollout_cache_dir = os.path.abspath(os.path.join(train_network_base, rollout_cache_dir_rel))
    train_cache_dir = os.path.abspath(os.path.join(train_network_base, train_cache_dir_rel))
    
    return rollout_cache_dir, train_cache_dir


def find_config_and_get_cache_paths(config_path_arg=None, rollout_cache_base_arg=None, train_cache_base_arg=None):
    """
    Find config file and get cache paths, handling defaults and user overrides.
    
    Args:
        config_path_arg: Optional path to config file (if None, will search for it)
        rollout_cache_base_arg: Optional path to rollout cache base (if None, uses config default)
        train_cache_base_arg: Optional path to train cache base (if None, uses config default)
        
    Returns:
        tuple: (rollout_cache_base, train_cache_base) as absolute paths
    """
    # Find config file if not provided
    if config_path_arg is None:
        # Try to find the config file relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up from scripts/libero/ to train_network/, then to config/task/
        config_path = os.path.join(script_dir, '../../config/task/libero_defaults.yaml')
        config_path = os.path.abspath(config_path)
        
        if not os.path.exists(config_path):
            # Try alternative: look for behavior_prompting in the path
            # This script is at: behavior_prompting/train_network/scripts/libero/link_generated_demonstrations.py
            # Config is at: behavior_prompting/train_network/config/task/libero_defaults.yaml
            parts = script_dir.split(os.sep)
            if 'behavior_prompting' in parts:
                behavior_prompting_idx = parts.index('behavior_prompting')
                base_path = os.sep.join(parts[:behavior_prompting_idx + 1])
                config_path = os.path.join(base_path, 'behavior_prompting', 'train_network', 'config', 'task', 'libero_defaults.yaml')
    else:
        config_path = config_path_arg
    
    if not os.path.exists(config_path):
        raise ValueError(f"Config file not found: {config_path}. Please specify --config-path")
    
    # Find train_network base directory (config is at train_network/config/task/libero_defaults.yaml)
    config_path_abs = os.path.abspath(config_path)
    # Go up from config/task/libero_defaults.yaml to train_network/
    train_network_base = os.path.dirname(os.path.dirname(os.path.dirname(config_path_abs)))
    
    # Load default cache paths from config (resolved relative to train_network base)
    rollout_cache_base_default, train_cache_base_default = load_config_cache_paths(config_path, train_network_base)
    
    # Use provided cache bases or defaults from config
    # User-provided paths are assumed to be absolute or relative to current directory
    if rollout_cache_base_arg:
        rollout_cache_base = os.path.abspath(rollout_cache_base_arg)
    else:
        rollout_cache_base = rollout_cache_base_default
        
    if train_cache_base_arg:
        train_cache_base = os.path.abspath(train_cache_base_arg)
    else:
        train_cache_base = train_cache_base_default
    
    return rollout_cache_base, train_cache_base


def delete_cache_entry(cache_path, cache_type):
    """
    Delete a cache directory after user confirmation.
    
    Args:
        cache_path: Path to the cache directory to delete
        cache_type: Type of cache (e.g., 'rollout_cache' or 'train_cache')
        
    Returns:
        bool: True if deletion was confirmed and performed, False otherwise
    """
    if not os.path.exists(cache_path):
        print(f"{cache_type} path does not exist: {cache_path}")
        return False
    
    print(f"\n{cache_type} path to delete: {cache_path}")
    response = input(f"Delete {cache_type}? (y/n): ").strip().lower()
    
    if response == 'y':
        try:
            if os.path.islink(cache_path):
                os.remove(cache_path)
                print(f"Removed symlink: {cache_path}")
            else:
                shutil.rmtree(cache_path)
                print(f"Deleted directory: {cache_path}")
            return True
        except Exception as e:
            print(f"Error deleting {cache_path}: {e}")
            return False
    else:
        print(f"Skipping deletion of {cache_type}")
        return False


def find_rollout_cache_splits(rollout_cache_base, filter_splits=None):
    """
    Find all splits in rollout cache that are NOT base splits.
    
    Base splits that are preserved: libero_10, libero_90, libero_goal, libero_object, libero_spatial.
    All other splits are candidates for deletion.
    
    Args:
        rollout_cache_base: Base path for rollout cache
        filter_splits: Optional set/list of split names to filter by. If provided, only returns
                      splits that are in this set and are not base splits.
        
    Returns:
        tuple: (list of split names to delete, rollout_cache_libero_dir) or (None, None) if not found
    """
    rollout_cache_libero_dir = os.path.join(rollout_cache_base, 'libero')
    
    if not os.path.exists(rollout_cache_libero_dir):
        return None, None
    
    # Base splits that should NOT be deleted
    base_splits = {'libero_10', 'libero_90', 'libero_goal', 'libero_object', 'libero_spatial'}
    
    # Find all directories that are not base splits
    splits_to_delete = []
    if os.path.isdir(rollout_cache_libero_dir):
        for item in os.listdir(rollout_cache_libero_dir):
            item_path = os.path.join(rollout_cache_libero_dir, item)
            if os.path.isdir(item_path) or os.path.islink(item_path):
                # Include if not a base split
                if item not in base_splits:
                    # If filter_splits is provided, only include splits that are in the filter
                    if filter_splits is None or item in filter_splits:
                        splits_to_delete.append(item)
    
    if not splits_to_delete:
        return None, None
    
    # Sort for consistent display
    splits_to_delete.sort()
    return splits_to_delete, rollout_cache_libero_dir


def delete_rollout_cache_splits(rollout_cache_libero_dir, splits_to_delete):
    """
    Delete the specified rollout cache splits.
    
    Args:
        rollout_cache_libero_dir: Base directory for rollout cache libero
        splits_to_delete: List of split names to delete
        
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    deleted_count = 0
    for split in splits_to_delete:
        split_path = os.path.join(rollout_cache_libero_dir, split)
        try:
            if os.path.islink(split_path):
                os.remove(split_path)
                print(f"Removed symlink: {split_path}")
            else:
                shutil.rmtree(split_path)
                print(f"Deleted directory: {split_path}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {split_path}: {e}")
    
    print(f"Successfully deleted {deleted_count}/{len(splits_to_delete)} rollout_cache split(s)")
    return deleted_count == len(splits_to_delete)


def find_train_cache_splits(train_cache_base, filter_splits=None):
    """
    Find all splits in train cache that are NOT base splits.
    
    Base splits that are preserved: libero_10, libero_90, libero_goal, libero_object, libero_spatial.
    All other splits are candidates for deletion.
    
    Args:
        train_cache_base: Base path for train cache
        filter_splits: Optional set/list of split names to filter by. If provided, only returns
                      splits that are in this set and are not base splits.
        
    Returns:
        tuple: (list of split names to delete, train_cache_libero_dir) or (None, None) if not found
    """
    train_cache_libero_dir = os.path.join(train_cache_base, 'libero')
    
    if not os.path.exists(train_cache_libero_dir):
        return None, None
    
    # Base splits that should NOT be deleted
    base_splits = {'libero_10', 'libero_90', 'libero_goal', 'libero_object', 'libero_spatial'}
    
    # Find all directories that are not base splits
    splits_to_delete = []
    if os.path.isdir(train_cache_libero_dir):
        for item in os.listdir(train_cache_libero_dir):
            item_path = os.path.join(train_cache_libero_dir, item)
            if os.path.isdir(item_path) or os.path.islink(item_path):
                # Include if not a base split
                if item not in base_splits:
                    # If filter_splits is provided, only include splits that are in the filter
                    if filter_splits is None or item in filter_splits:
                        splits_to_delete.append(item)
    
    if not splits_to_delete:
        return None, None
    
    # Sort for consistent display
    splits_to_delete.sort()
    return splits_to_delete, train_cache_libero_dir


def delete_train_cache_splits(train_cache_libero_dir, splits_to_delete):
    """
    Delete the specified train cache splits.
    
    Args:
        train_cache_libero_dir: Base directory for train cache libero
        splits_to_delete: List of split names to delete
    
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    deleted_count = 0
    for split in splits_to_delete:
        split_path = os.path.join(train_cache_libero_dir, split)
        try:
            if os.path.islink(split_path):
                os.remove(split_path)
                print(f"Removed symlink: {split_path}")
            else:
                shutil.rmtree(split_path)
                print(f"Deleted directory: {split_path}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {split_path}: {e}")
    
    print(f"Successfully deleted {deleted_count}/{len(splits_to_delete)} train_cache split(s)")
    return deleted_count == len(splits_to_delete)


def find_bddl_files_splits(splits_being_linked=None):
    """
    Find all splits in bddl_files that are NOT in splits_being_linked and NOT in base_splits.
    
    Base splits that are preserved: libero_10, libero_90, libero_goal, libero_object, libero_spatial.
    Splits being linked are also preserved.
    All other splits are candidates for deletion.
    
    Args:
        splits_being_linked: Optional set/list of split names that are being linked (should be preserved)
        
    Returns:
        tuple: (list of split names to delete, bddl_files_dir) or (None, None) if not found
    """
    bddl_files_dir = get_libero_path("bddl_files")
    
    if not os.path.exists(bddl_files_dir):
        return None, None
    
    # Base splits that should NOT be deleted
    base_splits = {'libero_10', 'libero_90', 'libero_goal', 'libero_object', 'libero_spatial'}
    
    # Splits to preserve (base splits + splits being linked)
    splits_to_preserve = set(base_splits)
    if splits_being_linked:
        splits_to_preserve.update(splits_being_linked)
    
    # Find all directories that are not in splits_to_preserve
    splits_to_delete = []
    if os.path.isdir(bddl_files_dir):
        for item in os.listdir(bddl_files_dir):
            item_path = os.path.join(bddl_files_dir, item)
            if os.path.isdir(item_path) or os.path.islink(item_path):
                # Include if not in splits_to_preserve
                if item not in splits_to_preserve:
                    splits_to_delete.append(item)
    
    if not splits_to_delete:
        return None, None
    
    # Sort for consistent display
    splits_to_delete.sort()
    return splits_to_delete, bddl_files_dir


def find_init_files_splits(splits_being_linked=None):
    """
    Find all splits in init_files that are NOT in splits_being_linked and NOT in base_splits.
    
    Base splits that are preserved: libero_10, libero_90, libero_goal, libero_object, libero_spatial.
    Splits being linked are also preserved.
    All other splits are candidates for deletion.
    
    Args:
        splits_being_linked: Optional set/list of split names that are being linked (should be preserved)
        
    Returns:
        tuple: (list of split names to delete, init_files_dir) or (None, None) if not found
    """
    init_files_dir = get_libero_path("init_states")
    
    if not os.path.exists(init_files_dir):
        return None, None
    
    # Base splits that should NOT be deleted
    base_splits = {'libero_10', 'libero_90', 'libero_goal', 'libero_object', 'libero_spatial'}
    
    # Splits to preserve (base splits + splits being linked)
    splits_to_preserve = set(base_splits)
    if splits_being_linked:
        splits_to_preserve.update(splits_being_linked)
    
    # Find all directories that are not in splits_to_preserve
    splits_to_delete = []
    if os.path.isdir(init_files_dir):
        for item in os.listdir(init_files_dir):
            item_path = os.path.join(init_files_dir, item)
            if os.path.isdir(item_path) or os.path.islink(item_path):
                # Include if not in splits_to_preserve
                if item not in splits_to_preserve:
                    splits_to_delete.append(item)
    
    if not splits_to_delete:
        return None, None
    
    # Sort for consistent display
    splits_to_delete.sort()
    return splits_to_delete, init_files_dir


def find_datasets_splits(splits_being_linked=None):
    """
    Find all splits in libero datasets folder that are NOT in splits_being_linked and NOT in base_splits.
    
    Base splits that are preserved: libero_10, libero_90, libero_goal, libero_object, libero_spatial.
    Splits being linked are also preserved.
    All other splits are candidates for deletion.
    
    Args:
        splits_being_linked: Optional set/list of split names that are being linked (should be preserved)
        
    Returns:
        tuple: (list of split names to delete, datasets_dir) or (None, None) if not found
    """
    datasets_dir = get_libero_path("datasets")
    
    if not os.path.exists(datasets_dir):
        return None, None
    
    # Base splits that should NOT be deleted
    base_splits = {'libero_10', 'libero_90', 'libero_goal', 'libero_object', 'libero_spatial'}
    
    # Splits to preserve (base splits + splits being linked)
    splits_to_preserve = set(base_splits)
    if splits_being_linked:
        splits_to_preserve.update(splits_being_linked)
    
    # Find all directories that are not in splits_to_preserve
    splits_to_delete = []
    if os.path.isdir(datasets_dir):
        for item in os.listdir(datasets_dir):
            item_path = os.path.join(datasets_dir, item)
            if os.path.isdir(item_path) or os.path.islink(item_path):
                # Include if not in splits_to_preserve
                if item not in splits_to_preserve:
                    splits_to_delete.append(item)
    
    if not splits_to_delete:
        return None, None
    
    # Sort for consistent display
    splits_to_delete.sort()
    return splits_to_delete, datasets_dir


def ask_about_all_deletions(rollout_cache_base, train_cache_base, splits_being_linked=None):
    """
    Ask user about all deletions (datasets, bddl_files, init_files, rollout_cache, train_cache).
    Returns a dictionary with what to delete, but does not perform any deletions.
    
    Args:
        rollout_cache_base: Base path for rollout cache (from config or command line)
        train_cache_base: Base path for train cache (from config or command line)
        splits_being_linked: Optional set/list of split names that are being linked (should be preserved)
        
    Returns:
        dict: Dictionary with keys 'datasets', 'bddl_files', 'init_files', 'rollout_cache', 'train_cache'
              Each value is a list of (split_name, directory_path) tuples to delete
    """
    deletions = {
        'datasets': [],
        'bddl_files': [],
        'init_files': [],
        'rollout_cache': [],
        'train_cache': []
    }
    
    print(f"\n{'='*60}")
    print(f"Deletion Confirmation - Asking About All Deletions")
    print(f"{'='*60}")
    if splits_being_linked:
        print(f"Splits being linked (will be preserved): {sorted(splits_being_linked)}")
    print(f"{'='*60}\n")
    
    # Ask about datasets deletion (first, before any modifications)
    datasets_splits, datasets_dir = find_datasets_splits(splits_being_linked=splits_being_linked)
    if datasets_splits is not None:
        print(f"\nDatasets splits found (from {datasets_dir}):")
        for split in datasets_splits:
            print(f"  - {split}")
        print(f"\nWill ask for confirmation for each split individually.")
        
        for split in datasets_splits:
            response = input(f"Delete datasets split '{split}'? (y/n): ").strip().lower()
            if response == 'y':
                deletions['datasets'].append((split, datasets_dir))
    else:
        print(f"No non-base/non-linked splits found in datasets")
    
    # Ask about bddl_files deletion
    bddl_splits, bddl_files_dir = find_bddl_files_splits(splits_being_linked=splits_being_linked)
    if bddl_splits is not None:
        print(f"\nBDDL files splits found (from {bddl_files_dir}):")
        for split in bddl_splits:
            print(f"  - {split}")
        print(f"\nWill ask for confirmation for each split individually.")
        
        for split in bddl_splits:
            response = input(f"Delete bddl_files split '{split}'? (y/n): ").strip().lower()
            if response == 'y':
                deletions['bddl_files'].append((split, bddl_files_dir))
    else:
        print(f"No non-base/non-linked splits found in bddl_files")
    
    # Ask about init_files deletion
    init_splits, init_files_dir = find_init_files_splits(splits_being_linked=splits_being_linked)
    if init_splits is not None:
        print(f"\nInit files splits found (from {init_files_dir}):")
        for split in init_splits:
            print(f"  - {split}")
        print(f"\nWill ask for confirmation for each split individually.")
        
        for split in init_splits:
            response = input(f"Delete init_files split '{split}'? (y/n): ").strip().lower()
            if response == 'y':
                deletions['init_files'].append((split, init_files_dir))
    else:
        print(f"No non-base/non-linked splits found in init_files")
    
    # Ask about rollout cache deletion
    rollout_splits, rollout_cache_libero_dir = find_rollout_cache_splits(rollout_cache_base)
    if rollout_splits is not None:
        print(f"\nRollout cache splits found (from {rollout_cache_libero_dir}):")
        for split in rollout_splits:
            print(f"  - {split}")
        print(f"\nWill ask for confirmation for each split individually.")
        
        for split in rollout_splits:
            response = input(f"Delete rollout_cache split '{split}'? (y/n): ").strip().lower()
            if response == 'y':
                deletions['rollout_cache'].append((split, rollout_cache_libero_dir))
    else:
        print(f"No non-base splits found in rollout cache: {rollout_cache_base}")
    
    # Ask about train cache deletion
    train_splits, train_cache_libero_dir = find_train_cache_splits(train_cache_base)
    if train_splits is not None:
        print(f"\nTrain cache splits found (from {train_cache_libero_dir}):")
        for split in train_splits:
            print(f"  - {split}")
        print(f"\nWill ask for confirmation for each split individually.")
        
        for split in train_splits:
            response = input(f"Delete train_cache split '{split}'? (y/n): ").strip().lower()
            if response == 'y':
                deletions['train_cache'].append((split, train_cache_libero_dir))
    else:
        print(f"No non-base splits found in train cache: {train_cache_base}")
    
    return deletions


def ask_about_cache_deletions(rollout_cache_base, train_cache_base):
    """
    Ask user about cache deletions (rollout_cache and train_cache only).
    Returns a dictionary with what to delete, but does not perform any deletions.
    
    Args:
        rollout_cache_base: Base path for rollout cache (from config or command line)
        train_cache_base: Base path for train cache (from config or command line)
        
    Returns:
        dict: Dictionary with keys 'rollout_cache', 'train_cache'
              Each value is a list of (split_name, directory_path) tuples to delete
    """
    deletions = {
        'rollout_cache': [],
        'train_cache': []
    }
    
    print(f"\n{'='*60}")
    print(f"Cache Cleanup - Asking About Cache Deletions")
    print(f"{'='*60}")
    print(f"Rollout cache base: {rollout_cache_base}")
    print(f"Train cache base: {train_cache_base}")
    print(f"{'='*60}\n")
    
    # Ask about rollout cache deletion
    rollout_splits, rollout_cache_libero_dir = find_rollout_cache_splits(rollout_cache_base)
    if rollout_splits is not None:
        print(f"\nRollout cache splits found (from {rollout_cache_libero_dir}):")
        for split in rollout_splits:
            print(f"  - {split}")
        print(f"\nWill ask for confirmation for each split individually.")
        
        for split in rollout_splits:
            response = input(f"Delete rollout_cache split '{split}'? (y/n): ").strip().lower()
            if response == 'y':
                deletions['rollout_cache'].append((split, rollout_cache_libero_dir))
    else:
        print(f"No non-base splits found in rollout cache: {rollout_cache_base}")
    
    # Ask about train cache deletion
    train_splits, train_cache_libero_dir = find_train_cache_splits(train_cache_base)
    if train_splits is not None:
        print(f"\nTrain cache splits found (from {train_cache_libero_dir}):")
        for split in train_splits:
            print(f"  - {split}")
        print(f"\nWill ask for confirmation for each split individually.")
        
        for split in train_splits:
            response = input(f"Delete train_cache split '{split}'? (y/n): ").strip().lower()
            if response == 'y':
                deletions['train_cache'].append((split, train_cache_libero_dir))
    else:
        print(f"No non-base splits found in train cache: {train_cache_base}")
    
    return deletions


def perform_cache_deletions(deletions):
    """
    Perform cache deletions based on the dictionary returned from ask_about_cache_deletions.
    
    Args:
        deletions: Dictionary with keys 'rollout_cache', 'train_cache'
                   Each value is a list of (split_name, directory_path) tuples to delete
    """
    print(f"\n{'='*60}")
    print(f"Performing Cache Deletions")
    print(f"{'='*60}\n")
    
    # Delete rollout cache splits
    if deletions['rollout_cache']:
        deleted_count = 0
        for split, base_dir in deletions['rollout_cache']:
            split_path = os.path.join(base_dir, split)
            try:
                if os.path.islink(split_path):
                    os.remove(split_path)
                    print(f"Removed symlink: {split_path}")
                else:
                    shutil.rmtree(split_path)
                    print(f"Deleted directory: {split_path}")
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {split_path}: {e}")
        print(f"Successfully deleted {deleted_count}/{len(deletions['rollout_cache'])} rollout_cache split(s)\n")
    else:
        print("No rollout_cache splits to delete\n")
    
    # Delete train cache splits
    if deletions['train_cache']:
        deleted_count = 0
        for split, base_dir in deletions['train_cache']:
            split_path = os.path.join(base_dir, split)
            try:
                if os.path.islink(split_path):
                    os.remove(split_path)
                    print(f"Removed symlink: {split_path}")
                else:
                    shutil.rmtree(split_path)
                    print(f"Deleted directory: {split_path}")
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {split_path}: {e}")
        print(f"Successfully deleted {deleted_count}/{len(deletions['train_cache'])} train_cache split(s)\n")
    else:
        print("No train_cache splits to delete\n")


def perform_all_deletions(deletions):
    """
    Perform all deletions based on the dictionary returned from ask_about_all_deletions.
    
    Args:
        deletions: Dictionary with keys 'datasets', 'bddl_files', 'init_files', 'rollout_cache', 'train_cache'
                   Each value is a list of (split_name, directory_path) tuples to delete
    """
    print(f"\n{'='*60}")
    print(f"Performing All Deletions")
    print(f"{'='*60}\n")
    
    # Delete datasets splits (first, before any other modifications)
    if deletions['datasets']:
        deleted_count = 0
        for split, base_dir in deletions['datasets']:
            split_path = os.path.join(base_dir, split)
            try:
                if os.path.islink(split_path):
                    os.remove(split_path)
                    print(f"Removed symlink: {split_path}")
                else:
                    shutil.rmtree(split_path)
                    print(f"Deleted directory: {split_path}")
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {split_path}: {e}")
        print(f"Successfully deleted {deleted_count}/{len(deletions['datasets'])} datasets split(s)\n")
    else:
        print("No datasets splits to delete\n")
    
    # Delete bddl_files splits
    if deletions['bddl_files']:
        deleted_count = 0
        for split, base_dir in deletions['bddl_files']:
            split_path = os.path.join(base_dir, split)
            try:
                if os.path.islink(split_path):
                    os.remove(split_path)
                    print(f"Removed symlink: {split_path}")
                else:
                    shutil.rmtree(split_path)
                    print(f"Deleted directory: {split_path}")
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {split_path}: {e}")
        print(f"Successfully deleted {deleted_count}/{len(deletions['bddl_files'])} bddl_files split(s)\n")
    else:
        print("No bddl_files splits to delete\n")
    
    # Delete init_files splits
    if deletions['init_files']:
        deleted_count = 0
        for split, base_dir in deletions['init_files']:
            split_path = os.path.join(base_dir, split)
            try:
                if os.path.islink(split_path):
                    os.remove(split_path)
                    print(f"Removed symlink: {split_path}")
                else:
                    shutil.rmtree(split_path)
                    print(f"Deleted directory: {split_path}")
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {split_path}: {e}")
        print(f"Successfully deleted {deleted_count}/{len(deletions['init_files'])} init_files split(s)\n")
    else:
        print("No init_files splits to delete\n")
    
    # Delete rollout cache splits
    if deletions['rollout_cache']:
        deleted_count = 0
        for split, base_dir in deletions['rollout_cache']:
            split_path = os.path.join(base_dir, split)
            try:
                if os.path.islink(split_path):
                    os.remove(split_path)
                    print(f"Removed symlink: {split_path}")
                else:
                    shutil.rmtree(split_path)
                    print(f"Deleted directory: {split_path}")
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {split_path}: {e}")
        print(f"Successfully deleted {deleted_count}/{len(deletions['rollout_cache'])} rollout_cache split(s)\n")
    else:
        print("No rollout_cache splits to delete\n")
    
    # Delete train cache splits
    if deletions['train_cache']:
        deleted_count = 0
        for split, base_dir in deletions['train_cache']:
            split_path = os.path.join(base_dir, split)
            try:
                if os.path.islink(split_path):
                    os.remove(split_path)
                    print(f"Removed symlink: {split_path}")
                else:
                    shutil.rmtree(split_path)
                    print(f"Deleted directory: {split_path}")
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {split_path}: {e}")
        print(f"Successfully deleted {deleted_count}/{len(deletions['train_cache'])} train_cache split(s)\n")
    else:
        print("No train_cache splits to delete\n")


def collect_splits_from_run_dirs_simple(run_dirs):
    """
    Collect all split names from multiple run directories without validation.
    Used for filtering cache deletion.
    
    Args:
        run_dirs: List of run directory paths
        
    Returns:
        set: Set of split names found in the run directories, or None if collection fails
    """
    all_splits = set()
    
    for run_dir in run_dirs:
        try:
            run_dir = os.path.abspath(run_dir)
            datasets_source_dir = os.path.join(run_dir, "datasets")
            
            if not os.path.exists(datasets_source_dir) or not os.path.isdir(datasets_source_dir):
                continue  # Skip this run_dir if datasets doesn't exist
            
            # Get all split directories (first level subdirectories in datasets_source_dir)
            split_dirs = [d for d in os.listdir(datasets_source_dir) 
                          if os.path.isdir(os.path.join(datasets_source_dir, d))]
            all_splits.update(split_dirs)
        except Exception:
            # If we can't read a directory, skip it
            continue
    
    return all_splits if all_splits else None


def collect_all_splits_from_run_dirs(run_dirs):
    """
    Collect all split names from multiple run directories and detect duplicates.
    
    Args:
        run_dirs: List of run directory paths
        
    Returns:
        dict: Mapping of split_name -> run_dir (the run directory containing this split)
        
    Raises:
        ValueError: If duplicate split names are found across run directories
    """
    split_to_run_dir = {}
    duplicates = []
    
    for run_dir in run_dirs:
        run_dir = os.path.abspath(run_dir)
        datasets_source_dir = os.path.join(run_dir, "datasets")
        
        if not os.path.exists(datasets_source_dir):
            raise ValueError(f"Datasets directory not found: {datasets_source_dir}")
        
        if not os.path.isdir(datasets_source_dir):
            raise ValueError(f"Source directory is not a directory: {datasets_source_dir}")
        
        # Get all split directories (first level subdirectories in datasets_source_dir)
        split_dirs = [d for d in os.listdir(datasets_source_dir) 
                      if os.path.isdir(os.path.join(datasets_source_dir, d))]
        
        for split_name in split_dirs:
            if split_name in split_to_run_dir:
                # Found a duplicate
                duplicates.append((split_name, split_to_run_dir[split_name], run_dir))
            else:
                split_to_run_dir[split_name] = run_dir
    
    if duplicates:
        error_msg = f"Found {len(duplicates)} duplicate split name(s) across run directories:\n"
        for split_name, first_run_dir, second_run_dir in duplicates:
            error_msg += f"  - Split '{split_name}' found in both:\n"
            error_msg += f"      {first_run_dir}\n"
            error_msg += f"      {second_run_dir}\n"
        error_msg += "\nRun directories must have independent sets of splits."
        raise ValueError(error_msg)
    
    return split_to_run_dir


def link_multiple_run_directories(run_dirs, symlink_extra=False):
    """
    Process multiple run directories sequentially.
    
    Args:
        run_dirs: List of run directory paths
        symlink: If True, create symlinks; if False, copy data
        
    Raises:
        ValueError: If duplicate splits found or if any run directory processing fails
    """
    print(f"\n{'='*60}")
    print(f"Processing Multiple Run Directories")
    print(f"{'='*60}")
    print(f"Number of run directories: {len(run_dirs)}")
    for i, run_dir in enumerate(run_dirs, 1):
        print(f"  {i}. {os.path.abspath(run_dir)}")
    print(f"{'='*60}\n")
    
    # First, validate that there are no duplicate splits across run directories
    print("Validating split names across run directories...")
    split_to_run_dir = collect_all_splits_from_run_dirs(run_dirs)
    print(f"✓ All splits are unique across {len(run_dirs)} run directory(ies)")
    print(f"  Total unique splits: {len(split_to_run_dir)}\n")
    
    # Process each run directory sequentially
    for i, run_dir in enumerate(run_dirs, 1):
        print(f"\n{'='*60}")
        print(f"Processing Run Directory {i}/{len(run_dirs)}")
        print(f"{'='*60}")
        try:
            link_generated_demonstrations(run_dir, symlink_extra=symlink_extra)
        except Exception as e:
            print(f"\nError processing run directory {i}/{len(run_dirs)}: {run_dir}")
            print(f"Error: {e}")
            raise
    
    print(f"\n{'='*60}")
    print(f"All Run Directories Processed Successfully")
    print(f"{'='*60}")
    print(f"Processed {len(run_dirs)} run directory(ies)")
    print(f"Total splits linked: {len(split_to_run_dir)}")


def link_generated_demonstrations(run_dir, symlink_extra=False):
    """
    Link dataset split directories from run directory to libero datasets folder.
    Also copies the BDDL files and init_files files (always hard copied, not symlinked).
    
    Default behavior:
    - Extra splits (ending with _extra): Copied to libero datasets folder
    - View splits (ending with _view): New directory created in libero datasets folder,
      with symlinks to individual files pointing to source splits in libero datasets folder
    
    With --symlink flag:
    - Extra splits: Symlinked to libero datasets folder
    - View splits: New directory created in libero datasets folder, with symlinks to
      individual files pointing to source splits in libero datasets folder (which may be symlinks)
    
    Args:
        run_dir: Path to the run directory containing the datasets folder
        symlink_extra: If True, symlink extra splits instead of copying them. Views are always
                       created as new directories with symlinks to individual files (pointing to
                       source splits in libero datasets folder).
                       Note: BDDL and init_files files are always hard copied regardless of this flag.
    """
    run_dir = os.path.abspath(run_dir)
    datasets_source_dir = os.path.join(run_dir, "datasets")
    
    if not os.path.exists(datasets_source_dir):
        raise ValueError(f"Datasets directory not found: {datasets_source_dir}")
    
    libero_datasets_dir = get_libero_path("datasets")
    
    print(f"Source directory: {datasets_source_dir}")
    print(f"Target directory: {libero_datasets_dir}")
    print()
    
    # Get all split directories (first level subdirectories in datasets_source_dir)
    if not os.path.isdir(datasets_source_dir):
        raise ValueError(f"Source directory is not a directory: {datasets_source_dir}")
    
    split_dirs = [d for d in os.listdir(datasets_source_dir) 
                  if os.path.isdir(os.path.join(datasets_source_dir, d))]
    
    # Identify view splits (end with _view) and regular splits for logging
    view_splits = [s for s in split_dirs if s.endswith('_view')]
    regular_splits = [s for s in split_dirs if not s.endswith('_view')]
    
    if view_splits:
        print(f"Found {len(view_splits)} view split(s): {sorted(view_splits)}")
    if regular_splits:
        print(f"Found {len(regular_splits)} regular split(s): {sorted(regular_splits)}")
    print()
    
    # Check if backup exists in run directory (check for bddl_files or init_files directories)
    bddl_backup_dir = os.path.join(run_dir, "bddl_files")
    init_files_backup_dir = os.path.join(run_dir, "init_files")
    
    # Assert that each BDDL file has a corresponding init file before making any changes
    # This validation works for both regular splits and view splits (views have their own BDDL files)
    if os.path.exists(bddl_backup_dir):
        print(f"\n{'='*60}")
        print(f"Verifying BDDL and init_files correspondence")
        print(f"{'='*60}")
        
        missing_init_files = []
        for split_name in sorted(split_dirs):
            bddl_split_dir = os.path.join(bddl_backup_dir, split_name)
            init_files_split_dir = os.path.join(init_files_backup_dir, split_name)
            
            if not os.path.exists(bddl_split_dir):
                continue  # Skip if BDDL split doesn't exist
            
            if not os.path.exists(init_files_split_dir):
                # If init_files split doesn't exist, all BDDL files in this split are missing init files
                bddl_files = [f for f in os.listdir(bddl_split_dir) if f.endswith('.bddl')]
                for bddl_file in bddl_files:
                    missing_init_files.append((split_name, bddl_file))
                continue
            
            # Check each BDDL file has a corresponding init file
            bddl_files = [f for f in os.listdir(bddl_split_dir) if f.endswith('.bddl')]
            init_files = set(os.listdir(init_files_split_dir))
            
            for bddl_file in bddl_files:
                # Init file should have the same name as BDDL file with .pruned_init extension
                bddl_name = os.path.splitext(bddl_file)[0]
                expected_init_file = f"{bddl_name}.pruned_init"
                if expected_init_file not in init_files:
                    missing_init_files.append((split_name, bddl_file))
        
        if missing_init_files:
            print(f"\nError: Found {len(missing_init_files)} BDDL file(s) without corresponding init files:")
            for split_name, bddl_file in missing_init_files:
                print(f"  - {split_name}/{bddl_file}")
            raise AssertionError(
                f"Each BDDL file must have a corresponding init file. "
                f"Found {len(missing_init_files)} BDDL file(s) missing init files. "
                f"Please ensure all BDDL files have corresponding init files in the run directory."
            )
        else:
            print(f"✓ All BDDL files have corresponding init files")
        print(f"{'='*60}\n")
    
    # First, process extra splits (copy or symlink based on symlink_extra flag)
    # Then process view splits (always symlink, but update internal symlinks to point to libero location)
    for split_name in sorted(regular_splits):
        source_split_dir = os.path.join(datasets_source_dir, split_name)
        target_split_dir = os.path.join(libero_datasets_dir, split_name)
        
        # Remove existing symlink or directory if it exists
        if os.path.islink(target_split_dir):
            print(f"Removing existing symlink: {target_split_dir}")
            os.remove(target_split_dir)
            print(f"Removed existing symlink: {target_split_dir}")
        elif os.path.exists(target_split_dir):
            print(f"Removing existing directory: {target_split_dir}")
            shutil.rmtree(target_split_dir)
            print(f"Removed existing directory: {target_split_dir}")
        
        if symlink_extra:
            # Create symlink to the entire split directory
            source_abs = os.path.abspath(source_split_dir)
            os.symlink(source_abs, target_split_dir)
            print(f"Created symlink for split: {target_split_dir} -> {source_abs}")
        else:
            # Copy the entire split directory
            print(f"Copying split: {source_split_dir} -> {target_split_dir}")
            shutil.copytree(source_split_dir, target_split_dir)
            print(f"Copied split: {source_split_dir} -> {target_split_dir}")
    
    # Now process view splits (create new directory and symlink files individually)
    for split_name in sorted(view_splits):
        source_split_dir = os.path.join(datasets_source_dir, split_name)
        target_split_dir = os.path.join(libero_datasets_dir, split_name)
        
        # Remove existing symlink or directory if it exists
        if os.path.islink(target_split_dir):
            print(f"Removing existing symlink: {target_split_dir}")
            os.remove(target_split_dir)
        elif os.path.exists(target_split_dir):
            print(f"Removing existing directory: {target_split_dir}")
            shutil.rmtree(target_split_dir)
        
        # Create new directory for the view
        os.makedirs(target_split_dir, exist_ok=True)
        print(f"Created view split directory: {target_split_dir}")
        
        # Walk through source view directory and recreate symlinks pointing to libero location
        for item in os.listdir(source_split_dir):
            source_item_path = os.path.join(source_split_dir, item)
            target_item_path = os.path.join(target_split_dir, item)
            
            if os.path.islink(source_item_path):
                # It's a symlink - create new symlink pointing to libero location
                link_target = os.readlink(source_item_path)
                link_target_abs = None
                
                # Resolve the link target to absolute path
                if os.path.isabs(link_target):
                    link_target_abs = link_target
                else:
                    link_target_abs = os.path.abspath(os.path.join(source_split_dir, link_target))
                
                # Determine the target location in libero_datasets_dir
                new_target_abs = None
                
                # Check if this symlink points to a source split in run_dir
                if link_target_abs.startswith(datasets_source_dir):
                    # Extract the relative path from datasets_source_dir
                    # e.g., run_dir/datasets/libero_spatial_extra/task_demo.hdf5
                    # -> libero_datasets_dir/libero_spatial_extra/task_demo.hdf5
                    rel_path_from_datasets = os.path.relpath(link_target_abs, datasets_source_dir)
                    new_target_abs = os.path.join(libero_datasets_dir, rel_path_from_datasets)
                elif link_target_abs.startswith(libero_datasets_dir):
                    # Already points to libero location (base split case)
                    new_target_abs = link_target_abs
                else:
                    # Unexpected - symlink points somewhere else
                    raise ValueError(
                        f"Symlink {source_item_path} points to unexpected location: {link_target_abs}. "
                        f"Expected symlink to point to either {datasets_source_dir} or {libero_datasets_dir}"
                    )
                
                # Compute relative path from target_item_path to new_target_abs
                new_target_rel = os.path.relpath(new_target_abs, os.path.dirname(target_item_path))
                
                # Create the symlink
                try:
                    os.symlink(new_target_rel, target_item_path)
                except Exception as e:
                    print(f"Warning: Could not create symlink {target_item_path}: {e}")
    
    extra_action = "symlinked" if symlink_extra else "copied"
    print(f"\nExtra splits {extra_action} successfully!")
    print(f"View splits symlinked successfully!")
    
    if not os.path.exists(bddl_backup_dir) and not os.path.exists(init_files_backup_dir):
        print(f"\nWarning: Backup directories not found in run directory: {run_dir}")
        print("This run directory was created with --no-backup flag, so linking is not available.")
        print("Skipping BDDL and init_files file linking.")
        return
    
    # Now copy the BDDL files from the run directory (always hard copied)
    # This includes BDDL files for view splits (views have their own BDDL files copied from source splits)
    if os.path.exists(bddl_backup_dir):
        print(f"\n{'='*60}")
        print(f"Copying BDDL files")
        print(f"{'='*60}")
        
        libero_bddl_dir = get_libero_path("bddl_files")
        
        print(f"BDDL source directory: {bddl_backup_dir}")
        print(f"BDDL target directory: {libero_bddl_dir}")
        print()
        
        # Remove existing target split directories before copying
        # This handles both regular splits and view splits
        for split_name in sorted(split_dirs):
            target_bddl_split_dir = os.path.join(libero_bddl_dir, split_name)
            if os.path.islink(target_bddl_split_dir):
                print(f"Removing existing symlink: {target_bddl_split_dir}")
                os.remove(target_bddl_split_dir)
            elif os.path.exists(target_bddl_split_dir):
                print(f"Removing existing directory: {target_bddl_split_dir}")
                shutil.rmtree(target_bddl_split_dir)
        
        # For each split that was linked, create the BDDL split directory
        # BDDL files are always hard copied (not symlinked)
        # This includes view splits, which have their own BDDL files
        for split_name in sorted(split_dirs):
            source_bddl_split_dir = os.path.join(bddl_backup_dir, split_name)
            target_bddl_split_dir = os.path.join(libero_bddl_dir, split_name)
            
            if not os.path.exists(source_bddl_split_dir):
                print(f"Warning: BDDL source split directory does not exist: {source_bddl_split_dir}")
                continue
            
            # Always copy (hard copy) for BDDL files
            shutil.copytree(source_bddl_split_dir, target_bddl_split_dir)
            split_type = "view split" if split_name.endswith('_view') else "split"
            print(f"Copied BDDL directory for {split_type}: {source_bddl_split_dir} -> {target_bddl_split_dir}")
        
        print(f"\nAll BDDL files copied successfully!")
    else:
        print(f"\nWarning: BDDL directory not found: {bddl_backup_dir}")
        print("Skipping BDDL file linking.")
    
    # Now copy the init_files files from the run directory (always hard copied)
    # This includes init_files for view splits (views have their own init_files copied from source splits)
    if os.path.exists(init_files_backup_dir):
        print(f"\n{'='*60}")
        print(f"Copying init_files files")
        print(f"{'='*60}")
        
        libero_init_files_dir = get_libero_path("init_states")
        
        print(f"Init_files source directory: {init_files_backup_dir}")
        print(f"Init_files target directory: {libero_init_files_dir}")
        print()
        
        # Remove existing target split directories before copying
        # This handles both regular splits and view splits
        for split_name in sorted(split_dirs):
            target_init_files_split_dir = os.path.join(libero_init_files_dir, split_name)
            if os.path.islink(target_init_files_split_dir):
                print(f"Removing existing symlink: {target_init_files_split_dir}")
                os.remove(target_init_files_split_dir)
            elif os.path.exists(target_init_files_split_dir):
                print(f"Removing existing directory: {target_init_files_split_dir}")
                shutil.rmtree(target_init_files_split_dir)
        
        # For each split that was linked, create the init_files split directory
        # Init_files files are always hard copied (not symlinked)
        # This includes view splits, which have their own init_files
        for split_name in sorted(split_dirs):
            source_init_files_split_dir = os.path.join(init_files_backup_dir, split_name)
            target_init_files_split_dir = os.path.join(libero_init_files_dir, split_name)
            
            if not os.path.exists(source_init_files_split_dir):
                print(f"Warning: Init_files source split directory does not exist: {source_init_files_split_dir}")
                continue
            
            # Always copy (hard copy) for init_files files
            shutil.copytree(source_init_files_split_dir, target_init_files_split_dir)
            split_type = "view split" if split_name.endswith('_view') else "split"
            print(f"Copied init_files directory for {split_type}: {source_init_files_split_dir} -> {target_init_files_split_dir}")
        
        print(f"\nAll init_files files copied successfully!")
    else:
        print(f"\nWarning: Init_files backup directory not found: {init_files_backup_dir}")
        print("Skipping init_files file linking.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Symlink or copy generated demonstrations to libero datasets folder")
    parser.add_argument("--run-dirs", type=str, required=False, nargs='+', help="Path(s) to the run directory(ies) containing the datasets folder. Can specify multiple directories.")
    parser.add_argument("--symlink", action="store_true", help="Symlink extra splits instead of copying them (default: copy extra, symlink views)")
    parser.add_argument("--no-delete-caches", dest="delete_caches", action="store_false", default=True, help="Disable cache deletion. By default, cache deletion is enabled: datasets, bddl_files, init_files, rollout_cache, and train_cache (all non-base/non-linked splits, individual confirmation for each split).")
    parser.add_argument("--cache-cleanup", action="store_true", help="Only perform cache cleanup (rollout_cache and train_cache). Does not require --run-dirs. Exits after cleanup.")
    parser.add_argument("--rollout-cache-base", type=str, default=None, help="Base path for rollout cache (defaults to value from config)")
    parser.add_argument("--train-cache-base", type=str, default=None, help="Base path for train cache (defaults to value from config)")
    parser.add_argument("--split-name", type=str, default="libero_all_extra", help="Deprecated: not used for cache deletion. Kept for backward compatibility.")
    parser.add_argument("--config-path", type=str, default=None, help="Path to libero_defaults.yaml config file (default: searches in config/task/)")
    args = parser.parse_args()
    
    # Handle cache cleanup mode (early exit)
    if args.cache_cleanup:
        rollout_cache_base, train_cache_base = find_config_and_get_cache_paths(
            config_path_arg=args.config_path,
            rollout_cache_base_arg=args.rollout_cache_base,
            train_cache_base_arg=args.train_cache_base
        )
        
        # Ask about cache deletions
        deletions = ask_about_cache_deletions(rollout_cache_base, train_cache_base)
        
        # Perform cache deletions
        perform_cache_deletions(deletions)
        
        print(f"\n{'='*60}")
        print(f"Cache cleanup complete. Exiting.")
        print(f"{'='*60}\n")
        exit(0)
    
    # Collect splits from run directories if provided (needed for bddl_files/init_files deletion and cache deletion)
    splits_being_linked = None
    if args.run_dirs:
        # Collect splits from run directories
        # Use simple collection without validation (we'll validate later when linking)
        splits_being_linked = collect_splits_from_run_dirs_simple(args.run_dirs)
        if splits_being_linked:
            print(f"Collected {len(splits_being_linked)} split(s) from run directories: {sorted(splits_being_linked)}")
        else:
            print("Warning: Could not collect splits from run directories")
    
    # Handle deletions - ask about everything first, then perform all deletions
    if args.delete_caches:
        rollout_cache_base, train_cache_base = find_config_and_get_cache_paths(
            config_path_arg=args.config_path,
            rollout_cache_base_arg=args.rollout_cache_base,
            train_cache_base_arg=args.train_cache_base
        )
        
        # Ask about all deletions first (before performing any deletions)
        deletions = ask_about_all_deletions(rollout_cache_base, train_cache_base, splits_being_linked=splits_being_linked)
        
        # Perform all deletions
        perform_all_deletions(deletions)
    
    # Handle symlinking (original functionality) - do this after all deletions
    if args.run_dirs:
        # args.run_dirs is a list when nargs='+' is used
        # Can be a list with one or more elements
        link_multiple_run_directories(args.run_dirs, symlink_extra=args.symlink)
