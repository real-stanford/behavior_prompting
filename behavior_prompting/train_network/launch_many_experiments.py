"""
Script to launch experiments sequentially with wandb tracking.

Usage:
    # Check status only
    python launch_many_experiments.py --runs-files experiments/draw/baselines_and_ablations.sh --gpus 0,1,2,3 --status-only --seeds 0,1,2
    
    # Launch experiments (single runs file)
    python launch_many_experiments.py --runs-files experiments/draw/baselines_and_ablations.sh --gpus 0,1,2,3 --seeds 0,1,2

    # Use a non-default Wandb project for status checks and for each run (Hydra logging.project override)
    python launch_many_experiments.py ... --project my_wandb_project
    
    # Launch experiments (multiple runs files; same as merging them into one)
    python launch_many_experiments.py --runs-files experiments/draw/a.sh experiments/draw/b.sh --gpus 0,1,2,3 --seeds 0,1,2
    
    # Dry run (print commands without executing)
    python launch_many_experiments.py --runs-files experiments/draw/baselines_and_ablations.sh --gpus 0,1,2,3 --dry-run --seeds 0,1,2
"""

import argparse
import os
import random
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

import wandb


def parse_command_line(command: str) -> Dict[str, str]:
    """Parse a command line to extract key-value pairs."""
    # Extract exp_name and group_tag from the command
    exp_name_match = re.search(r'exp_name="([^"]+)"', command)
    group_tag_match = re.search(r'group_tag="([^"]+)"', command)
    
    exp_name = exp_name_match.group(1) if exp_name_match else None
    group_tag = group_tag_match.group(1) if group_tag_match else None
    
    return {
        'command': command,
        'exp_name': exp_name,
        'group_tag': group_tag
    }


def read_runs_file(runs_file: str) -> List[Dict[str, str]]:
    """Read and parse runs from a file."""
    runs = []
    with open(runs_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parsed = parse_command_line(line)
                if parsed['exp_name'] and parsed['group_tag']:
                    runs.append(parsed)
    return runs


def get_wandb_runs_status(group_tag: str, exp_name: str, seeds: List[int], project: str = "behavior_prompting") -> Dict[int, str]:
    """
    Query wandb to get the status of runs for a given experiment.
    
    Returns:
        Dict mapping seed to status: 'finished', 'running', or 'not_found'
    """
    try:
        api = wandb.Api()
        status_dict = {seed: 'not_found' for seed in seeds}
        
        # Filter runs by tags (group_tag and exp_name) to only get relevant runs
        try:
            filters = {
                "$and": [
                    {"tags": {"$in": [group_tag]}},
                    {"tags": {"$in": [exp_name]}}
                ]
            }
            
            runs = api.runs(project, filters=filters, order="-created_at")
            
            for run in runs:
                # Check if run has the matching seed in config
                try:
                    config = run.config
                    run_seed = None
                    
                    # Try to access config.training.seed
                    if isinstance(config, dict):
                        training_config = config.get('training', {})
                        if isinstance(training_config, dict):
                            run_seed = training_config.get('seed')
                    elif hasattr(config, 'training'):
                        training_attr = getattr(config, 'training', None)
                        if hasattr(training_attr, 'seed'):
                            run_seed = getattr(training_attr, 'seed')
                        elif isinstance(training_attr, dict):
                            run_seed = training_attr.get('seed')
                    
                    if run_seed is None:
                        continue
                    
                    # Check if this seed is in our list
                    if run_seed not in seeds:
                        continue
                    
                    # If we already found a more recent run for this seed, skip
                    if status_dict.get(run_seed) in ['finished', 'running']:
                        continue
                    
                    # Update status
                    state = run.state
                    if state == 'finished':
                        status_dict[run_seed] = 'finished'
                    elif state == 'running':
                        status_dict[run_seed] = 'running'
                    # For crashed, failed, etc., we leave as 'not_found' so it can be rerun
                    
                except Exception as e:
                    # Skip runs where we can't read config
                    continue
                    
        except Exception as e:
            print(f"Warning: Failed to query wandb runs: {e}")
            # Return all as not_found
            return {seed: 'not_found' for seed in seeds}
        
        return status_dict
    except Exception as e:
        print(f"Error: Failed to initialize wandb API: {e}")
        print("Make sure wandb is installed and you're logged in (run 'wandb login')")
        # Return all as not_found so script can continue
        return {seed: 'not_found' for seed in seeds}


def print_status(runs: List[Dict[str, str]], seeds: List[int], project: str = "behavior_prompting"):
    """Print the status of all experiments."""
    print(f"\n{'='*80}")
    print(f"Experiment Status (seeds: {seeds})")
    print(f"{'='*80}\n")
    
    for run_info in runs:
        group_tag = run_info['group_tag']
        exp_name = run_info['exp_name']
        exp_id = f"{group_tag}/{exp_name}"
        
        print(f"Experiment: {exp_id}")
        status_dict = get_wandb_runs_status(group_tag, exp_name, seeds, project=project)
        
        for seed in seeds:
            status = status_dict.get(seed, 'not_found')
            status_symbol = {
                'finished': '✓',
                'running': '⟳',
                'not_found': '○'
            }.get(status, '?')
            print(f"  Seed {seed}: {status_symbol} {status}")
        
        print()


def get_next_run_to_launch(
    runs: List[Dict[str, str]],
    seeds: List[int],
    project: str = "behavior_prompting",
    random_order: bool = True,
    prioritize_all_seeds: bool = False,
) -> Optional[Tuple[Dict[str, str], int]]:
    """
    Find the next run that needs to be launched.
    Returns (run_info, seed) or None if all runs are finished.
    
    Default sequential mode iterates over seeds first (outer loop), then runs (inner loop).
    This ensures all experiments for one seed complete before moving to the next seed.

    If prioritize_all_seeds is True, iterates runs first (outer loop), then seeds (inner loop),
    so all seeds for one experiment complete before moving to the next experiment.
    
    If random_order is True, checks runs in random order and returns first valid one.
    Otherwise, goes through runs in order.
    """
    # Create a shuffled copy of runs if random_order is True
    runs_to_check = runs.copy()
    if random_order:
        random.shuffle(runs_to_check)
    
    if prioritize_all_seeds:
        # Iterate over runs first (outer loop), then seeds (inner loop)
        for run_info in runs_to_check:
            group_tag = run_info['group_tag']
            exp_name = run_info['exp_name']
            for seed in seeds:
                status_dict = get_wandb_runs_status(group_tag, exp_name, [seed], project=project)
                
                status = status_dict.get(seed, 'not_found')
                if status == 'not_found':
                    # Double-check it's not running (race condition protection)
                    time.sleep(0.5)  # Small delay to avoid rapid queries
                    status_dict = get_wandb_runs_status(group_tag, exp_name, [seed], project=project)
                    if status_dict.get(seed, 'not_found') == 'not_found':
                        return (run_info, seed)
    else:
        # Iterate over seeds first (outer loop), then runs (inner loop)
        for seed in seeds:
            for run_info in runs_to_check:
                group_tag = run_info['group_tag']
                exp_name = run_info['exp_name']
                status_dict = get_wandb_runs_status(group_tag, exp_name, [seed], project=project)
                
                status = status_dict.get(seed, 'not_found')
                if status == 'not_found':
                    # Double-check it's not running (race condition protection)
                    time.sleep(0.5)  # Small delay to avoid rapid queries
                    status_dict = get_wandb_runs_status(group_tag, exp_name, [seed], project=project)
                    if status_dict.get(seed, 'not_found') == 'not_found':
                        return (run_info, seed)
    
    return None


def build_command(
    run_info: Dict[str, str],
    seed: int,
    gpus: str,
    project: str = "behavior_prompting",
) -> str:
    """Build the command to run with the specified seed and GPU IDs."""
    command = run_info['command']
    
    # Count number of GPUs
    num_gpus = len([gpu for gpu in gpus.split(',') if gpu.strip()])
    
    # Replace shell variables: $GPUS, $NUM_PROCESSES, $SEED
    command = command.replace('$GPUS', gpus)
    command = command.replace('$NUM_PROCESSES', str(num_gpus))
    command = command.replace('$SEED', str(seed))

    # Match Wandb project used for status queries (--project) with training (policy_base logging.project).
    command = f"{command.rstrip()} logging.project={shlex.quote(project)}"
    
    return command


def launch_run(run_info: Dict[str, str], seed: int, train_network_root: Path, dry_run: bool = False, gpus: str = "", project: str = "behavior_prompting"):
    """Launch a single run."""
    group_tag = run_info['group_tag']
    exp_name = run_info['exp_name']
    exp_id = f"{group_tag}/{exp_name}"
    
    # Final check before launching (race condition protection)
    status_dict = get_wandb_runs_status(group_tag, exp_name, [seed], project=project)
    if status_dict.get(seed) in ['finished', 'running']:
        print(f"\nSkipping {exp_id} (seed={seed}): Already {status_dict.get(seed)}")
        return True  # Consider it successful since it's already running/finished
    
    print(f"\n{'='*80}")
    if dry_run:
        print(f"[DRY RUN] Would launch: {exp_id} (seed={seed})")
    else:
        print(f"Launching: {exp_id} (seed={seed})")
    print(f"{'='*80}\n")
    
    command = build_command(run_info, seed, gpus=gpus, project=project)
    print(f"Command: {command}\n")
    
    if dry_run:
        print("[DRY RUN] Command would be executed (skipped)")
        input("Press Enter to continue to next experiment...")
        return True
    
    # Change to train_network root directory before executing
    # (train.py in commands is relative to train_network_root)
    original_cwd = os.getcwd()
    try:
        os.chdir(train_network_root)
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            cwd=str(train_network_root)
        )
        print(f"\n✓ Completed: {exp_id} (seed={seed})")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed: {exp_id} (seed={seed}) - Exit code: {e.returncode}")
        return False
    finally:
        # Restore original directory
        os.chdir(original_cwd)


def load_runs(runs_file_paths: List[str]) -> List[Dict[str, str]]:
    """Load and merge runs from one or more files."""
    runs = []
    for runs_file_path in runs_file_paths:
        runs_file = Path(runs_file_path).resolve()
        if not runs_file.exists():
            print(f"Error: Runs file not found: {runs_file}")
            sys.exit(1)
        file_runs = read_runs_file(str(runs_file))
        runs.extend(file_runs)
        if len(runs_file_paths) > 1 and file_runs:
            print(f"  {runs_file}: {len(file_runs)} runs")
    return runs


def main():
    parser = argparse.ArgumentParser(
        description="Launch experiments sequentially with wandb tracking"
    )
    parser.add_argument(
        '--runs-files',
        type=str,
        nargs='+',
        required=True,
        help='Path(s) to file(s) containing run commands (ex: experiments/draw/a.sh). Multiple files are merged in order, as if concatenated.'
    )
    parser.add_argument(
        '--seeds',
        type=str,
        default='0,1,2',
        help='Comma-separated list of seeds (default: 0,1,2)'
    )
    parser.add_argument(
        '--status-only',
        action='store_true',
        help='Only print status, do not launch runs'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='behavior_prompting',
        help='Wandb project name (default: behavior_prompting)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without executing them'
    )
    parser.add_argument(
        '--gpus',
        type=str,
        default=None,
        help='GPU IDs to use (e.g., "0,1,2,3"). Replaces --gpu_ids in commands. Required unless --status-only is set.'
    )
    parser.add_argument(
        '--random-order',
        action='store_true',
        help='Run experiments in random order (default: sequential order)'
    )
    parser.add_argument(
        '--prioritize-all-seeds',
        action='store_true',
        help='Run all seeds for each experiment before moving to the next experiment (only valid without --random-order)'
    )

    args = parser.parse_args()

    if args.random_order and args.prioritize_all_seeds:
        parser.error('--prioritize-all-seeds cannot be used with --random-order')

    if not args.status_only and args.gpus is None:
        parser.error('--gpus is required unless --status-only is set')

    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    # Get train_network root directory (absolute path)
    # Script is now in train_network root, so __file__ parent is the root
    train_network_root = Path(__file__).parent.resolve()

    # Initial load of runs files
    runs = load_runs(args.runs_files)

    if not runs:
        print(f"Error: No valid runs found in any of the specified files")
        sys.exit(1)

    print(f"Found {len(runs)} experiments" + (" total" if len(args.runs_files) > 1 else ""))
    print(f"Seeds: {seeds}")
    print(f"Wandb project: {args.project}")
    print(f"GPUs: {args.gpus}")
    print()

    # Print status
    print_status(runs, seeds, project=args.project)

    if args.status_only:
        print("Status-only mode: Exiting without launching runs.")
        return

    # Launch runs
    print(f"\n{'='*80}")
    random_order = args.random_order
    prioritize_all_seeds = args.prioritize_all_seeds
    if random_order:
        order_mode = "random"
    elif prioritize_all_seeds:
        order_mode = "sequential (experiment-first: all seeds per experiment)"
    else:
        order_mode = "sequential (seed-first)"
    if args.dry_run:
        print(f"Starting {order_mode} launch (DRY RUN mode)...")
    else:
        print(f"Starting {order_mode} launch...")
    print(f"{'='*80}\n")

    while True:
        # Reload runs files before each decision so edits made while an experiment
        # was running take effect for the next experiment to launch.
        runs = load_runs(args.runs_files)

        next_run = get_next_run_to_launch(
            runs,
            seeds,
            project=args.project,
            random_order=random_order,
            prioritize_all_seeds=prioritize_all_seeds,
        )

        if next_run is None:
            print("\n✓ All experiments completed!")
            break

        run_info, seed = next_run
        success = launch_run(run_info, seed, train_network_root, dry_run=args.dry_run, gpus=args.gpus, project=args.project)

        if not success and not args.dry_run:
            print(f"\nWarning: Run failed. Continuing with next run...")

        # Wait a bit before checking status again (skip in dry-run mode)
        if not args.dry_run:
            time.sleep(2)
        else:
            # In dry-run mode, just continue to next run
            time.sleep(0.1)

        # Print updated status after each run
        print_status(runs, seeds, project=args.project)


if __name__ == "__main__":
    main()

