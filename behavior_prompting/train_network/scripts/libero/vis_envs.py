#!/usr/bin/env python3
"""
Visualize environments from BDDL files by initializing and capturing agentview images.

Usage:
    python vis_envs.py <path_to_bddl_file_or_folder> [--grid]
    
If a single BDDL file is provided, saves a single image.
If a folder is provided:
    - By default: saves separate images to tmp_vis_envs/<folder_name>/
    - With --grid flag: creates a grid image of all BDDL files in the folder
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torchvision
from PIL import Image
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import h5py

from behavior_prompting.train_network import fix_robosuite_log_permission_issue
fix_robosuite_log_permission_issue()

from behavior_prompting.train_network.utils.libero_util import bddl_to_hdf5
from libero.libero.envs import OffScreenRenderEnv


def make_grid(images, nrow=8, padding=2, normalize=False, pad_value=0):
    """Make a grid of images. Make sure images is a 4D tensor in the shape of (B x C x H x W)) or a list of torch tensors."""
    grid_image = torchvision.utils.make_grid(images, nrow=nrow, padding=padding, normalize=normalize, pad_value=pad_value).permute(1, 2, 0)
    return grid_image


def visualize_single_bddl(bddl_path, camera_height=128, camera_width=128, num_noop_steps=10, final_state=None):
    """
    Visualize a single BDDL environment by initializing it and taking a number of no-op steps,
    or by jumping to a specific mujoco state.

    Args:
        bddl_path: Path to the BDDL file
        camera_height: Camera height for rendering
        camera_width: Camera width for rendering
        num_noop_steps: Number of no-op steps after reset (ignored when final_state is provided)
        final_state: Optional flattened mujoco state array. If provided, the environment is set
            to this state instead of running noop steps.

    Returns:
        torch.Tensor: Image as (C, H, W) tensor
    """
    bddl_path = Path(bddl_path).resolve()

    if not bddl_path.exists():
        raise FileNotFoundError(f"BDDL file not found: {bddl_path}")

    if not bddl_path.suffix == ".bddl":
        raise ValueError(f"File is not a BDDL file: {bddl_path}")

    # Create environment
    env_args = {
        "bddl_file_name": str(bddl_path),
        "camera_heights": camera_height,
        "camera_widths": camera_width
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # Fix random seed for reproducibility

    # Initialize environment
    env.reset()

    if final_state is not None:
        # Jump to the provided mujoco state and render from there
        obs = env.regenerate_obs_from_state(final_state)
    else:
        # Take num_noop_steps steps with no action
        obs = env.reset()
        for _ in range(num_noop_steps):
            obs, _, _, _ = env.step([0.] * 7)

    # Capture the image
    image = torch.from_numpy(obs["agentview_image"]).permute(2, 0, 1)

    env.close()

    return image


def _process_bddl_worker(args):
    """
    Worker function for multiprocessing. Processes a single BDDL file.

    Args:
        args: Tuple of (bddl_path, camera_height, camera_width, num_noop_steps)

    Returns:
        tuple: (bddl_path, image_tensor, None) on success, (bddl_path, None, error_msg) on failure
    """
    bddl_path, camera_height, camera_width, num_noop_steps, use_final_state = args
    try:
        final_state = None
        if use_final_state:
            hdf5_path = bddl_to_hdf5(bddl_path)
            with h5py.File(hdf5_path, "r") as f:
                first_demo = sorted(f["data"].keys())[0]
                final_state = f[f"data/{first_demo}/states"][-1]

        image = visualize_single_bddl(
            bddl_path,
            camera_height=camera_height,
            camera_width=camera_width,
            num_noop_steps=num_noop_steps,
            final_state=final_state,
        )
        return (bddl_path, image, None)
    except Exception as e:
        return (bddl_path, None, str(e))


def find_bddl_files(input_path):
    """
    Find BDDL files from input path (file or folder).
    
    Args:
        input_path: Path to a BDDL file or folder containing BDDL files
        
    Returns:
        list: List of Path objects to BDDL files
    """
    input_path = Path(input_path).resolve()
    
    if not input_path.exists():
        raise FileNotFoundError(f"Path not found: {input_path}")
    
    bddl_files = []
    
    if input_path.is_file():
        if input_path.suffix == ".bddl":
            bddl_files.append(input_path)
        else:
            raise ValueError(f"File is not a BDDL file: {input_path}")
    elif input_path.is_dir():
        # Find all BDDL files recursively
        bddl_files = list(input_path.glob("**/*.bddl"))
        if not bddl_files:
            raise ValueError(f"No BDDL files found in directory: {input_path}")
    else:
        raise ValueError(f"Path is neither a file nor a directory: {input_path}")
    
    return sorted(bddl_files)


def save_image(image_tensor, output_path):
    """
    Save a single image tensor to file.
    
    Args:
        image_tensor: torch.Tensor of shape (C, H, W)
        output_path: Path to save the image
    """
    # Convert to numpy and flip vertically (images are typically flipped)
    image_array = image_tensor.permute(1, 2, 0).numpy()[::-1]
    
    # Ensure values are in [0, 255] range for PIL Image
    if image_array.max() <= 1.0:
        image_array = (image_array * 255).astype(np.uint8)
    else:
        image_array = image_array.astype(np.uint8)
    
    # Convert to PIL Image and save
    image_pil = Image.fromarray(image_array)
    image_pil.save(output_path)


def visualize_envs(input_path, output_dir=None, nrow=10, camera_height=128, camera_width=128, num_workers=None, grid=False, num_noop_steps=10, only_with_demonstrations=False, final_state=False, max_envs=None):
    """
    Visualize environments from BDDL files.
    
    Args:
        input_path: Path to a BDDL file or folder containing BDDL files
        output_dir: Directory to save the output image(s). If None, uses tmp_vis_envs in the same folder as this script.
        nrow: Number of images per row in the grid (only used for folder mode with grid=True)
        camera_height: Camera height for rendering
        camera_width: Camera width for rendering
        num_workers: Number of parallel workers. If None, uses cpu_count(). Only used for folder mode.
        grid: If True, create a grid image. If False (default), save separate images when folder is provided.
        num_noop_steps: Number of no-op steps to take after reset before capturing the image.
    """
    input_path = Path(input_path).resolve()
    
    # Find BDDL files
    bddl_files = find_bddl_files(input_path)
    
    if only_with_demonstrations or final_state:
        bddl_files = [f for f in bddl_files if Path(bddl_to_hdf5(str(f))).exists()]
        print(f"Found {len(bddl_files)} BDDL file(s) with associated HDF5 demonstrations")
    else:
        print(f"Found {len(bddl_files)} BDDL file(s)")

    if max_envs is not None:
        bddl_files = bddl_files[:max_envs]
        print(f"Limiting to first {len(bddl_files)} environment(s)")
    
    # Determine output directory
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "tmp_vis_envs"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Single file mode
    if len(bddl_files) == 1:
        bddl_file = bddl_files[0]
        print(f"Visualizing single BDDL file: {bddl_file}")
        
        single_final_state = None
        if final_state:
            hdf5_path = bddl_to_hdf5(str(bddl_file))
            with h5py.File(hdf5_path, "r") as f:
                first_demo = sorted(f["data"].keys())[0]
                single_final_state = f[f"data/{first_demo}/states"][-1]

        image = visualize_single_bddl(
            bddl_file,
            camera_height=camera_height,
            camera_width=camera_width,
            num_noop_steps=num_noop_steps,
            final_state=single_final_state,
        )
        
        # Save single image
        output_filename = f"{bddl_file.stem}_env.png"
        output_path = output_dir / output_filename
        save_image(image, output_path)
        
        print(f"Saved visualization to: {output_path.resolve()}")
        return output_path
    
    # Folder mode - process all BDDL files
    else:
        print(f"Visualizing {len(bddl_files)} BDDL files from folder")
        
        # Prepare arguments for worker function
        if num_workers is None:
            num_workers = os.cpu_count()
        num_workers = min(num_workers, len(bddl_files))  # Don't use more workers than files
        
        worker_args = [
            (str(bddl_file), camera_height, camera_width, num_noop_steps, final_state)
            for bddl_file in bddl_files
        ]
        
        # For non-grid mode, set up output directory upfront so we can save as results arrive
        if not grid:
            folder_name = input_path.name if input_path.is_dir() else input_path.parent.name
            separate_output_dir = output_dir / folder_name
            if separate_output_dir.exists():
                import shutil
                shutil.rmtree(separate_output_dir)
            separate_output_dir.mkdir(parents=True)
            print(f"Writing images to: {separate_output_dir.resolve()}")

        # Process BDDL files
        images_dict = {}  # only populated in grid mode
        failed_files = []
        num_saved = 0

        def _handle_result(bddl_path_str, image, error_msg):
            nonlocal num_saved
            bddl_path = Path(bddl_path_str)
            if image is not None:
                if grid:
                    images_dict[bddl_path] = image
                else:
                    output_filename = f"{bddl_path.stem}_env.png"
                    save_image(image, separate_output_dir / output_filename)
                    num_saved += 1
            else:
                failed_files.append((bddl_path, error_msg))

        if num_workers == 1:
            for args in tqdm(worker_args, desc="Processing BDDL files"):
                _handle_result(*_process_bddl_worker(args))
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                futures = [pool.submit(_process_bddl_worker, args) for args in worker_args]
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Processing BDDL files",
                ):
                    _handle_result(*future.result())

        # Report failures
        if failed_files:
            print(f"Warning: Failed to visualize {len(failed_files)} BDDL file(s):")
            for failed_file, error_msg in failed_files:
                if error_msg:
                    print(f"  - {failed_file}: {error_msg}")
                else:
                    print(f"  - {failed_file}")

        if grid:
            if not images_dict:
                raise RuntimeError("No images were successfully generated")

            # Sort images by BDDL file path to maintain consistent order
            sorted_bddl_files = sorted(images_dict.keys())
            images = [images_dict[bddl_file] for bddl_file in sorted_bddl_files]

            print(f"Successfully processed {len(images)} out of {len(bddl_files)} BDDL files")

            # Create grid
            grid_image = make_grid(images, nrow=nrow, padding=2, pad_value=0)

            # Convert to numpy and flip vertically
            grid_array = grid_image.numpy()[::-1]

            # Ensure values are in [0, 255] range for PIL Image
            if grid_array.max() <= 1.0:
                grid_array = (grid_array * 255).astype(np.uint8)
            else:
                grid_array = grid_array.astype(np.uint8)

            # Save grid image
            folder_name = input_path.name if input_path.is_dir() else input_path.parent.name
            output_filename = f"{folder_name}_grid.png"
            output_path = output_dir / output_filename

            # Convert to PIL Image and save
            grid_pil = Image.fromarray(grid_array)
            grid_pil.save(output_path)

            print(f"Saved grid visualization to: {output_path.resolve()}")
            return output_path
        else:
            if num_saved == 0:
                raise RuntimeError("No images were successfully generated")
            
            print(f"Saved {num_saved} separate images to: {separate_output_dir.resolve()}")
            return separate_output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Visualize environments from BDDL files"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to a BDDL file or folder containing BDDL files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the output image(s). Default: tmp_vis_envs in script directory"
    )
    parser.add_argument(
        "--nrow",
        type=int,
        default=10,
        help="Number of images per row in the grid (default: 10, only used for folder mode)"
    )
    parser.add_argument(
        "--camera_height",
        type=int,
        default=128,
        help="Camera height for rendering (default: 128)"
    )
    parser.add_argument(
        "--camera_width",
        type=int,
        default=128,
        help="Camera width for rendering (default: 128)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers for processing BDDL files. Default: number of CPU cores (only used for folder mode)"
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Create a grid image when processing a folder. By default, saves separate images to tmp_vis_envs/<folder_name>/"
    )
    parser.add_argument(
        "--num_noop_steps",
        type=int,
        default=10,
        help="Number of no-action steps to take after reset before capturing the image (default: 10)",
    )
    parser.add_argument(
        "--only-with-demonstrations",
        action="store_true",
        help="Only generate images for environments that have an associated HDF5 demonstration file",
    )
    parser.add_argument(
        "--final-state",
        action="store_true",
        help="Show the final state from the first demonstration instead of the initial reset state. Implies --only-with-demonstrations.",
    )
    parser.add_argument(
        "--max-envs",
        type=int,
        default=None,
        help="Only process the first N environments (after any filtering). Only used for folder mode.",
    )

    args = parser.parse_args()

    visualize_envs(
        args.input_path,
        output_dir=args.output_dir,
        nrow=args.nrow,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        num_workers=args.num_workers,
        grid=args.grid,
        num_noop_steps=args.num_noop_steps,
        only_with_demonstrations=args.only_with_demonstrations,
        final_state=args.final_state,
        max_envs=args.max_envs,
    )


if __name__ == "__main__":
    main()
