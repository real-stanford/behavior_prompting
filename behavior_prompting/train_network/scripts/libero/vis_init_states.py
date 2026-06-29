#!/usr/bin/env python3
"""
Visualize all init states from an init state file and save as a grid image.

Usage:
    python vis_init_states.py <path_to_init_state_file>
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

from libero.libero.envs import OffScreenRenderEnv


def find_bddl_file(init_state_path):
    """
    Find the corresponding BDDL file for an init state file.
    
    Args:
        init_state_path: Path to the .pruned_init file
        
    Returns:
        Path to the BDDL file, or None if not found
    """
    init_state_path = Path(init_state_path)
    
    # Try same directory first (replace .pruned_init with .bddl)
    bddl_path = init_state_path.parent / (init_state_path.stem + ".bddl")
    if bddl_path.exists():
        return str(bddl_path)
    
    # Try searching in common BDDL directories
    from libero.libero import get_libero_path
    bddl_files_path = get_libero_path("bddl_files")
    
    # Try to find by searching in subdirectories
    for root, dirs, files in os.walk(bddl_files_path):
        bddl_file = os.path.join(root, init_state_path.stem + ".bddl")
        if os.path.exists(bddl_file):
            return bddl_file
    
    return None


def make_grid(images, nrow=8, padding=2, normalize=False, pad_value=0):
    """Make a grid of images. Make sure images is a 4D tensor in the shape of (B x C x H x W)) or a list of torch tensors."""
    grid_image = torchvision.utils.make_grid(images, nrow=nrow, padding=padding, normalize=normalize, pad_value=pad_value).permute(1, 2, 0)
    return grid_image


def visualize_init_states(init_state_path, output_dir=None, nrow=10, camera_height=128, camera_width=128):
    """
    Visualize all init states from an init state file.
    
    Args:
        init_state_path: Path to the .pruned_init file
        output_dir: Directory to save the output image. If None, uses tmp_init_states_vis in the same folder as this script.
        nrow: Number of images per row in the grid
        camera_height: Camera height for rendering
        camera_width: Camera width for rendering
    """
    init_state_path = Path(init_state_path).resolve()
    
    if not init_state_path.exists():
        raise FileNotFoundError(f"Init state file not found: {init_state_path}")
    
    # Find BDDL file
    bddl_file = find_bddl_file(init_state_path)
    if bddl_file is None:
        raise FileNotFoundError(
            f"Could not find corresponding BDDL file for {init_state_path}. "
            f"Expected: {init_state_path.parent / (init_state_path.stem + '.bddl')}"
        )
    
    print(f"Loading init states from: {init_state_path}")
    print(f"Using BDDL file: {bddl_file}")
    
    # Load init states - handle both torch.save format and zipfile format
    import zipfile
    import pickle
    
    init_state_path_obj = Path(init_state_path)
    
    # Try to load as zipfile format (from generate_init_states.py)
    if zipfile.is_zipfile(str(init_state_path)):
        try:
            with zipfile.ZipFile(str(init_state_path), 'r') as zipf:
                # Check if it's our custom format
                if "archive/data.pkl" in zipf.namelist():
                    pickled_data = zipf.read("archive/data.pkl")
                    init_states = pickle.loads(pickled_data)
                    print(f"Loaded {len(init_states)} init states from zipfile format")
                else:
                    # Try torch.load format (torch.save also creates zipfiles)
                    init_states = torch.load(str(init_state_path), weights_only=False)
                    print(f"Loaded {len(init_states)} init states from torch format")
        except Exception as e:
            # Fall back to torch.load
            print(f"Warning: Failed to load as zipfile format, trying torch.load: {e}")
            init_states = torch.load(str(init_state_path), weights_only=False)
            print(f"Loaded {len(init_states)} init states from torch format")
    else:
        # Not a zipfile, try torch.load directly
        init_states = torch.load(str(init_state_path), weights_only=False)
        print(f"Loaded {len(init_states)} init states")
    
    # Create environment
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": camera_height,
        "camera_widths": camera_width
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # Fix random seed for reproducibility
    
    # Render each init state
    images = []
    env.reset()
    for eval_index in tqdm(range(len(init_states)), desc="Rendering init states"):
        env.set_init_state(init_states[eval_index])
        
        # Step a few times to let the environment settle
        for _ in range(5):
            obs, _, _, _ = env.step([0.] * 7)
        
        # Capture the image
        image = torch.from_numpy(obs["agentview_image"]).permute(2, 0, 1)
        images.append(image)
    
    env.close()
    
    # Create grid
    grid_image = make_grid(images, nrow=nrow, padding=2, pad_value=0)
    
    # Convert to numpy and flip vertically (images are typically flipped)
    grid_array = grid_image.numpy()[::-1]
    
    # Ensure values are in [0, 255] range for PIL Image
    if grid_array.max() <= 1.0:
        grid_array = (grid_array * 255).astype(np.uint8)
    else:
        grid_array = grid_array.astype(np.uint8)
    
    # Determine output directory
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "tmp_init_states_vis"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the image
    output_filename = f"{init_state_path.stem}_grid.png"
    output_path = output_dir / output_filename
    
    # Convert to PIL Image and save
    grid_pil = Image.fromarray(grid_array)
    grid_pil.save(output_path)
    
    print(f"Saved visualization to: {output_path.resolve()}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize all init states from an init state file"
    )
    parser.add_argument(
        "init_state_path",
        type=str,
        help="Path to the .pruned_init file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the output image. Default: tmp_init_states_vis in script directory"
    )
    parser.add_argument(
        "--nrow",
        type=int,
        default=10,
        help="Number of images per row in the grid (default: 10)"
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
    
    args = parser.parse_args()
    
    visualize_init_states(
        args.init_state_path,
        output_dir=args.output_dir,
        nrow=args.nrow,
        camera_height=args.camera_height,
        camera_width=args.camera_width
    )


if __name__ == "__main__":
    main()
