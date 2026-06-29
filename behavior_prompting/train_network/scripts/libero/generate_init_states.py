"""Adapted from LIBERO Pro"""

import os
import zipfile
import pickle
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import argparse

from libero.libero.envs import OffScreenRenderEnv


def generate_init_states_for_single_bddl(
    bddl_file_path: str,
    output_dir: str,
    num_inits: int = 50,
    height: int = 128,
    width: int = 128,
    overwrite: bool = False,
    show_progress: bool = False,
):
    """
    Generate init states for a single BDDL file.
    
    Args:
        bddl_file_path: Path to the BDDL file
        output_dir: Directory to save the .pruned_init file
        num_inits: Number of init states to generate
        height: Camera height
        width: Camera width
        overwrite: Whether to overwrite existing files
        show_progress: Whether to show progress bar for individual init state generation
    """
    np.random.seed(0)
    random.seed(0)

    bddl_file = Path(bddl_file_path).resolve()
    output_dir = Path(output_dir).resolve()
    os.makedirs(output_dir, exist_ok=True)
    
    task_base_name = bddl_file.stem
    output_filename = f"{task_base_name}.pruned_init"
    output_filepath = output_dir / output_filename
    
    if not overwrite and output_filepath.exists():
        return
    
    all_initial_states = []
    
    iter_range = tqdm(range(num_inits), desc=f"Generating initial states for {task_base_name}") if show_progress else range(num_inits)
    
    for i in iter_range:
        env_args = {
            "bddl_file_name": str(bddl_file),
            "camera_heights": height,
            "camera_widths": width,
        }
        env = OffScreenRenderEnv(**env_args)
        
        initial_state = env.get_sim_state()
        all_initial_states.append(initial_state)
        env.close()
    
    with zipfile.ZipFile(output_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        all_initial_states = np.array(all_initial_states)
        pickled_states_list = pickle.dumps(all_initial_states)
        zipf.writestr("archive/data.pkl", pickled_states_list)
        zipf.writestr("archive/version", b"1")
    
    print(f"Successfully saved {len(all_initial_states)} states to: {output_filepath}")


def generate_init_states(
    bddl_base_dir: str,
    output_dir: str,
    num_inits: int = 50,
    height: int = 128,
    width: int = 128,
    overwrite: bool = False,
):
    bddl_base_dir = Path(bddl_base_dir).resolve()
    output_dir = Path(output_dir).resolve()
    os.makedirs(output_dir, exist_ok=True)

    bddl_files = list(bddl_base_dir.glob("*.bddl"))

    for bddl_file in tqdm(bddl_files, desc="Processing BDDL files"):
        generate_init_states_for_single_bddl(
            bddl_file_path=str(bddl_file),
            output_dir=str(output_dir),
            num_inits=num_inits,
            height=height,
            width=width,
            overwrite=overwrite,
            show_progress=True,
        )

    print("\nAll tasks processed successfully!")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate init states for LIBERO BDDL tasks.")
    parser.add_argument("--bddl_base_dir", type=str, required=True, help="Directory containing BDDL files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save .pruned_init files.")
    parser.add_argument("--num_inits", type=int, default=50, help="Number of init states to generate per task.")
    parser.add_argument("--height", type=int, default=128, help="Camera height.")
    parser.add_argument("--width", type=int, default=128, help="Camera width.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_init_states(
        bddl_base_dir=args.bddl_base_dir,
        output_dir=args.output_dir,
        num_inits=args.num_inits,
        height=args.height,
        width=args.width,
    )
