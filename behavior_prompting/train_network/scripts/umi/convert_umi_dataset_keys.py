"""
Convert a single-arm UMI zarr replay buffer from the old robot0/camera0 naming convention
to the new gripper_right/camera_right (default) or gripper_left/camera_left naming convention.

Usage:
    # Convert to new naming, treating robot0 as the right arm (default):
    python convert_umi_dataset_keys.py --input old.zarr --output new.zarr

    # Convert to new naming, treating robot0 as the left arm:
    python convert_umi_dataset_keys.py --input old.zarr --output new.zarr --side left

    # Convert in-place (modifies the original — make a backup first):
    python convert_umi_dataset_keys.py --input old.zarr --side right
"""

import argparse
import shutil
import zarr
import numpy as np
from pathlib import Path


OLD_TO_NEW_DATA_KEYS = {
    'robot0_eef_pos': 'gripper_{side}_eef_pos',
    'robot0_eef_rot_axis_angle': 'gripper_{side}_eef_rot_axis_angle',
    'robot0_gripper_width': 'gripper_{side}_gripper_width',
    'camera0_main_rgb': 'camera_{side}_main_rgb',
    'camera0_ultrawide_rgb': 'camera_{side}_ultrawide_rgb',
}

OLD_TO_NEW_META_KEYS = {
    'downsample_index_camera0_ultrawide_rgb': 'downsample_index_camera_{side}_ultrawide_rgb',
    'upsample_index_camera0_ultrawide_rgb': 'upsample_index_camera_{side}_ultrawide_rgb',
    'episode_ends_camera0_ultrawide_rgb': 'episode_ends_camera_{side}_ultrawide_rgb',
}


def _rename_keys_in_group(group, key_map, side, dry_run=False):
    renamed = []
    for old_key, new_key_template in key_map.items():
        if old_key not in group:
            continue
        new_key = new_key_template.format(side=side)
        if dry_run:
            print(f"  Would rename: {old_key} -> {new_key}")
        else:
            zarr.copy(group[old_key], group, name=new_key)
            del group[old_key]
            print(f"  Renamed: {old_key} -> {new_key}")
        renamed.append((old_key, new_key))
    return renamed


def convert_dataset(input_path: str, output_path: str, side: str, dry_run: bool = False):
    assert side in ('left', 'right'), f"side must be 'left' or 'right', got '{side}'"

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    in_place = (input_path.resolve() == output_path.resolve())

    if not in_place:
        if output_path.exists():
            raise FileExistsError(
                f"Output path already exists: {output_path}. Remove it first or choose a different path."
            )
        print(f"Copying {input_path} -> {output_path} ...")
        if not dry_run:
            shutil.copytree(str(input_path), str(output_path))
        print("Copy complete.")
    else:
        print(f"Converting in-place: {input_path}")

    target = output_path if not dry_run else input_path
    print(f"Opening zarr store at {target} ...")
    root = zarr.open(str(target), mode='r+' if not dry_run else 'r')

    print("\n--- data/ keys ---")
    if 'data' in root:
        _rename_keys_in_group(root['data'], OLD_TO_NEW_DATA_KEYS, side, dry_run=dry_run)
    else:
        print("  No 'data' group found.")

    print("\n--- meta/ keys ---")
    if 'meta' in root:
        _rename_keys_in_group(root['meta'], OLD_TO_NEW_META_KEYS, side, dry_run=dry_run)
    else:
        print("  No 'meta' group found.")

    print("\nDone." if not dry_run else "\nDry run complete (no changes made).")


def main():
    parser = argparse.ArgumentParser(
        description="Convert UMI zarr dataset keys from old robot0/camera0 convention to gripper_side/camera_side."
    )
    parser.add_argument(
        '--input', required=True,
        help='Path to the input zarr replay buffer.'
    )
    parser.add_argument(
        '--output', default=None,
        help='Path for the output zarr replay buffer. If omitted, converts in-place (modifies input).'
    )
    parser.add_argument(
        '--side', default='right', choices=['left', 'right'],
        help="Which side to map robot0/camera0 to. Default: 'right'."
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print what would be renamed without making any changes.'
    )
    args = parser.parse_args()

    output = args.output if args.output is not None else args.input

    convert_dataset(
        input_path=args.input,
        output_path=output,
        side=args.side,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    main()
