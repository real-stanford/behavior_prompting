#!/usr/bin/env python3
"""
Script to convert a zarr directory store (DirectoryStore) to a zarr zip file (ZipStore).
"""

import argparse
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import zarr
from zarr.storage import ZipStore, DirectoryStore


def convert_single_directory_to_zip(dir_path: Path, output_dir: Path):
    """
    Convert a single zarr directory store to a zarr zip file.

    Args:
        dir_path: Path to the input zarr directory store
        output_dir: Directory where the output zarr zip file will be saved

    Returns:
        tuple: (dir_path, success: bool, message: str)
    """
    try:
        if not dir_path.exists():
            return (dir_path, False, f"Zarr directory not found: {dir_path}")

        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / (dir_path.name + '.zip')

        print(f"[{dir_path.name}] Converting to {output_path}")

        dir_store = DirectoryStore(str(dir_path))

        try:
            zip_store = ZipStore(str(output_path), mode='w')
            try:
                zarr.copy_store(dir_store, zip_store, source_path='/', dest_path='/')
                message = f"Successfully converted {dir_path.name} to {output_path}"
                print(f"[{dir_path.name}] {message}")
                return (dir_path, True, message)
            finally:
                zip_store.close()
        finally:
            dir_store.close()

    except Exception as e:
        message = f"Error converting {dir_path.name}: {e}"
        print(f"[{dir_path.name}] {message}")
        return (dir_path, False, message)


def get_output_path(dir_path: Path, output_dir: Path) -> Path:
    return output_dir / (dir_path.name + '.zip')


def convert_directory_to_zip(input_path: str, output_dir: str, overwrite: bool = False, num_workers: int = None):
    """
    Convert a zarr directory store or all zarr directory stores in a directory to zarr zip files.

    Args:
        input_path: Path to the input zarr directory store or directory containing zarr directory stores
        output_dir: Directory where the output zarr zip files will be saved
        overwrite: If True, delete existing output files after confirmation
        num_workers: Number of parallel workers (default: number of CPU cores)
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir() and not (input_path / '.zgroup').exists() and not (input_path / '.zarray').exists():
        # Directory containing multiple zarr stores
        dir_paths = [p for p in sorted(input_path.iterdir()) if p.is_dir() and ((p / '.zgroup').exists() or (p / '.zarray').exists())]

        if not dir_paths:
            print(f"No zarr directory stores found in {input_path}")
            return

        print(f"Found {len(dir_paths)} zarr directory store(s) in {input_path}")
    else:
        dir_paths = [input_path]

    existing_outputs = []
    dirs_to_process = []

    for dir_path in dir_paths:
        output_path = get_output_path(dir_path, output_dir)
        if output_path.exists():
            existing_outputs.append((dir_path, output_path))
        else:
            dirs_to_process.append(dir_path)

    if overwrite and existing_outputs:
        print(f"\nFound {len(existing_outputs)} existing output file(s).")
        print("Overwrite mode enabled. Confirmation required for each deletion:")

        to_delete = []
        for dir_path, output_path in existing_outputs:
            response = input(f"Delete {output_path}? (y/n): ").strip().lower()
            if response == 'y':
                to_delete.append((dir_path, output_path))
            else:
                print(f"Skipping {output_path}")

        if to_delete:
            print(f"\nDeleting {len(to_delete)} file(s)...")
            for dir_path, output_path in to_delete:
                print(f"Deleting {output_path}...")
                output_path.unlink()
                dirs_to_process.append(dir_path)
            print("Deletions complete.")

        print(f"\nWill process {len(dirs_to_process)} store(s) after deletions.")
    elif existing_outputs:
        print(f"\nFound {len(existing_outputs)} existing output file(s). Skipping those.")
        print(f"Will process {len(dirs_to_process)} store(s) that don't have existing outputs.")
        if dirs_to_process:
            print("Use --overwrite to delete existing outputs and reprocess all.")

    if not dirs_to_process:
        print("No stores to process.")
        return

    if num_workers is None:
        num_workers = cpu_count()

    num_workers = min(num_workers, len(dirs_to_process))

    print(f"\nProcessing {len(dirs_to_process)} store(s) using {num_workers} worker(s)...")

    convert_func = partial(convert_single_directory_to_zip, output_dir=output_dir)

    with Pool(processes=num_workers) as pool:
        results = pool.map(convert_func, dirs_to_process)

    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful

    print(f"\n{'='*60}")
    print(f"Conversion complete: {successful} successful, {failed} failed")
    print(f"{'='*60}")

    if failed > 0:
        print("\nFailed conversions:")
        for dir_path, success, message in results:
            if not success:
                print(f"  - {dir_path.name}: {message}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a zarr directory store (DirectoryStore) to a zarr zip file (ZipStore)"
    )
    parser.add_argument(
        "--dir_path",
        "-i",
        type=str,
        required=True,
        help="Path to the input zarr directory store or directory containing zarr directory stores"
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        required=False,
        type=str,
        default=None,
        help="Output directory where the zarr zip file will be saved (default: same directory as input)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output zip files (with confirmation) and reprocess all"
    )
    parser.add_argument(
        "--num_workers",
        "-n",
        type=int,
        default=None,
        help="Number of parallel workers (default: number of CPU cores)"
    )

    args = parser.parse_args()

    if args.output_dir is None:
        dir_path = Path(args.dir_path)
        output_dir = str(dir_path.parent)
        print(f"Output directory not specified, using: {output_dir}")
    else:
        output_dir = args.output_dir

    convert_directory_to_zip(args.dir_path, output_dir, args.overwrite, args.num_workers)


if __name__ == "__main__":
    main()
