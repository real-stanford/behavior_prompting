#!/usr/bin/env python3
"""
Script to convert a zarr zip file (ZipStore) to a zarr directory store (DirectoryStore).
"""

import argparse
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import zarr
from zarr.storage import ZipStore, DirectoryStore


def convert_single_zip_to_directory(zip_path: Path, output_dir: Path):
    """
    Convert a single zarr zip file to a zarr directory store.
    
    Args:
        zip_path: Path to the input zarr zip file
        output_dir: Directory where the output zarr directory store will be saved
    
    Returns:
        tuple: (zip_path, success: bool, message: str)
    """
    try:
        # Validate input zip file exists
        if not zip_path.exists():
            return (zip_path, False, f"Zarr zip file not found: {zip_path}")
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine output path: same filename as input but without .zip extension
        zip_name = zip_path.stem  # filename without extension
        output_path = output_dir / zip_name
        
        print(f"[{zip_path.name}] Converting to {output_path}")
        
        # Open the zip store for reading
        zip_store = ZipStore(str(zip_path), mode='r')
        
        try:
            # Open the zarr group/array from the zip store
            source_zarr = zarr.open(zip_store, mode='r')
            
            # Create the directory store for writing
            dir_store = DirectoryStore(str(output_path))
            
            # Copy the zarr structure to the directory store
            if isinstance(source_zarr, zarr.Group):
                # If it's a group, copy recursively
                zarr.copy_store(zip_store, dir_store, source_path='/', dest_path='/')
            else:
                # If it's an array, copy it
                zarr.copy_store(zip_store, dir_store, source_path='/', dest_path='/')
            
            message = f"Successfully converted {zip_path.name} to {output_path}"
            print(f"[{zip_path.name}] {message}")
            return (zip_path, True, message)
            
        finally:
            zip_store.close()
    
    except Exception as e:
        message = f"Error converting {zip_path.name}: {e}"
        print(f"[{zip_path.name}] {message}")
        return (zip_path, False, message)


def get_output_path(zip_path: Path, output_dir: Path) -> Path:
    """Get the output path for a zip file."""
    zip_name = zip_path.stem  # filename without extension
    return output_dir / zip_name


def convert_zip_to_directory(input_path: str, output_dir: str, overwrite: bool = False, num_workers: int = None):
    """
    Convert a zarr zip file or all zarr.zip files in a directory to zarr directory stores.
    
    Args:
        input_path: Path to the input zarr zip file or directory containing zarr.zip files
        output_dir: Directory where the output zarr directory stores will be saved
        overwrite: If True, delete existing output directories after confirmation
        num_workers: Number of parallel workers (default: number of CPU cores)
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    
    # Validate input path exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect zip files to process
    if input_path.is_dir():
        # Find all .zarr.zip files in the directory
        zip_files = list(input_path.glob("*.zarr.zip"))
        
        if not zip_files:
            print(f"No .zarr.zip files found in {input_path}")
            return
        
        print(f"Found {len(zip_files)} .zarr.zip file(s) in {input_path}")
    else:
        # Single file
        zip_files = [input_path]
    
    # Check which output paths exist
    existing_outputs = []
    files_to_process = []
    
    for zip_file in zip_files:
        output_path = get_output_path(zip_file, output_dir)
        if output_path.exists() and output_path.is_dir():
            existing_outputs.append((zip_file, output_path))
        else:
            files_to_process.append(zip_file)
    
    # Handle overwrite mode
    if overwrite and existing_outputs:
        print(f"\nFound {len(existing_outputs)} existing output directory(ies).")
        print("Overwrite mode enabled. Confirmation required for each deletion:")
        
        # First, ask for confirmation for each deletion
        to_delete = []
        for zip_file, output_path in existing_outputs:
            response = input(f"Delete {output_path}? (y/n): ").strip().lower()
            if response == 'y':
                to_delete.append((zip_file, output_path))
            else:
                print(f"Skipping {output_path}")
        
        # Now perform all the deletions
        if to_delete:
            print(f"\nDeleting {len(to_delete)} directory(ies)...")
            for zip_file, output_path in to_delete:
                print(f"Deleting {output_path}...")
                shutil.rmtree(output_path)
                files_to_process.append(zip_file)
            print("Deletions complete.")
        
        print(f"\nWill process {len(files_to_process)} file(s) after deletions.")
    elif existing_outputs:
        print(f"\nFound {len(existing_outputs)} existing output directory(ies). Skipping those files.")
        print(f"Will process {len(files_to_process)} file(s) that don't have existing outputs.")
        if files_to_process:
            print("Use --overwrite to delete existing outputs and reprocess all files.")
    
    if not files_to_process:
        print("No files to process.")
        return
    
    # Process files in parallel
    if num_workers is None:
        num_workers = cpu_count()

    num_workers = min(num_workers, len(files_to_process))
    
    print(f"\nProcessing {len(files_to_process)} file(s) using {num_workers} worker(s)...")
    
    # Create a partial function with output_dir fixed
    convert_func = partial(convert_single_zip_to_directory, output_dir=output_dir)
    
    # Process in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(convert_func, files_to_process)
    
    # Print summary
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful
    
    print(f"\n{'='*60}")
    print(f"Conversion complete: {successful} successful, {failed} failed")
    print(f"{'='*60}")
    
    if failed > 0:
        print("\nFailed conversions:")
        for zip_path, success, message in results:
            if not success:
                print(f"  - {zip_path.name}: {message}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a zarr zip file (ZipStore) to a zarr directory store (DirectoryStore)"
    )
    parser.add_argument(
        "--zip_path",
        "-i",
        type=str,
        required=True,
        help="Path to the input zarr zip file or directory containing .zarr.zip files"
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        required=False,
        type=str,
        default=None,
        help="Output directory where the zarr directory store will be saved (default: same as zip_path if directory, or parent directory if file)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output directories (with confirmation) and reprocess all files"
    )
    parser.add_argument(
        "--num_workers",
        "-n",
        type=int,
        default=None,
        help="Number of parallel workers (default: number of CPU cores)"
    )
    
    args = parser.parse_args()
    
    # Determine output directory if not specified
    if args.output_dir is None:
        zip_path = Path(args.zip_path)
        if zip_path.is_dir():
            output_dir = str(zip_path)
        else:
            output_dir = str(zip_path.parent)
        print(f"Output directory not specified, using: {output_dir}")
    else:
        output_dir = args.output_dir
    
    convert_zip_to_directory(args.zip_path, output_dir, args.overwrite, args.num_workers)


if __name__ == "__main__":
    main()
