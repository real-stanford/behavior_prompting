#!/usr/bin/env python3
"""
Image Concatenation Script (with recursive search & filename filter)

This script takes a directory (optionally searched recursively) full of images
and concatenates them into one output image. You can filter images by a substring
in their filename.

Usage:
    python concat_images.py --input_dir /path/to/images --output output.jpg
    python concat_images.py --input_dir /path/to/images --output output.jpg --layout horizontal
    python concat_images.py --input_dir /path/to/images --output output.jpg --layout grid --cols 3
    python concat_images.py --input_dir /path/to/images --recursive --filename_contains cat --output cats.jpg
"""

import os
import argparse
import glob
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageOps


def get_image_files(directory: str,
                    extensions: List[str] = None,
                    recursive: bool = False,
                    filename_contains: Optional[str] = None) -> List[str]:
    """
    Get image files from a directory (optionally recursively) and optionally filter by filename substring.
    
    Args:
        directory: Path to the directory containing images
        extensions: List of file extensions to include (e.g., ['*.jpg', '*.png'])
        recursive: Whether to search subdirectories recursively
        filename_contains: Only include files whose name contains this substring (case-insensitive)
    
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.gif']

    image_files = []
    if recursive:
        for ext in extensions:
            pattern = os.path.join(directory, '**', ext)
            image_files.extend(glob.glob(pattern, recursive=True))
            pattern = os.path.join(directory, '**', ext.upper())
            image_files.extend(glob.glob(pattern, recursive=True))
    else:
        for ext in extensions:
            pattern = os.path.join(directory, ext)
            image_files.extend(glob.glob(pattern))
            pattern = os.path.join(directory, ext.upper())
            image_files.extend(glob.glob(pattern))

    if filename_contains:
        substring = filename_contains.lower()
        image_files = [f for f in image_files if substring in os.path.basename(f).lower()]

    return sorted(image_files)


def resize_images(images: List[Image.Image], target_size: Tuple[int, int] = None, 
                 resize_mode: str = 'fit') -> List[Image.Image]:
    if not images:
        return []
    if target_size is None:
        max_width = max(img.width for img in images)
        max_height = max(img.height for img in images)
        target_size = (max_width, max_height)
    resized_images = []
    for img in images:
        if resize_mode in ('fit', 'crop'):
            img_resized = ImageOps.fit(img, target_size, method=Image.Resampling.LANCZOS)
        elif resize_mode == 'pad':
            img_resized = ImageOps.pad(img, target_size, color='black')
        else:
            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
        resized_images.append(img_resized)
    return resized_images


def concatenate_horizontal(images: List[Image.Image]) -> Image.Image:
    if not images:
        raise ValueError("No images provided")
    min_height = min(img.height for img in images)
    resized_images = [img.resize((int(img.width * min_height / img.height), min_height),
                                 Image.Resampling.LANCZOS) for img in images]
    total_width = sum(img.width for img in resized_images)
    result = Image.new('RGB', (total_width, min_height))
    x_offset = 0
    for img in resized_images:
        result.paste(img, (x_offset, 0))
        x_offset += img.width
    return result


def concatenate_vertical(images: List[Image.Image]) -> Image.Image:
    if not images:
        raise ValueError("No images provided")
    min_width = min(img.width for img in images)
    resized_images = [img.resize((min_width, int(img.height * min_width / img.width)),
                                 Image.Resampling.LANCZOS) for img in images]
    total_height = sum(img.height for img in resized_images)
    result = Image.new('RGB', (min_width, total_height))
    y_offset = 0
    for img in resized_images:
        result.paste(img, (0, y_offset))
        y_offset += img.height
    return result


def concatenate_grid(images: List[Image.Image], cols: int = None, bg_color='white') -> Image.Image:
    """
    Concatenate images in a balanced grid layout while preserving aspect ratio.

    Args:
        images: List of PIL Image objects
        cols: Number of columns (auto if None)
        bg_color: Background color for padding

    Returns:
        Concatenated PIL Image
    """
    if not images:
        raise ValueError("No images provided")

    n_images = len(images)

    # Auto-determine number of columns for a roughly square grid
    if cols is None:
        cols = int(np.ceil(np.sqrt(n_images)))
    rows = int(np.ceil(n_images / cols))

    # Determine the target cell size based on the largest image dimensions
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)
    cell_size = (max_width, max_height)

    # Resize each image to fit inside the cell while keeping aspect ratio
    resized_images = []
    for img in images:
        img_copy = img.copy()
        img_copy.thumbnail(cell_size, Image.Resampling.LANCZOS)

        # Create a new blank image for the cell and paste the resized image centered
        cell_img = Image.new('RGB', cell_size, color=bg_color)
        offset_x = (cell_size[0] - img_copy.width) // 2
        offset_y = (cell_size[1] - img_copy.height) // 2
        cell_img.paste(img_copy, (offset_x, offset_y))
        resized_images.append(cell_img)

    # Create the final grid canvas
    grid_width = cols * cell_size[0]
    grid_height = rows * cell_size[1]
    result = Image.new('RGB', (grid_width, grid_height), color=bg_color)

    # Paste images into grid
    for idx, img in enumerate(resized_images):
        row = idx // cols
        col = idx % cols
        x = col * cell_size[0]
        y = row * cell_size[1]
        result.paste(img, (x, y))

    return result


def main():
    parser = argparse.ArgumentParser(description='Concatenate images from a directory')
    parser.add_argument('--input_dir', '-i', required=True,
                        help='Directory containing input images')
    parser.add_argument('--output', '-o', required=True,
                        help='Output image file path')
    parser.add_argument('--layout', '-l', choices=['horizontal', 'vertical', 'grid'],
                    default='grid', help='Layout for concatenation (default: grid)')
    parser.add_argument('--cols', '-c', type=int, default=None,
                        help='Number of columns for grid layout (auto if not specified)')
    parser.add_argument('--resize_mode', choices=['fit', 'crop', 'pad'], default='fit',
                        help='How to resize images (default: fit)')
    parser.add_argument('--target_size', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'),
                        help='Target size for all images (width height)')
    parser.add_argument('--extensions', nargs='+',
                        default=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'gif'],
                        help='Image file extensions to include')
    parser.add_argument('--recursive', action='store_true',
                        help='Recursively search for images in subdirectories')
    parser.add_argument('--filename_contains', type=str, default=None,
                        help='Only include images whose filename contains this substring (case-insensitive)')
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return 1

    extensions = [f'*.{ext}' for ext in args.extensions]
    image_files = get_image_files(args.input_dir, extensions,
                                  recursive=args.recursive,
                                  filename_contains=args.filename_contains)

    if not image_files:
        print(f"Error: No image files found in '{args.input_dir}' with given filters")
        return 1

    print(f"Found {len(image_files)} images:")
    for img_file in image_files:
        print(f"  - {os.path.basename(img_file)}")

    print("Loading images...")
    images = []
    for img_file in image_files:
        try:
            img = Image.open(img_file)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"Warning: Could not load {img_file}: {e}")

    if not images:
        print("Error: No images could be loaded")
        return 1

    target_size = tuple(args.target_size) if args.target_size else None
    if target_size:
        print(f"Resizing images to {target_size}...")
        images = resize_images(images, target_size, args.resize_mode)

    print(f"Concatenating images using {args.layout} layout...")
    try:
        if args.layout == 'horizontal':
            result = concatenate_horizontal(images)
        elif args.layout == 'vertical':
            result = concatenate_vertical(images)
        elif args.layout == 'grid':
            result = concatenate_grid(images, args.cols)
    except Exception as e:
        print(f"Error during concatenation: {e}")
        return 1

    print(f"Saving result to {args.output}...")
    try:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        result.save(args.output, quality=95)
        print(f"Successfully created concatenated image: {args.output}")
        print(f"Final image size: {result.size[0]}x{result.size[1]} pixels")
    except Exception as e:
        print(f"Error saving output image: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())