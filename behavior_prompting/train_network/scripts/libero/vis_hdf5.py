import os
import h5py
import numpy as np
import argparse
import imageio
from tqdm import tqdm

def _visit_item(name, obj):
    print(name)
    if isinstance(obj, h5py.Group):
        for key in obj.keys():
            pass  # Groups are traversed by visititems
    elif isinstance(obj, h5py.Dataset):
        print(f"  Dataset shape: {obj.shape}, dtype: {obj.dtype}")

def generate_video_from_hdf5(hdf5_path, output_dir=None, enable_print=True, single_episode=False):
    """Generate a video from an hdf5 dataset file.
    
    Args:
        hdf5_path: Path to the hdf5 dataset file
        output_dir: Optional directory to save the video. If it ends with .mp4, it's used directly as the video path.
                    Otherwise, if provided, saves to output_dir/videos/{split_name}/{dataset_name}.mp4
                    If None, uses default tmp_vis/hdf5_videos location
        enable_print: Whether to print progress messages
    """
    def fix_rgb_image(rgb_image):
        rgb_image = rgb_image[::-1] # (H, W, C), flip image to make it right-side up as it's stored upside down in the dataset
        return rgb_image

    with h5py.File(hdf5_path, 'r') as f:
        dataset_name = os.path.basename(hdf5_path).replace('.hdf5', '')
        
        # Check if output_dir is a direct video path (ends with .mp4)
        if output_dir is not None and output_dir.endswith('.mp4'):
            video_path = output_dir
        elif output_dir is not None:
            # Save video in the output directory, maintaining the same relative structure
            video_path = os.path.join(output_dir, 'videos', os.path.basename(os.path.dirname(hdf5_path)), dataset_name + '.mp4')
        else:
            video_path = os.path.abspath(os.path.join('tmp_vis', 'hdf5_videos', os.path.basename(os.path.dirname(hdf5_path)), dataset_name + '.mp4'))
        
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        out = imageio.get_writer(video_path, fps=20, codec='libx264')

        # Sort demo keys numerically to ensure correct order
        demo_keys = sorted(f['data'].keys(), key=lambda x: int(x.split('_')[1]))

        if single_episode:
            demo_keys = demo_keys[:1]
        iterable = tqdm(demo_keys, desc='Writing combined camera views video') if enable_print else demo_keys
        
        for demo in iterable:
            demo_data = f['data'][demo]
            agentview_rgb = demo_data['obs']['agentview_rgb']
            eye_in_hand_rgb = demo_data['obs']['eye_in_hand_rgb']

            for i in range(agentview_rgb.shape[0]):
                agentview_frame = fix_rgb_image(agentview_rgb[i][...,::-1]) # (H, W, C)
                eye_in_hand_frame = fix_rgb_image(eye_in_hand_rgb[i][...,::-1])
                combined_frame = np.hstack((agentview_frame, eye_in_hand_frame))
                out.append_data(combined_frame[...,::-1]) # convert to RGB
        out.close()
        if enable_print:
            print(f'Saved combined camera views video to {video_path}')
        return video_path

def print_hdf5_structure(hdf5_path, generate_video, single_episode=False):
    with h5py.File(hdf5_path, 'r') as f:
        print(f"Structure of {hdf5_path}:")
        f.visititems(_visit_item)

        if generate_video:
            generate_video_from_hdf5(hdf5_path, single_episode=single_episode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5_path", type=str)
    parser.add_argument("--generate-video", action="store_true")
    parser.add_argument("--single-episode", action="store_true",
                        help="Only generate video for the first episode")
    args = parser.parse_args()
    print_hdf5_structure(args.hdf5_path, args.generate_video, args.single_episode)
