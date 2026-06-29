import cv2
import imageio
import os

def cut_first_n_frames(video_path: str, n_frames: int, skip_if_no_frames: bool=False, skip_if_too_few_frames: bool=False) -> None:
    """
    Takes a video path and saves a new video with the first n frames cut out.
    The new video overwrites the input video file.
    
    Args:
        video_path (str): Path to input video file
        n_frames (int): Number of frames to cut from start
        skip_if_no_frames (bool): If True, skip the video if it has no frames
        skip_if_too_few_frames (bool): If True, skip the video if it has less than n_frames frames
    """
    # Create temporary output path
    base, ext = os.path.splitext(video_path)
    tmp_path = f"{base}_tmp{ext}"
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0 and skip_if_no_frames:
        cap.release()
        return

    if total_frames <= n_frames and skip_if_too_few_frames:
        cap.release()
        return
    
    if n_frames >= total_frames:
        raise ValueError(f"Cannot cut {n_frames} frames from video {video_path} with only {total_frames} frames")
    
    # Create video writer with imageio using H.264 codec
    writer = imageio.get_writer(tmp_path, fps=fps, codec='libx264')
    
    # Skip first n frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, n_frames)
    
    # Write remaining frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR (OpenCV) to RGB for imageio/ffmpeg
        writer.append_data(frame[:, :, ::-1])
    
    # Release everything
    cap.release()
    writer.close()
    
    # Replace original with trimmed version
    os.remove(video_path)
    os.rename(tmp_path, video_path)
