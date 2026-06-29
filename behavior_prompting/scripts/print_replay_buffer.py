import argparse
from behavior_prompting.common.replay_buffer_util import print_replay_buffer_libero, print_replay_buffer_umi, print_replay_buffer_draw
from behavior_prompting.common.imagecodecs_numcodecs import register_codecs
register_codecs(verbose=False)

parser = argparse.ArgumentParser(description='View the contents of a replay buffer.')
parser.add_argument('--dataset_path', type=str, help='Path to the replay buffer to view.')
parser.add_argument('--vis_frame', action='store_true', help='Save images of the RGB data.')
parser.add_argument('--vis_video', action='store_true', help='Save video of the RGB data.')
parser.add_argument('--task_name', type=str, help='The name of the task.', choices=['libero', 'umi', 'draw'], required=True)
parser.add_argument('--load_buffer_into_memory', action='store_true', help='Load the replay buffer into memory instead of keeping it on disk. Faster for small datasets or when saving videos.')
parser.add_argument('--task-for-video', type=str, default=None, help='For umi and draw tasks: specify a task name to filter video frames to only include frames from tasks with this name.')
args = parser.parse_args()

if args.task_name == 'libero':
    print_replay_buffer_libero(args.dataset_path, vis_frame=args.vis_frame, vis_video=args.vis_video, load_buffer_into_memory=args.load_buffer_into_memory)
elif args.task_name == 'umi':
    print_replay_buffer_umi(args.dataset_path, vis_frame=args.vis_frame, vis_video=args.vis_video, load_buffer_into_memory=args.load_buffer_into_memory, task_for_video=args.task_for_video)
elif args.task_name == 'draw':
    print_replay_buffer_draw(args.dataset_path, vis_video=args.vis_video, load_buffer_into_memory=args.load_buffer_into_memory, task_for_video=args.task_for_video)
else:
    raise ValueError(f'Invalid task name: {args.task_name}')
