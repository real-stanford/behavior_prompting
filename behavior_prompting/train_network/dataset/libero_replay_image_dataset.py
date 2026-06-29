import shutil
from typing import Dict, List
from libero.libero import get_libero_path
import torch
import numpy as np
import h5py
from tqdm import tqdm
import zarr
import os
import copy
from filelock import FileLock
import concurrent.futures
import multiprocessing
from behavior_prompting.common.pytorch_util import dict_apply
from behavior_prompting.train_network.model.common.normalizer import Normalizer
from behavior_prompting.train_network.model.common.rotation_transformer import RotationTransformer
from behavior_prompting.common.replay_buffer import (
    ReplayBuffer,
    check_chunks_compatible,
    get_optimal_chunks,
)
from behavior_prompting.train_network.common.sampler import SequenceSampler, get_train_mask, get_training_split_info_from_train_mask
from behavior_prompting.train_network.model.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_identity_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats,
)
from typing import Optional
import torch.nn.functional as F

from transformers import CLIPTokenizer

from behavior_prompting.train_network.utils.libero_util import get_hdf5_files, hdf5_to_task
from behavior_prompting.train_network.dataset.base_dataset import BaseDataset


def _receding_rgb_numpy_thwc_to_float_chw(arr: np.ndarray) -> np.ndarray:
    """(T, H, W, C) uint8 -> (T, C, H, W) float32, channel flip + resize 224 (matches training pipeline)."""
    x = np.moveaxis(arr, -1, 1).astype(np.float32) / 255.0
    x = x[:, :, ::-1] # (T, C, H, W), images are stored upside down in the dataset, so we flip them to be correct
    resize = 224
    return (
        F.interpolate(
            torch.from_numpy(x.copy()),
            size=(resize, resize),
            mode="bilinear",
            align_corners=False,
        )
        .numpy()
    )


class LiberoReplayImageDataset(BaseDataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_splits: Optional[List[str]]=None,
        dataset_name: Optional[str]=None,
        dataset_path: Optional[str]=None,
        replay_buffer: Optional[ReplayBuffer]=None,
        text_encoder_model_name: Optional[str]=None,
        action_padding: bool=False,
        use_cache=True,
        overwrite_cache=False,
        cache_dir: Optional[str]=None,
        seed=42,
        val_ratio=0.0,
        sample_type:str='episode',
        only_prompt:bool=False,
        max_segments:int=-1,
        name_suffix:str='',
        include_file_filters: list[str]=[],
        training_split_info: Optional[Dict[str, bool]]=None,
        only_goal_image: bool=False,
    ):
        rotation_transformer = RotationTransformer(
            from_rep="axis_angle", to_rep="rotation_6d"
        )

        if len(include_file_filters) > 0:
            assert name_suffix != '', 'name_suffix is required when include_file_filters is specified'

        assert dataset_path is None, 'dataset_path is not used, but has to be here since the training code expects it to be present'
        dataset_path = get_libero_path('datasets')

        if replay_buffer is None:
            if use_cache:
                assert cache_dir is not None, 'cache_dir must be provided when use_cache is True'
                cache_dir = os.path.join(cache_dir, os.path.basename(dataset_path), dataset_name)
                file_name = os.path.join(cache_dir, dataset_name)
                if name_suffix:
                    file_name = f"{file_name}_{name_suffix}"
                cache_zarr_path = file_name + ".zarr"
                cache_lock_path = cache_zarr_path + ".lock"
                cache_success_path = file_name + "_success.txt"
                print("Acquiring lock on cache.")
                print("Cache path:", cache_zarr_path)

                with FileLock(cache_lock_path):
                    # Check if cache exists but success marker doesn't (partial cache)
                    if os.path.exists(cache_zarr_path) and not os.path.exists(cache_success_path):
                        print("Cache folder exists but success marker is missing. Deleting partial cache.")
                        shutil.rmtree(cache_zarr_path)
                    
                    if overwrite_cache and os.path.exists(cache_zarr_path):
                        print("Overwriting cache.")
                        shutil.rmtree(cache_zarr_path)
                        # Also delete success marker if it exists
                        if os.path.exists(cache_success_path):
                            os.remove(cache_success_path)
                    
                    if not os.path.exists(cache_zarr_path):
                        # cache does not exists
                        # Delete success marker if it exists (shouldn't happen, but be safe)
                        if os.path.exists(cache_success_path):
                            os.remove(cache_success_path)
                        try:
                            print("Cache does not exist. Creating!")
                            with zarr.DirectoryStore(cache_zarr_path) as directory_store:
                                _convert_robomimic_to_replay(
                                    store=directory_store,
                                    shape_meta=shape_meta,
                                    dataset_path=dataset_path,
                                    dataset_splits=dataset_splits,
                                    rotation_transformer=rotation_transformer,
                                    include_file_filters=include_file_filters
                                )
                            # Create success marker after successful cache generation
                            with open(cache_success_path, 'w') as f:
                                f.write("Cache generation completed successfully\n")
                            print("Cache generation completed successfully.")
                        except Exception as e:
                            if os.path.exists(cache_zarr_path):
                                shutil.rmtree(cache_zarr_path)
                            # Also remove success marker if it exists (shouldn't happen, but be safe)
                            if os.path.exists(cache_success_path):
                                os.remove(cache_success_path)
                            raise e

                    replay_buffer = ReplayBuffer.create_from_path(cache_zarr_path)
            else:
                raise NotImplementedError('we only support caching for now')
        
        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb" and attr.get("in_replay_buffer", True):
                rgb_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

        assert replay_buffer.n_episodes == replay_buffer.n_tasks, 'we assume one episode corresponds to one task'
        train_mask = get_train_mask(replay_buffer, sample_type, val_ratio, training_split_info, seed)

        self.use_goal_image = 'goal_image' in shape_meta['obs'] and not shape_meta['obs']['goal_image'].get('ignore_by_policy', False)

        language_keys = ['task_language']
        self.using_language = not shape_meta['obs']['task_language'].get('ignore_by_policy', False)
        self.text_encoder_model_name = text_encoder_model_name
        if self.using_language:
            assert shape_meta['obs']['task_language']['horizon'] == 1, 'task_language horizon must be 1'
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(text_encoder_model_name)
        else:
            self.clip_tokenizer = None

        self.sampler_lowdim_keys = list()
        for key in lowdim_keys:
            if key not in language_keys:
                self.sampler_lowdim_keys.append(key)

        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.language_keys = language_keys
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.action_padding = action_padding
        self.train_mask = train_mask
        self.sample_type = sample_type
        self.only_prompt = only_prompt
        self.max_segments = max_segments
        self.include_file_filters = include_file_filters
        self.dataset_path = dataset_path
        self.action_rep = shape_meta['action']['rep']
        self.action_key = 'action' if self.action_rep == 'delta' else 'abs_action'

        self.use_prompting = self.shape_meta.use_prompting
        assert not (only_goal_image and self.only_prompt), "only_goal_image and only_prompt cannot both be True"
        self.only_goal_image = only_goal_image

        self.sampler_kwargs = {
            'shape_meta': self.shape_meta,
            'replay_buffer': self.replay_buffer,
            'action_padding': self.action_padding,
            'sample_type': self.sample_type,
            'only_prompt': self.only_prompt,
            'only_goal_image': self.only_goal_image,
            'max_segments': self.max_segments,
            'seed': seed,
            'action_key': self.action_key
        }

        sampler = SequenceSampler(
            mask=self.train_mask,
            **self.sampler_kwargs
        )
        self.sampler = sampler

    def get_validation_dataset(self):
        val_set = copy.copy(self)

        val_set.sampler = SequenceSampler(
            mask=~self.train_mask,
            **self.sampler_kwargs
        )

        return val_set

    def get_normalizer(self, **kwargs) -> Normalizer:
        """Note that unlike the UMI normalizer, we don't iterate over the dataset itself and instead just directly access values in the replay buffer. This works for LIBERO because we are just directly providing the values from the replay_buffer without any transformations (unlike UMI which does a bunch of conversions to relative trajectories)."""
        normalizer = Normalizer()

        # TODO implement temporally indepent normalization (can follow umi_task_dataset.py). This will require creating a dataloader on this dataset and iterating over the data such that the horizons are included in the data

        # action normalizer
        stat = array_to_stats(self.replay_buffer[self.action_key])
        normalizer["action"] = robomimic_abs_action_only_normalizer_from_stat(stat) # this normalizer is only for position (makes it -1 to 1) and leaves the rotation (already -1 to 1 in rot6d format) and gripper (already -1 to 1). This normalizer works regardless of whether the action representation is delta or absolute.

        # lowdim normalizer (excluding language)
        for key in self.lowdim_keys:
            if key == 'task_language':
                continue

            stat = array_to_stats(self.replay_buffer[key])

            if key == "ee_pos": # ee_pos is absolute position
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key == "ee_ori": # ee_ori is absolute rotation in rot6d format so already between -1 and 1
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key == "gripper_states": # gripper_states is finger positions [0 to 0.04, -0.04 to 0]
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("language"):
                continue  ## skip
            else:
                raise RuntimeError("unsupported")
            normalizer[key] = this_normalizer

        # image normalizer
        for key in self.rgb_keys:
            normalizer[key] = get_image_identity_normalizer()

        if self.use_goal_image:
            normalizer["goal_image"] = get_image_identity_normalizer()

        prompt_normalizer = copy.deepcopy(normalizer)
        prompt_normalizer.set_prompt_normalizer(None)

        normalizer.set_prompt_normalizer(prompt_normalizer) # use the same normalizer for prompt and receding obs

        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)

        if self.only_goal_image:
            goal_image = data.pop("goal_image")
            metadata = data.pop("metadata", {})
            goal_image = _receding_rgb_numpy_thwc_to_float_chw(goal_image)
            return {
                "obs": {"goal_image": torch.from_numpy(goal_image)},
                "metadata": metadata,
            }

        def prepare_obs_dict(data):
            for key in self.rgb_keys:
                if not key in data:
                    continue
                data[key] = _receding_rgb_numpy_thwc_to_float_chw(data[key])
            for key in self.sampler_lowdim_keys:
                if key not in data:
                    continue
                data[key] = data[key].astype(np.float32)
        
        if len(self.rgb_keys + self.lowdim_keys) > 0 and (self.rgb_keys + self.lowdim_keys)[0] in data:
            prepare_obs_dict(data)
        if 'prompt' in data:
            prepare_obs_dict(data['prompt']['obs'])

        if "goal_image" in data:
            data["goal_image"] = _receding_rgb_numpy_thwc_to_float_chw(data["goal_image"])

        # language encoding
        if self.using_language:
            task_idx = data['metadata']['task_idx']
            task_name = self.replay_buffer.task_names[task_idx]
            tokens = self.clip_tokenizer(
                task_name,
                padding='max_length',
                truncation=True,
                max_length=77, # CLIP's default max length
                return_tensors='np'
            )
            data['task_language'] = tokens['input_ids'].astype(np.int64)  # (1, 77)

        # action and metadata
        action = data.pop('action', None)
        metadata = data.pop('metadata', {})
        if "goal_image_task_idx" in metadata:
            metadata["goal_image_task_idx"] = torch.tensor(
                metadata["goal_image_task_idx"], dtype=torch.int64
            )

        # convert to torch
        torch_data = {
            "obs": dict_apply(data, torch.from_numpy),
            "metadata": metadata
        }
        if action is not None:
            torch_data['action'] = torch.from_numpy(action.astype(np.float32))

        return torch_data

    def shuffle_data_ordering(self, seed:int):
        self.sampler.shuffle_data_ordering(seed)

    def requires_epoch_shuffle(self) -> bool:
        return self.sampler.requires_epoch_shuffle()

    def is_multi_task(self) -> bool:
        return self.sample_type == 'task'

    def get_unique_task_name_to_dataset_indices(self) -> Dict[str, list[int]]:
        return self.sampler.get_unique_task_name_to_dataset_indices()

    def get_training_split_info(self) -> Dict[str, bool]:
        return get_training_split_info_from_train_mask(self.replay_buffer, self.sample_type, self.train_mask)

    def set_ignore_prompt(self, ignore_prompt: bool):
        self.sampler.set_ignore_prompt(ignore_prompt)

    def get_ignore_prompt(self) -> bool:
        return self.sampler.get_ignore_prompt()

def _convert_actions(raw_actions, rotation_transformer):
    """Convert the rotation in the actions according to `rotation_transformer`."""
    pos = raw_actions[..., :3]
    rot = raw_actions[..., 3:6]
    gripper = raw_actions[..., 6:]
    rot = rotation_transformer.forward(rot)
    raw_actions = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)
    return raw_actions


def _convert_robomimic_to_replay(
    store,
    shape_meta,
    dataset_path,
    dataset_splits,
    rotation_transformer,
    n_workers=None,
    max_inflight_tasks=None,
    include_file_filters=[]
):
    """The obs keys in the demos are: ['agentview_rgb', 'ee_pos', 'ee_ori' (axis angle format), 'ee_states' (ee_pos concat with ee_ori), 'eye_in_hand_rgb', 'gripper_states' , 'joint_states']"""
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        shape = attr["shape"]
        type = attr.get("type", "low_dim")
        if type == "rgb" and attr.get("in_replay_buffer", True):
            rgb_keys.append(key)
        elif type == "low_dim":
            lowdim_keys.append(key)

    def _get_chunks(array):
        chunks = get_optimal_chunks(shape=array.shape, dtype=array.dtype, target_chunk_bytes=2e4)
        check_chunks_compatible(chunks=chunks, shape=array.shape)
        return chunks

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    file_handles = []  # Store file handles if you need to keep them open
    demos_all = {}
    language_all = {}
    count = 0
    count_i_to_relative_i = {}

    dataset_paths = get_hdf5_files(dataset_path, dataset_splits, include_file_filters)

    for dataset_path_each in dataset_paths:
        task = hdf5_to_task(dataset_path_each)
        language_goal = task.language

        print(f"Loading {dataset_path_each}")
        file = h5py.File(
            dataset_path_each, "r"
        )  # Open the file without closing it immediately
        file_handles.append(
            file
        )  # Keep track of the file handle to avoid it being closed
        demos = file["data"]
        demos_indices = sorted([int(x.replace('demo_', '')) for x in demos])

        for i, demo_i in enumerate(demos_indices):
            demo = demos[f"demo_{demo_i}"]
            demos_all[f"demo_{count}"] = demo
            language_all[f"demo_{count}"] = language_goal
            count_i_to_relative_i[count] = demo_i
            count += 1
    print("Total demos:", count)

    demos = demos_all
    episode_ends = list()
    task_lengths = list()
    episode_names = list()
    prev_end = 0
    for i in range(len(demos)):
        demo_i_str = f"demo_{i}"
        relative_demo_i_str = f"demo_{count_i_to_relative_i[i]}"
        demo = demos[demo_i_str]
        episode_length = demo["actions"].shape[0]
        episode_end = prev_end + episode_length
        prev_end = episode_end
        episode_ends.append(episode_end)
        task_lengths.append(episode_length)
        episode_names.append(f"{language_all[demo_i_str]} - {relative_demo_i_str}")
    n_steps = episode_ends[-1]
    episode_starts = [0] + episode_ends[:-1]
    img_size = demos['demo_0']['obs']['agentview_rgb'].shape[1]
    _ = meta_group.array(
        "episode_ends",
        episode_ends,
        dtype=np.int64,
        compressor=None,
        chunks=_get_chunks(np.asarray(episode_ends)),
        overwrite=True,
    )
    _ = meta_group.array(
        "episode_names",
        episode_names,
        dtype=str,
        compressor=None,
        chunks=_get_chunks(np.asarray(episode_names)),
        overwrite=True,
    )
    _ = meta_group.array(
        "task_lengths",
        task_lengths,
        dtype=np.int64,
        compressor=None,
        chunks=_get_chunks(np.asarray(task_lengths)),
        overwrite=True,
    )
    _ = meta_group.array(
        "task_data_ends",
        episode_ends,
        dtype=np.int64,
        compressor=None,
        chunks=_get_chunks(np.asarray(episode_ends)),
        overwrite=True,
    )
    _ = meta_group.array(
        "task_data_ends",
        episode_ends,
        dtype=np.int64,
        compressor=None,
        chunks=_get_chunks(np.asarray(episode_ends)),
        overwrite=True,
    )
    _ = meta_group.array(
        "task_labels_ends",
        episode_ends,
        dtype=np.int64,
        compressor=None,
        chunks=_get_chunks(np.asarray(episode_ends)),
        overwrite=True,
    )

    # save lowdim data
    for key in tqdm(lowdim_keys + ["action", "abs_action"], desc="Loading lowdim data"):
        data_key = "obs/" + key
        if key == "action":
            data_key = "actions"
            this_language_data = list()
        if key == "abs_action":
            # by default the provided dataset does not have abs_actions unless added in
            data_key = "abs_actions"
            skip_key = False
            for demo in demos.values():
                if data_key not in demo:
                    skip_key = True
                    break
            if skip_key:
                continue
        if key == "task_language":
            continue
        this_data = list()
        for i in range(len(demos)):
            demo = demos[f"demo_{i}"]
            this_data.append(demo[data_key][:].astype(np.float32))

            if key == "action":
                this_language_data.append(language_all[f"demo_{i}"])

        this_data = np.concatenate(this_data, axis=0)

        if key == "ee_ori":
            # convert ee_ori from axis angle to specified rotation representation
            this_data = rotation_transformer.forward(this_data).astype(np.float32)

        if key == "action" or key == "abs_action":
            this_data = _convert_actions(
                raw_actions=this_data,
                rotation_transformer=rotation_transformer,
            )

            assert this_data.shape == (n_steps,) + tuple(shape_meta["action"]["shape"])

            if key == "action":
                this_language_data = np.array(this_language_data)
        else:
            assert this_data.shape == (n_steps,) + tuple(
                shape_meta["obs"][key]["shape"]
            )
        _ = data_group.array(
            name=key,
            data=this_data,
            shape=this_data.shape,
            chunks=_get_chunks(this_data),
            compressor=None,
            dtype=this_data.dtype,
        )

        if key == "action":
            _ = meta_group.array(
                name="task_names",
                data=this_language_data,
                shape=this_language_data.shape,
                chunks=_get_chunks(this_language_data),
                compressor=None,
                dtype=this_language_data.dtype,
            )

    def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
        try:
            zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
            return True
        except Exception as e:
            print(f"Error copying image: {e}")
            return False

    with tqdm(
        total=n_steps * len(rgb_keys), desc="Loading image data", mininterval=1.0
    ) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = set()
            for key in rgb_keys:
                data_key = "obs/" + key
                shape = (3, img_size, img_size)
                c, h, w = shape
                this_compressor = None
                img_arr = data_group.require_dataset(
                    name=key,
                    shape=(n_steps, h, w, c),
                    chunks=(1, h, w, c),
                    compressor=this_compressor,
                    dtype=np.uint8,
                )

                for episode_idx in range(len(demos)):
                    demo = demos[f"demo_{episode_idx}"]
                    hdf5_arr = demo["obs"][key]
                    for hdf5_idx in range(hdf5_arr.shape[0]):
                        if len(futures) >= max_inflight_tasks:
                            # limit number of inflight tasks
                            completed, futures = concurrent.futures.wait(
                                futures, return_when=concurrent.futures.FIRST_COMPLETED
                            )
                            for f in completed:
                                if not f.result():
                                    raise RuntimeError("Failed to encode image!")
                            pbar.update(len(completed))

                        zarr_idx = episode_starts[episode_idx] + hdf5_idx
                        futures.add(
                            executor.submit(
                                img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx
                            )
                        )
            completed, futures = concurrent.futures.wait(futures)
            for f in completed:
                if not f.result():
                    raise RuntimeError("Failed to encode image!")
            pbar.update(len(completed))

    # Ensure you close all files when you're done with them
    for file in file_handles:
        file.close()

    # Add missing fields in
    _ = root.require_group("labels", overwrite=True)

    replay_buffer = ReplayBuffer.create_from_group(root)
    return replay_buffer
