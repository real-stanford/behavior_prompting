"""
RMQ-based policy inference server for real-env bimanual iPhUMI deployment.

All pose-representation conversion (xyz_wxyz ↔ relative pos3+rot6d) is handled by
the task (iphumi_arx5_bimanual_task.py). This server only normalizes inputs, runs
the model, and unnormalizes outputs.

Exposes the following topics (compatible with real_env PolicyAgent):
  policy_config     → serialized config dict
  policy_reset      → "RESET" → "OK"
  policy_prompt     → {"task_name": str, "prompt_index": int} → "OK"
  policy_inference  → obs dict (model format) → action dict (model format)
  export_recorded_data → ignored, returns ""

Observation input keys (model format, relative pos3 + rot6d):
  gripper_left_eef_pos:                        (N, 3) relative position wrt latest obs
  gripper_left_eef_rot_axis_angle:             (N, 6) relative rot6d wrt latest obs
  gripper_left_gripper_width:                  (N, 1)
  gripper_right_eef_pos:                       (N, 3)
  gripper_right_eef_rot_axis_angle:            (N, 6)
  gripper_right_gripper_width:                 (N, 1)
  gripper_left_eef_pos_wrt_gripper_right:      (N, 3) if in shape_meta
  gripper_left_eef_rot_axis_angle_wrt_gripper_right: (N, 6) if in shape_meta
  gripper_right_eef_pos_wrt_gripper_left:      (N, 3) if in shape_meta
  gripper_right_eef_rot_axis_angle_wrt_gripper_left: (N, 6) if in shape_meta
  gripper_left_eef_rot_axis_angle_wrt_start:   (N, 6) if in shape_meta
  camera_left_main:                            (N, H, W, 3) uint8
  camera_left_ultrawide:                       (N, H, W, 3) uint8
  camera_right_main:                           (N, H, W, 3) uint8
  camera_right_ultrawide:                      (N, H, W, 3) uint8
  episode_idx: scalar (ignored)

Action output keys (model format, relative pos3 + rot6d, unnormalized):
  action_left_eef_pos:           (K, 3)
  action_left_eef_rot_axis_angle: (K, 6)
  action_left_gripper_width:     (K, 1)
  action_right_eef_pos:          (K, 3)
  action_right_eef_rot_axis_angle: (K, 6)
  action_right_gripper_width:    (K, 1)
"""

import datetime
import os
import sys
import threading
import time
import traceback

import imageio

import click
import dill
import hydra
import numpy as np
import omegaconf
import robotmq
import torch
from transformers import CLIPTokenizer

from behavior_prompting.common.cv_util import get_image_transform_with_border
from behavior_prompting.common.pytorch_util import dict_apply, move_batch_to_numpy
from behavior_prompting.train_network.model.common.base_policy import BasePolicy
from behavior_prompting.train_network.common.sampler import get_training_task_names_from_training_split_info
from behavior_prompting.train_network.utils.plot_util import PromptAttentionLogger
from behavior_prompting.train_network.utils.umi_util import vis_prompt
from behavior_prompting.train_network.workspace.base_workspace import BaseWorkspace


def _echo_exception() -> str:
    exc_type, exc_value, exc_tb = sys.exc_info()
    return "".join(traceback.format_exception(exc_type, exc_value, exc_tb))


class PolicyInferenceNodeRealEnv:
    def __init__(
        self,
        ckpt_path: str,
        server_endpoint: str,
        dataset_path: str | None,
        device: str,
        prompt_from_dataset: str = "train",
        vis_prompt_flag: bool = False,
        attention_vis: bool = False,
        allow_unseen_language: bool = False,
        is_eval_dataset: bool = False,
    ):
        if not ckpt_path.endswith(".ckpt"):
            ckpt_path = os.path.join(ckpt_path, "checkpoints", "latest.ckpt")
        self.ckpt_path = ckpt_path
        self.allow_unseen_language = allow_unseen_language

        payload = torch.load(open(ckpt_path, "rb"), map_location="cpu", pickle_module=dill)
        self.cfg = payload["cfg"]
        self.epoch: int = payload["pickles"].get("epoch", 0)
        if isinstance(self.epoch, bytes):
            self.epoch = dill.loads(self.epoch)

        cfg_path = ckpt_path.replace(".ckpt", ".yaml")
        with open(cfg_path, "w") as f:
            f.write(omegaconf.OmegaConf.to_yaml(self.cfg))
            print(f"[policy] Exported config to {cfg_path}")

        print(
            f"[policy] Loading: name={self.cfg.name}, workspace={self.cfg._target_}, "
            f"policy={self.cfg.model._target_}"
        )

        cls = hydra.utils.get_class(self.cfg._target_)
        self.workspace: BaseWorkspace = cls(self.cfg)
        self.workspace.load_payload(payload, exclude_keys=["optimizer"], include_keys=None)

        self.policy: BasePolicy = self.workspace.model
        self.device = torch.device(device)
        self.policy.eval().to(self.device)
        self.policy.reset()

        self.shape_meta = self.cfg.task.shape_meta

        # Language support
        lang_meta = self.shape_meta.get("obs", {}).get("task_language", None)
        self.using_language = lang_meta is not None and not lang_meta.get("ignore_by_policy", False)
        if self.using_language:
            text_encoder_model_name = self.cfg.model.obs_encoder.text_encoder_model_name
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(text_encoder_model_name)
            self._trained_task_names: set[str] = set(
                get_training_task_names_from_training_split_info(self.policy.get_training_split_info())
            )
            print(f"[policy] Language support enabled (model={text_encoder_model_name}), trained tasks: {sorted(self._trained_task_names)}")
        else:
            self.clip_tokenizer = None
            self._trained_task_names = set()

        # Infer ordered robot prefixes from shape_meta (sorted for stable robot{i} assignment).
        self.robot_prefixes: list[str] = sorted([
            key[: -len("_eef_pos")]
            for key, attr in self.shape_meta["obs"].items()
            if key.endswith("_eef_pos") and "wrt" not in key
        ])
        print(f"[policy] robot_prefixes={self.robot_prefixes}")

        # Build obs_key_mapping: deploy key (robot{i}_*) → model key.
        # Low-dim obs arrive as pos3+rot6d (already converted by PolicyAgent).
        # Image obs arrive as uint8 THWC and need format conversion.
        self._obs_key_map: dict[str, str] = {}
        for i, prefix in enumerate(self.robot_prefixes):
            arm = prefix[len("gripper_"):]  # "left" or "right"
            for suffix in ["eef_pos", "eef_rot_axis_angle", "gripper_width", "eef_rot_axis_angle_wrt_start"]:
                model_key = f"{prefix}_{suffix}"
                if model_key in self.shape_meta["obs"]:
                    self._obs_key_map[f"robot{i}_{suffix}"] = model_key
            for j, other_prefix in enumerate(self.robot_prefixes):
                if i == j:
                    continue
                for suffix in ["eef_pos", "eef_rot_axis_angle"]:
                    model_key = f"{prefix}_{suffix}_wrt_{other_prefix}"
                    if model_key in self.shape_meta["obs"]:
                        self._obs_key_map[f"robot{i}_{suffix}_wrt_robot{j}"] = model_key
            for cam in ["main", "ultrawide"]:
                model_key = f"camera_{arm}_{cam}_rgb"
                if model_key in self.shape_meta["obs"]:
                    self._obs_key_map[f"robot{i}_{cam}_camera"] = model_key
        print(f"[policy] obs_key_map={list(self._obs_key_map.keys())}")

        # optional prompt dataset
        uses_prompting = bool(self.shape_meta.get("use_prompting", False))
        if dataset_path and not uses_prompting:
            raise ValueError(
                "--dataset-path was provided but this policy was not trained with prompting. "
                "Remove --dataset-path."
            )
        assert prompt_from_dataset in ("train", "val"), \
            f"prompt_from_dataset must be 'train' or 'val', got {prompt_from_dataset!r}"
        self.prompt_dataset = None
        if dataset_path:
            from behavior_prompting.common.imagecodecs_numcodecs import register_codecs
            register_codecs(verbose=False)
            dataset_cfg = omegaconf.OmegaConf.create(dict(self.cfg.task.dataset))
            dataset_cfg.val_ratio = 0 # if we are loading an eval dataset we don't want any of the demonstrations to be used as validation
            prompt_dataset = hydra.utils.instantiate(
                dataset_cfg,
                only_prompt=True,
                dataset_path=dataset_path,
                training_split_info=None if is_eval_dataset else self.policy.get_training_split_info(),
                allow_zip_file=True,
            )
            if not is_eval_dataset and prompt_from_dataset == "val":
                prompt_dataset = prompt_dataset.get_validation_dataset()
            self.prompt_dataset = prompt_dataset
            task_names = list(self.prompt_dataset.get_unique_task_name_to_dataset_indices().keys())
            print(f"[policy] Prompt dataset loaded ({prompt_from_dataset} split), tasks:")
            for name in task_names:
                print(f"  {name}")

        self.vis_prompt_flag = vis_prompt_flag
        self.enable_attention_visualization = attention_vis

        # Attention visualization state
        self.attn_logger: PromptAttentionLogger | None = (
            PromptAttentionLogger(self.policy, 0) if attention_vis else None
        )
        self.input_prompt_dict: dict | None = None
        self._rollout_frames: list[np.ndarray] = []  # one hstacked frame per inference call
        self._rollout_timestamps: list[float] = []   # monotonic timestamp for each frame

        # Camera obs keys used by PromptAttentionLogger
        self._camera_obs_keys = [
            k for k in self.shape_meta["obs"] if "rgb" in k
        ]

        # RMQ server
        self.server_endpoint = server_endpoint
        self.server = robotmq.RMQServer("policy_server", server_endpoint)
        self.server.add_topic("policy_config", message_remaining_time_s=60)
        self.server.add_topic("policy_reset", message_remaining_time_s=60)
        self.server.add_topic("policy_prompt", message_remaining_time_s=60)
        self.server.add_topic("policy_inference", message_remaining_time_s=60)
        self.server.add_topic("export_recorded_data", message_remaining_time_s=60)

    # ------------------------------------------------------------------
    # Policy config dict (format expected by real_env PolicyAgent)
    # ------------------------------------------------------------------

    def _build_policy_config(self) -> dict:
        now = datetime.datetime.now()
        cfg_container = omegaconf.OmegaConf.to_container(self.cfg, resolve=True)
        # low_dim_obs_horizon is the number of obs steps sent; proprio_length = horizon - 1
        # (policy_agent adds +1 back, so proprio_history_len = proprio_length + 1 = low_dim_obs_horizon)
        low_dim_obs_horizon: int = self.cfg.task.get("low_dim_obs_horizon", 2)
        proprio_length: int = low_dim_obs_horizon - 1
        task_cfg = cfg_container.get("task", {})
        return {
            "workspace": {
                "proprio_length": proprio_length,
                "img_obs_horizon": task_cfg.get("img_obs_horizon"),
                "obs_down_sample_steps": task_cfg.get("obs_down_sample_steps"),
                "ultrawide_down_sample_steps": task_cfg.get("ultrawide_down_sample_steps"),
            },
            "use_prompting": bool(self.shape_meta.get("use_prompting", False)),
            "policy_name": self.cfg.name,
            "run_name": self.cfg.name,
            "epoch": self.epoch,
            "date_str": now.strftime("%Y-%m-%d"),
            "time_str": now.strftime("%H-%M-%S"),
        }

    # ------------------------------------------------------------------
    # Main server loop
    # ------------------------------------------------------------------

    def run_node(self):
        print(f"[policy] Serving on {self.server_endpoint}")
        while True:
            data, topic = self.server.wait_for_request(timeout_s=0.1)
            if topic is None:
                continue

            if topic == "policy_config":
                try:
                    config = self._build_policy_config()
                    self.server.reply_request(topic=topic, data=robotmq.serialize(config))
                except Exception:
                    err = _echo_exception()
                    print(f"[policy] policy_config error:\n{err}")
                    self.server.reply_request(topic=topic, data=robotmq.serialize(f"ERROR: {err}"))

            elif topic == "policy_reset":
                try:
                    inference_fps = 1.0  # fallback only; actual fps derived from timestamps

                    # Capture state before clearing — rendering happens in background.
                    should_render = (
                        self.enable_attention_visualization
                        and self.input_prompt_dict is not None
                        and self.attn_logger is not None
                        and len(self._rollout_frames) > 0
                    )
                    attn_logger_ref   = self.attn_logger
                    input_prompt_ref  = self.input_prompt_dict
                    rollout_frames     = self._rollout_frames[:]
                    rollout_timestamps = self._rollout_timestamps[:]

                    self.policy.reset()
                    self.attn_logger        = PromptAttentionLogger(self.policy, 0) if self.enable_attention_visualization else None
                    self.input_prompt_dict  = None
                    self._rollout_frames    = []
                    self._rollout_timestamps = []
                    print(f"[policy] Policy reset (attention_vis={self.enable_attention_visualization})")
                    self.server.reply_request(topic=topic, data=robotmq.serialize("OK"))

                    if should_render:
                        time_str     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        vis_out_path = os.path.abspath(os.path.join(
                            "tmp_vis", f"attention_vis_{self.cfg.name}_{time_str}.mp4"
                        ))
                        os.makedirs(os.path.dirname(vis_out_path), exist_ok=True)
                        camera_keys = self._camera_obs_keys
                        def _render(attn=attn_logger_ref, prompt=input_prompt_ref,
                                    frames=rollout_frames, timestamps=rollout_timestamps,
                                    fallback_fps=inference_fps, out=vis_out_path, ckeys=camera_keys):
                            try:
                                # Compute fps from measured inter-frame intervals so the
                                # video duration matches the actual episode duration.
                                if len(timestamps) >= 2:
                                    fps = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
                                else:
                                    fps = fallback_fps
                                rollout_path = out.replace(".mp4", "_rollout.mp4")
                                writer = imageio.get_writer(rollout_path, fps=fps, codec="libx264")
                                for f in frames:
                                    writer.append_data(f)
                                writer.close()
                                # steps_per_render=1, exec_action_horizon=1: one rollout frame
                                # per attention entry; fps controls playback speed.
                                attn.vis(out, prompt, rollout_path, ckeys,
                                         1, 1, fps)
                                print(f"[policy] Attention visualization saved to {out}")
                            except Exception:
                                print(f"[policy] Attention vis failed:\n{_echo_exception()}")
                        print(f"[policy] Starting attention visualization rendering in separate thread")
                        threading.Thread(target=_render, daemon=True).start()

                except Exception:
                    err = _echo_exception()
                    print(f"[policy] policy_reset error:\n{err}")
                    self.server.reply_request(topic=topic, data=robotmq.serialize(f"ERROR: {err}"))

            elif topic == "policy_prompt":
                try:
                    request = robotmq.deserialize(data)
                    self._handle_prompt(request["task_name"], request["prompt_index"])
                    self.server.reply_request(topic=topic, data=robotmq.serialize("OK"))
                except Exception:
                    err = _echo_exception()
                    print(f"[policy] policy_prompt error:\n{err}")
                    self.server.reply_request(topic=topic, data=robotmq.serialize(f"ERROR: {err}"))

            elif topic == "policy_inference":
                try:
                    obs_deployment = robotmq.deserialize(data)
                    start_t = time.monotonic()
                    obs_model = self._process_obs(obs_deployment)
                    obs_tensors = {
                        k: torch.from_numpy(np.array(v, dtype=np.int64 if k == "task_language" else np.float32))
                        .unsqueeze(0)
                        .to(self.device)
                        for k, v in obs_model.items()
                    }
                    kwargs = {}
                    if self.enable_attention_visualization:
                        kwargs["need_weights"] = True
                        kwargs["average_attn_weights"] = True
                    with torch.inference_mode():
                        result = self.policy.predict_action(obs_tensors, **kwargs)
                    if self.enable_attention_visualization:
                        self.attn_logger.log(result)
                        frame = self._extract_rollout_frame(obs_model)
                        if frame is not None:
                            self._rollout_frames.append(frame)
                            self._rollout_timestamps.append(time.monotonic())
                    action_pred = result["action_pred"][0].detach().cpu().numpy()  # (K, 20)
                    actions = self._process_action(action_pred)
                    print(f"[policy] inference time: {time.monotonic() - start_t:.3f}s")
                    self.server.reply_request(topic=topic, data=robotmq.serialize(actions))
                except Exception:
                    err = _echo_exception()
                    print(f"[policy] policy_inference error:\n{err}")
                    self.server.reply_request(topic=topic, data=robotmq.serialize(f"ERROR: {err}"))

            elif topic == "export_recorded_data":
                self.server.reply_request(topic=topic, data=robotmq.serialize(""))

    # ------------------------------------------------------------------
    # Prompt loading
    # ------------------------------------------------------------------

    def _handle_prompt(self, task_name: str, prompt_index: int) -> None:
        assert self.prompt_dataset is not None, "No prompt dataset — pass --dataset-path"
        task_map = self.prompt_dataset.get_unique_task_name_to_dataset_indices()
        assert task_name in task_map, (
            f"Task {task_name!r} not in dataset. Available: {list(task_map)}"
        )
        indices = task_map[task_name]
        assert 0 <= prompt_index < len(indices), (
            f"prompt_index {prompt_index} out of range for task {task_name!r} ({len(indices)} demos available)"
        )
        sample = self.prompt_dataset[indices[prompt_index]]
        prompt_dict = sample["obs"]["prompt"]
        prompt_tensors = dict_apply(
            prompt_dict,
            lambda x: torch.from_numpy(np.array(x)).unsqueeze(0).to(self.device),
        )
        with torch.inference_mode():
            self.policy.prompt(prompt_tensors)
        self.input_prompt_dict = move_batch_to_numpy(sample["obs"]["prompt"])
        print(f"[policy] Applied prompt: task={task_name!r}, index={prompt_index}")

        if self.vis_prompt_flag:
            out_path = os.path.abspath(f"tmp_vis/prompt_{task_name.replace(' ', '_')}_{prompt_index}.mp4")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            vis_prompt(self.input_prompt_dict, out_path)
            print(f"[policy] Prompt visualization saved to {out_path}")

    def _extract_rollout_frame(self, obs_model: dict) -> np.ndarray | None:
        """Extract the most recent left+right ultrawide frame from obs and hstack them."""
        uw_keys = sorted(k for k in obs_model if "ultrawide" in k)
        if not uw_keys:
            uw_keys = sorted(k for k in obs_model if "rgb" in k)
        if not uw_keys:
            return None
        left_key  = next((k for k in uw_keys if "left"  in k), uw_keys[0])
        right_key = next((k for k in uw_keys if "right" in k), uw_keys[-1])
        def to_hwc(arr):  # (C, H, W) float [0,1] → (H, W, C) uint8
            return (arr[-1] * 255).astype(np.uint8).transpose(1, 2, 0)
        left_frame  = to_hwc(obs_model[left_key])
        right_frame = to_hwc(obs_model[right_key])
        return np.hstack([left_frame, right_frame])

    def _process_obs(self, obs: dict) -> dict:
        """Translate robot{i}_* deployment keys to model keys and convert image format.

        Low-dim obs arrive as pos3+rot6d (float32) from PolicyAgent — pass through.
        Images arrive as uint8 (N, H, W, 3) — convert to float32 (N, 3, H, W) in [0, 1].
        """
        result: dict[str, np.ndarray] = {}
        for deploy_key, model_key in self._obs_key_map.items():
            if deploy_key not in obs:
                continue
            val = obs[deploy_key]
            if "camera" in model_key:
                imgs = np.array(val)  # (N, H, W, 3) uint8
                _, hi, wi, _ = imgs.shape
                co, ho, wo = self.shape_meta["obs"][model_key]["shape"]
                if hi != ho or wi != wo:
                    tf = get_image_transform_with_border(in_res=(wi, hi), out_res=(wo, ho))
                    imgs = np.stack([tf(x) for x in imgs])
                result[model_key] = np.moveaxis(imgs, -1, 1).astype(np.float32) / 255.0
            else:
                result[model_key] = np.array(val, dtype=np.float32)

        if self.using_language:
            assert "task_name" in obs, "Language policy requires 'task_name' string in obs dict"
            task_name = obs["task_name"]
            if task_name not in self._trained_task_names and not self.allow_unseen_language:
                raise ValueError(
                    f"Language command {task_name!r} was not seen during training. "
                    f"Trained tasks: {sorted(self._trained_task_names)}"
                )
            tokens = self.clip_tokenizer(
                task_name,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="np",
            )
            result["task_language"] = tokens["input_ids"].astype(np.int64)  # (1, 77)

        return result

    def _process_action(self, action_pred: np.ndarray) -> dict:
        """(K, D) unnormalized action → action{i}_eef_pos / rot / gripper dict.

        action_pred is already unnormalized by the policy's predict_action.
        """
        dim_per_robot = action_pred.shape[1] // len(self.robot_prefixes)  # = 10
        output = {}
        for robot_idx, prefix in enumerate(self.robot_prefixes):
            start = robot_idx * dim_per_robot
            arm = action_pred[:, start : start + dim_per_robot]  # (K, 10)
            output[f"action{robot_idx}_eef_pos"] = arm[:, :3]
            output[f"action{robot_idx}_eef_rot_axis_angle"] = arm[:, 3:9]
            output[f"action{robot_idx}_gripper_width"] = arm[:, 9:]
        return output


@click.command()
@click.option("--input", "-i", required=True, help="Path to checkpoint (.ckpt)")
@click.option("--endpoint", default="tcp://0.0.0.0:18765", help="RMQ server endpoint")
@click.option("--dataset-path", default=None, help="Dataset path for prompt loading")
@click.option("--prompt-from-dataset", "prompt_from_dataset", default="train",
              type=click.Choice(["train", "val"]), show_default=True,
              help="Load prompts from the train or val split of the dataset")
@click.option("--device", default="cuda", help="Device to run on (cuda / cpu)")
@click.option("--vis-prompt", "vis_prompt_flag", is_flag=True, default=False,
              help="Visualize each prompt as an mp4 when it is loaded")
@click.option("--attention-vis", "attention_vis", is_flag=True, default=False,
              help="Enable attention visualization (logged during inference, rendered via vis_attention_map topic)")
@click.option("--allow-unseen-language", "allow_unseen_language", is_flag=True, default=False,
              help="Allow language commands not seen during training (bypasses unseen-task error)")
@click.option("--eval-dataset", "is_eval_dataset", is_flag=True, default=False,
              help="Treat the dataset as eval (unseen) — skips train/val split and uses all demos for prompting")
def main(input, endpoint, dataset_path, prompt_from_dataset, device, vis_prompt_flag, attention_vis, allow_unseen_language, is_eval_dataset):
    node = PolicyInferenceNodeRealEnv(
        ckpt_path=input,
        server_endpoint=endpoint,
        dataset_path=dataset_path,
        device=device,
        prompt_from_dataset=prompt_from_dataset,
        vis_prompt_flag=vis_prompt_flag,
        attention_vis=attention_vis,
        allow_unseen_language=allow_unseen_language,
        is_eval_dataset=is_eval_dataset,
    )
    node.run_node()


if __name__ == "__main__":
    main()
