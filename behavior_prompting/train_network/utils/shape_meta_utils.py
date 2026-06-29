# TODO: we should use this function in many other places. right now it is only used in the prompt_obs_encoder.py and similar functionality is replicated in other places which can be unified using this function.
def get_obs_keys_from_shape_meta(shape_meta: dict, for_policy: bool=True, skip_prompt_proprio: bool=False, skip_prompt_observation: bool=False) -> dict[str, list[str]]:
    """
    There are three types of places were the observations are used that each can have different sets of keys present:
    - current_obs: used in both prompting and non-prompting policies as the main keys in the current observation that go into the policy
    - prompt: used to form the prompt sequence used in prompting models
    - prompt_current_obs: used to form the current observation specifically used the prompt encoder which can differ from what is used in `current_obs` the action decoding part of the policy
    """
    current_obs_rgb_keys = []
    current_obs_low_dim_keys = []
    prompt_rgb_keys = []
    prompt_low_dim_keys = []
    prompt_current_obs_rgb_keys = []
    prompt_current_obs_low_dim_keys = []
    for key in shape_meta['obs']:
        key_type = shape_meta['obs'][key]['type']
        prompt_type = shape_meta['obs'][key].get('prompt_type', 'ignore')
        ignore_by_policy = shape_meta['obs'][key].get('ignore_by_policy', False)
        ignore_by_prompt = prompt_type == 'ignore'
        include_in_prompt_current_obs = shape_meta['obs'][key].get('include_in_prompt_current_obs', True)
        include_in_receding_obs = shape_meta['obs'][key].get('include_in_receding_obs', True)

        if for_policy and ignore_by_policy:
            continue

        if key_type == 'low_dim':
            if include_in_receding_obs:
                current_obs_low_dim_keys.append(key)
            if include_in_prompt_current_obs:
                prompt_current_obs_low_dim_keys.append(key)
            if not ignore_by_prompt and not skip_prompt_proprio:
                assert prompt_type == 'proprioception'
                prompt_low_dim_keys.append(key)
        elif key_type == 'rgb':
            if include_in_receding_obs:
                current_obs_rgb_keys.append(key)
            if include_in_prompt_current_obs:
                prompt_current_obs_rgb_keys.append(key)
            if not ignore_by_prompt and not skip_prompt_observation:
                assert prompt_type == 'observation'
                prompt_rgb_keys.append(key)

    result = {
        'current_obs_rgb': current_obs_rgb_keys,
        'current_obs_low_dim': current_obs_low_dim_keys,
        'current_obs_all': current_obs_rgb_keys + current_obs_low_dim_keys,
        'prompt_rgb': prompt_rgb_keys,
        'prompt_low_dim': prompt_low_dim_keys,
        'prompt_all': prompt_rgb_keys + prompt_low_dim_keys,
        'prompt_current_obs_rgb': prompt_current_obs_rgb_keys,
        'prompt_current_obs_low_dim': prompt_current_obs_low_dim_keys,
        'prompt_current_obs_all': prompt_current_obs_rgb_keys + prompt_current_obs_low_dim_keys
    }
    return result
