from typing import Dict
from behavior_prompting.train_network.model.common.base_policy import BasePolicy

class BaseRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BasePolicy, enable_expensive_vis: bool=True) -> Dict:
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
