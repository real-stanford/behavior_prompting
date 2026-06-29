from behavior_prompting.common.imagecodecs_numcodecs import register_codecs
register_codecs(verbose=False)

import os
import logging

def fix_robosuite_log_permission_issue():
    # Patch to fix permission issue where multiple users want to write to /tmp/robosuite.log by instead rerouting the file to be in the user directory
    default_log_path = "/tmp/robosuite.log"
    user_log_dir = os.path.expanduser("~/.robosuite")
    os.makedirs(user_log_dir, exist_ok=True)
    user_log_path = os.path.join(user_log_dir, "robosuite.log")

    original_file_handler_init = logging.FileHandler.__init__

    def patched_file_handler_init(self, filename, *args, **kwargs):
        if filename == default_log_path:
            filename = user_log_path
        return original_file_handler_init(self, filename, *args, **kwargs)

    logging.FileHandler.__init__ = patched_file_handler_init
