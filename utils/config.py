import os
import json

from .flags import FLAGS

dir_config_all = [tmp for tmp in sorted(os.listdir("./config_all")) if tmp.endswith(".json")]
if len(dir_config_all) == 0:
    print("No config files left. Exiting...")
    exit(1)

FLAGS.path_config = os.path.join("./config_all", dir_config_all[0])
with open(FLAGS.path_config, "r") as f:
    CONFIG = json.load(f)
