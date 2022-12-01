import os

from .flags import FLAGS


def move_config():
    os.rename(FLAGS.path_config, os.path.join(FLAGS.path_result, "config.json"))
