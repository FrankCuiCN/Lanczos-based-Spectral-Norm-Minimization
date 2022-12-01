import os
import json

from .flags import FLAGS


def save_metrics():
    with open(os.path.join(FLAGS.path_result, "metrics.json"), "w") as f:
        json.dump(FLAGS.metrics, f)
