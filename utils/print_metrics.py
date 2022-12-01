from .flags import FLAGS


def print_metrics():
    message = ""
    for key in FLAGS.metrics.keys():
        message += key
        message += ": "
        if key == "lr":
            message += str(round(FLAGS.metrics[key][-1], 7))
        else:
            message += str(round(FLAGS.metrics[key][-1], 3))
        message += "; "
    print(message)
