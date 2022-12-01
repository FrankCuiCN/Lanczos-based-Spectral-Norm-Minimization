from .flags import FLAGS


def update_lr():
    for param_group in FLAGS.optimizer.param_groups:
        param_group['lr'] = FLAGS.lr
