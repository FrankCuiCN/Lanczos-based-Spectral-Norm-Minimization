import torch.nn as nn

from utils import CONFIG


def normalization(num_channels):
    if CONFIG["normalization"] == "group_norm":
        if CONFIG["gn_mode"] == 0:
            return nn.GroupNorm(CONFIG["gn_num_groups"], num_channels)
        elif CONFIG["gn_mode"] == 1:
            return nn.GroupNorm(num_channels // CONFIG["gn_channels_per_group"], num_channels)
        elif CONFIG["gn_mode"] == 2:
            return nn.GroupNorm(min(CONFIG["gn_num_groups"], num_channels // CONFIG["gn_channels_per_group"]),
                                num_channels)
        elif CONFIG["gn_mode"] == 3:
            if num_channels >= CONFIG["gn_num_groups"]:
                return nn.GroupNorm(CONFIG["gn_num_groups"], num_channels)
            else:
                return nn.GroupNorm(num_channels, num_channels)
        else:
            raise ValueError("Wrong value. (CONFIG)")
    elif CONFIG["normalization"] == "batch_norm":
        return nn.BatchNorm2d(num_channels)
    else:
        raise ValueError("Wrong value. (CONFIG)")
