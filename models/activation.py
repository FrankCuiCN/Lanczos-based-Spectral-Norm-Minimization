import torch.nn as nn

from utils import CONFIG


def activation():
    if CONFIG["activation"] == "softplus":
        return nn.Softplus(beta=CONFIG["sp_beta"], threshold=CONFIG["sp_thr"])
    elif CONFIG["activation"] == "relu":
        return nn.ReLU()
    else:
        raise ValueError("Wrong value. (CONFIG)")
