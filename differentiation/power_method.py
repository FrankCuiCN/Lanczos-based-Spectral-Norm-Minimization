import torch

from .random_directions import random_directions


def power_method(matvec, n, d, num_iter=32, eps=1e-5):
    v = random_directions(n, d)
    for idx in range(num_iter):
        # perform matrix-vector product
        v = matvec(v)

        # perform vector normalization
        norm_v = torch.linalg.norm(v, dim=1)
        if norm_v.min() >= eps:
            v = v / norm_v[:, None]
        else:
            mask1 = norm_v >= eps
            mask2 = ~mask1
            v[mask1] = v[mask1] / norm_v[:, None][mask1]
            v[mask2] = random_directions(mask2.sum(), d)
    return v
