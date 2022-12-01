import torch


def random_directions(n, d, device="cuda"):
    # sample from a unit hypersphere
    # (note): we assume d is large, because nan may occur if a vector is too close to the origin
    v = torch.randn(n, d, dtype=torch.float, device=device)
    return v / torch.linalg.norm(v, dim=1)[:, None]
