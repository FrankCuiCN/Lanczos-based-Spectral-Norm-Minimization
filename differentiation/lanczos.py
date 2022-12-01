import torch

from .random_directions import random_directions


def lanczos(matvec, num_batch, d, num_iter=32, mode="abs", eps=1e-5, device="cuda"):
    # note: for matvec:
    #     input  shape: (-1, d)
    #     output shape: (-1, d)
    # note: mode: "max", "min", or "abs"
    a = torch.zeros((num_iter, num_batch), dtype=torch.float, device=device)
    b = torch.zeros((num_iter, num_batch), dtype=torch.float, device=device)  # note: b[0] is not used
    v = torch.zeros((num_iter, num_batch, d), dtype=torch.float, device=device)
    t = torch.zeros((num_iter, num_batch, num_iter), dtype=torch.float, device=device)

    v[0] = random_directions(num_batch, d, device=device)
    tmp = matvec(v[0])
    a[0] = (tmp * v[0]).sum(dim=1)
    tmp = tmp - a[0][:, None] * v[0]

    for idx in range(1, num_iter):
        b[idx] = torch.linalg.norm(tmp, dim=1)
        if b[idx].min() >= eps:
            v[idx] = tmp / b[idx][:, None]
        else:
            mask = b[idx] >= eps
            v[idx][mask] = tmp[mask] / b[idx][:, None][mask]
            v[idx][~mask] = random_directions((~mask).sum(), d, device=device)  # note: this is a simplification
        tmp = matvec(v[idx])
        a[idx] = (tmp * v[idx]).sum(dim=1)
        tmp = tmp - a[idx][:, None] * v[idx] - b[idx][:, None] * v[idx - 1]

    t[range(num_iter), :, range(num_iter)] = a
    t[range(1, num_iter), :, range(0, num_iter - 1)] = b[1:]
    t[range(0, num_iter - 1), :, range(1, num_iter)] = b[1:]
    t = t.permute((1, 0, 2))
    v = v.permute((1, 0, 2))

    evals, evecs = torch.linalg.eigh(t)
    if mode == "max":
        idxs = evals.argmax(dim=1)  # note: evals and evecs are already sorted, argmax is not needed
    elif mode == "min":
        idxs = evals.negative().argmax(dim=1)  # note: evals and evecs are already sorted, argmax is not needed
    elif mode == "abs":
        idxs = evals.abs().argmax(dim=1)
    else:
        raise ValueError("Wrong value. (lanczos)")
    evals = torch.take_along_dim(evals, idxs.view(-1, 1), dim=1)[:, 0]
    evecs = evecs.permute((0, 2, 1))
    evecs = torch.take_along_dim(evecs, idxs.view(-1, 1, 1), dim=1)
    evecs = torch.bmm(evecs, v)[:, 0]
    evecs = evecs / torch.linalg.norm(evecs, dim=1)[:, None]
    return evals, evecs
