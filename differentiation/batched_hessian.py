import torch

from torch.autograd import grad


def batched_hessian(f, x, out, ibs=32):
    # note: we expect f to have output shape (-1, 1)
    # note: we expect x to have shape (num_batch, num_inputs)
    # note: we expect out to have shape (-1, num_inputs, num_inputs)
    #     for out, we recommend to set pin_memory to True if out is on CPU
    # note: ibs: internal batch size
    # note: we assume that num_inputs % ibs == 0
    num_batch, num_inputs = x.shape
    assert num_inputs % ibs == 0

    v_all = torch.eye(num_inputs, dtype=torch.float, device="cuda")
    for idx1 in range(num_batch):
        x_tmp = x[idx1:idx1 + 1].repeat(ibs, 1)
        p = f(x_tmp)
        g = grad(p.sum(), x_tmp, create_graph=True)[0]
        for idx2 in range(num_inputs // ibs):
            v = v_all[idx2 * ibs:(idx2 + 1) * ibs]
            hv = grad(g, x_tmp, v, retain_graph=True)[0]
            out[idx1, idx2 * ibs:(idx2 + 1) * ibs] = hv
    return out
