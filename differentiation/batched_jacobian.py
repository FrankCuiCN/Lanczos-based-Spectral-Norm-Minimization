import torch
import torch.nn.functional as F

from torch.autograd import grad


def batched_jacobian(f, x, out, ibs=32):
    # note: we expect f to have output shape (-1, num_outputs)
    # note: we expect x to have shape (num_batch, num_inputs)
    # note: we expect out to have shape (-1, num_outputs, num_inputs)
    #     for out, we recommend to set pin_memory to True if out is on CPU
    # note: ibs: internal batch size
    # note: we assume that num_outputs % ibs == 0
    num_batch = len(x)
    _, num_outputs, num_inputs = out.shape
    assert num_outputs % ibs == 0

    v_all = F.one_hot(torch.tensor(range(num_outputs))).float().cuda()
    for idx1 in range(num_batch):
        x_tmp = x[idx1:idx1 + 1].repeat(ibs, 1)
        p = f(x_tmp)
        for idx2 in range(num_outputs // ibs):
            v = v_all[idx2 * ibs:(idx2 + 1) * ibs]
            jv = grad(p, x_tmp, v, retain_graph=True)[0]
            out[idx1, idx2 * ibs:(idx2 + 1) * ibs] = jv
    return out
