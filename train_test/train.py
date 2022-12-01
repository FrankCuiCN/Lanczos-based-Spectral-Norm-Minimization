import torch
import torch.nn.functional as F

from torch.autograd import grad
from utils import CONFIG, FLAGS
from differentiation import get_hvp, get_jjtvp, lanczos, power_method


def train():
    FLAGS.model.train()
    correct = 0
    total = 0
    loss_cls_sum = 0
    loss_reg_sum = 0
    for idx, (inputs, targets) in enumerate(FLAGS.trainloader):
        # zero grad
        FLAGS.model.zero_grad(set_to_none=True)
        # get data
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs = inputs.view(-1, 3072).requires_grad_()  # assume "inputs" is detached and is a clone
        # forward pass
        p = FLAGS.model(inputs)
        # calculate loss
        if CONFIG["train_mode"] == "normal":
            loss_cls = F.cross_entropy(p, targets, reduction="mean")
            loss_reg = 0
            loss = loss_cls
        elif CONFIG["train_mode"] == "combined":
            ce = F.cross_entropy(p, targets, reduction="sum")
            g = grad(ce, inputs, create_graph=True)[0]
            if CONFIG["reg_mode"] == "lanczos":
                v_j = lanczos(get_jjtvp(p, inputs), *p.shape, FLAGS.lanczos_iter_num)[1]
                v_h = lanczos(get_hvp(g, inputs), *inputs.shape, FLAGS.lanczos_iter_num)[1]
            else:
                raise ValueError("Wrong value. (CONFIG)")
            vj = grad(p, inputs, v_j, create_graph=True)[0]
            hv = grad(g, inputs, v_h, create_graph=True)[0]
            loss_reg_j = vj.square().sum(dim=1).mean()
            loss_reg_h = hv.square().sum(dim=1).mean()
            loss_cls = ce / len(inputs)
            loss_reg = (loss_reg_j + loss_reg_h) / 2
            loss = (1 - FLAGS.rp) * loss_cls + FLAGS.rp * loss_reg
        elif CONFIG["train_mode"] == "jacobian":
            if CONFIG["reg_mode"] == "lanczos":
                v = lanczos(get_jjtvp(p, inputs), *p.shape, FLAGS.lanczos_iter_num)[1]
            elif CONFIG["reg_mode"] == "power_method":
                v = power_method(get_jjtvp(p, inputs), *p.shape, FLAGS.lanczos_iter_num)
            elif CONFIG["reg_mode"] == "hutchinson":
                v = torch.randn(*p.shape, device="cuda")
            else:
                raise ValueError("Wrong value. (CONFIG)")
            vj = grad(p, inputs, v, create_graph=True)[0]
            loss_cls = F.cross_entropy(p, targets, reduction="mean")
            loss_reg = vj.square().sum(dim=1).mean()
            if CONFIG["reg_mode"] == "hutchinson":
                loss_reg = CONFIG["hutchinson_factor"] * loss_reg
            loss = (1 - FLAGS.rp) * loss_cls + FLAGS.rp * loss_reg
        elif CONFIG["train_mode"] == "hessian":
            ce = F.cross_entropy(p, targets, reduction="sum")
            g = grad(ce, inputs, create_graph=True)[0]
            if CONFIG["reg_mode"] == "lanczos":
                v = lanczos(get_hvp(g, inputs), *inputs.shape, FLAGS.lanczos_iter_num)[1]
            elif CONFIG["reg_mode"] == "power_method":
                v = power_method(get_hvp(g, inputs), *inputs.shape, FLAGS.lanczos_iter_num)
            elif CONFIG["reg_mode"] == "hutchinson":
                v = torch.randn(*inputs.shape, device="cuda")
            else:
                raise ValueError("Wrong value. (CONFIG)")
            hv = grad(g, inputs, v, create_graph=True)[0]
            loss_cls = ce / len(inputs)
            loss_reg = hv.square().sum(dim=1).mean()
            if CONFIG["reg_mode"] == "hutchinson":
                loss_reg = CONFIG["hutchinson_factor"] * loss_reg
            loss = (1 - FLAGS.rp) * loss_cls + FLAGS.rp * loss_reg
        else:
            raise ValueError("Wrong value. (CONFIG)")
        # backward pass
        loss.backward()
        FLAGS.optimizer.step()
        # calculate metrics
        correct += p.argmax(dim=1).eq(targets).sum().item()
        total += len(targets)
        loss_cls_sum += float(loss_cls) * len(targets)
        loss_reg_sum += float(loss_reg) * len(targets)
        # print progress
        print("\r", idx, end="")
    print()
    # calculate metrics
    FLAGS.metrics["loss_cls"].append(loss_cls_sum / total)
    FLAGS.metrics["loss_reg"].append(loss_reg_sum / total)
    FLAGS.metrics["train_acc"].append(correct / total)
