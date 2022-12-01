import torch
import torchattacks

from utils import FLAGS


def test_adv():
    FLAGS.model.eval()
    eps, steps = 8 / 255, 20
    alpha = 2 * eps / steps
    attack = torchattacks.PGD(FLAGS.model, eps=eps, alpha=alpha, steps=steps, random_start=True)
    correct, total = 0, 0
    for idx, (inputs, targets) in enumerate(FLAGS.testloader):
        # get data
        inputs, targets = inputs.cuda(), targets.cuda()
        # perform attack
        inputs_adv = attack(inputs, targets)
        # forward pass
        with torch.no_grad():
            p_adv = FLAGS.model(inputs_adv)
        # calculate metrics
        correct += p_adv.argmax(dim=1).eq(targets).sum().item()
        total += len(targets)
        # print progress
        print("\r", idx, end="")
    print()
    # calculate metrics
    FLAGS.metrics["test_adv_acc"].append(correct / total)
    if len(FLAGS.metrics["best_adv_acc"]) == 0:
        FLAGS.metrics["best_adv_acc"].append(FLAGS.metrics["test_adv_acc"][-1])
    elif FLAGS.metrics["best_adv_acc"][-1] <= FLAGS.metrics["test_adv_acc"][-1]:
        FLAGS.metrics["best_adv_acc"].append(FLAGS.metrics["test_adv_acc"][-1])
    else:
        FLAGS.metrics["best_adv_acc"].append(FLAGS.metrics["best_adv_acc"][-1])
