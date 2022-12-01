import torch

from utils import FLAGS


def test():
    FLAGS.model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(FLAGS.testloader):
            # get data
            inputs, targets = inputs.cuda(), targets.cuda()
            # forward pass
            p = FLAGS.model(inputs)
            # calculate metrics
            correct += p.argmax(dim=1).eq(targets).sum().item()
            total += len(targets)
            # print progress
            print("\r", idx, end="")
    print()
    # calculate metrics
    FLAGS.metrics["test_acc"].append(correct / total)
    if len(FLAGS.metrics["best_acc"]) == 0:
        FLAGS.metrics["best_acc"].append(FLAGS.metrics["test_acc"][-1])
    elif FLAGS.metrics["best_acc"][-1] <= FLAGS.metrics["test_acc"][-1]:
        FLAGS.metrics["best_acc"].append(FLAGS.metrics["test_acc"][-1])
    else:
        FLAGS.metrics["best_acc"].append(FLAGS.metrics["best_acc"][-1])
