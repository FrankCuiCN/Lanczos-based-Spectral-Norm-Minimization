import os
import time
import torch

from datasets import cifar10, cifar100
from models import ResNet, WideResNet
from train_test import train, test, test_adv
from utils import CONFIG, FLAGS, print_metrics, prepare_schedule, update_lr, get_name, move_config, save_metrics

# ----- #
# preparations
# ----- #
# configure pytorch backends (we assume the following are effective)
torch.backends.cuda.matmul.allow_tf32 = CONFIG["torch_allow_tf32"]
torch.backends.cudnn.allow_tf32 = CONFIG["torch_allow_tf32"]
torch.backends.cudnn.benchmark = CONFIG["torch_benchmark"]
# define dataset loaders
if CONFIG["dataset"] == "cifar10":
    FLAGS.trainloader, FLAGS.testloader = cifar10()
elif CONFIG["dataset"] == "cifar100":
    FLAGS.trainloader, FLAGS.testloader = cifar100()
else:
    raise ValueError("Wrong value. (CONFIG)")
# define model
if CONFIG["model"] == "wide_resnet":
    FLAGS.model = WideResNet().cuda()
elif CONFIG["model"] == "resnet":
    FLAGS.model = ResNet().cuda()
else:
    raise ValueError("Wrong value. (CONFIG)")
FLAGS.model = torch.nn.DataParallel(FLAGS.model)
# define optimizer
if CONFIG["optimizer"] == "sgd":
    FLAGS.optimizer = torch.optim.SGD(FLAGS.model.parameters(), lr=0.1, weight_decay=CONFIG["weight_decay"],
                                      momentum=CONFIG["sgd_momentum"], nesterov=CONFIG["sgd_nesterov"])
elif CONFIG["optimizer"] == "adamw":
    FLAGS.optimizer = torch.optim.AdamW(FLAGS.model.parameters(), lr=0.1, weight_decay=CONFIG["weight_decay"],
                                        eps=CONFIG["adamw_eps"])
elif CONFIG["optimizer"] == "adam":
    FLAGS.optimizer = torch.optim.Adam(FLAGS.model.parameters(), lr=0.1, weight_decay=CONFIG["weight_decay"],
                                       eps=CONFIG["adam_eps"])
else:
    raise ValueError("Wrong value. (CONFIG)")
# define lr_schedule
FLAGS.lr_schedule = prepare_schedule(CONFIG["lr_schedule"])
# define rp_schedule
FLAGS.rp_schedule = prepare_schedule(CONFIG["rp_schedule"])
# define lz_schedule
FLAGS.lz_schedule = prepare_schedule(CONFIG["lz_schedule"])
# define metrics
FLAGS.metrics = {"time_cost": [], "lr": [], "rp": [], "loss_cls": [], "loss_reg": [],
                 "train_acc": [], "test_acc": [], "best_acc": [], "test_adv_acc": [], "best_adv_acc": []}
# create result folder
FLAGS.path_result = os.path.join("./results", get_name())
os.mkdir(FLAGS.path_result)

# ----- #
# start training
# ----- #
# print CONFIG
print(CONFIG)
# start training
for idx_epoch in range(CONFIG["epoch_num"]):
    # record start time
    torch.cuda.synchronize()
    t1 = time.time()
    # update lr
    FLAGS.lr = FLAGS.lr_schedule[idx_epoch]
    update_lr()
    # update loss weight
    FLAGS.rp = FLAGS.rp_schedule[idx_epoch]
    # update lanczos iteration number
    FLAGS.lanczos_iter_num = round(FLAGS.lz_schedule[idx_epoch])
    # start the epoch
    print("epoch:", idx_epoch)
    train()
    test()
    if (CONFIG["train_mode"] != "normal") and (idx_epoch in CONFIG["test_adv_milestones"]):
        test_adv()
    elif idx_epoch == 0:
        FLAGS.metrics["test_adv_acc"].append(-1)
        FLAGS.metrics["best_adv_acc"].append(-1)
    else:
        FLAGS.metrics["test_adv_acc"].append(-1)
        FLAGS.metrics["best_adv_acc"].append(FLAGS.metrics["best_adv_acc"][-1])
    # record end time
    torch.cuda.synchronize()
    t2 = time.time()
    # calculate metrics
    FLAGS.metrics["time_cost"].append(t2 - t1)
    FLAGS.metrics["lr"].append(FLAGS.lr)
    FLAGS.metrics["rp"].append(FLAGS.rp)
    # print metrics
    print_metrics()
# save the result
move_config()
save_metrics()
