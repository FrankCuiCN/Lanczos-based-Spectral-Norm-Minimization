import torch
import torchvision
import torchvision.transforms as transforms

from utils import CONFIG


def cifar10():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG["batch_size_train"],
                                              shuffle=True, num_workers=2, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=CONFIG["batch_size_test"],
                                             shuffle=False, num_workers=2)
    return trainloader, testloader
