"""Data Loader"""
import logging
import sys

import torch
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms
from torchvision.datasets import MNIST

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def get_train_data_loader(batch_size, training_dir, is_distributed, **kwargs):
    logger.info("Get train data loader", training_dir)

    mnist_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (1.0,))
    ])
    dataset = MNIST(training_dir, transform=mnist_transform, train=True)
#     dataset = MNIST("./data", transform=mnist_transform, train=True, download=True)
    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **kwargs
    )


def get_test_data_loader(test_batch_size, test_dir, **kwargs):
    logger.info("Get test data loader", test_dir)
    mnist_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (1.0,))
    ])
    dataset = MNIST(test_dir, transform=mnist_transform, train=False)
#     dataset = MNIST("./data", transform=mnist_transform, train=False, download=True)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=test_batch_size,
        shuffle=True,
        **kwargs
    )
