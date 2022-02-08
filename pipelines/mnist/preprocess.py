"""Prepare Mnist Dataset"""
import argparse
import logging
import pathlib
import os
from torchvision import transforms
from torchvision.datasets import MNIST


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    os.system("pwd")
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--arg", type=str, required=False)
    args = parser.parse_args()
    arg = args.arg
    
    logger.debug("Defining transforms")
    mnist_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (1.0,))
    ])
    
    base_dir = "/opt/ml/processing"

    logger.debug("Downloading and applying transforms")
    MNIST(f'{base_dir}/train', transform=mnist_transform, train=True, download=True)
    MNIST(f'{base_dir}/test', transform=mnist_transform, train=False, download=True)
