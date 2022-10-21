"""Evaluation script for measuring mean squared error."""
import os
import json
import argparse
import logging
import pathlib
import tarfile

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from code.model import Net
from code import dataloader

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )
    return test_loss, accuracy


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(Net())
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


if __name__ == "__main__":
    logger.debug("Starting evaluation.")
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="/opt/ml/processing/test",
        help="input batch size for testing (default: 1000)",
    )
    args = parser.parse_args()
    
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=".")

    logger.debug("Loading MNIST model.")
    model = model_fn("./")

    logger.debug("Reading test data.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    testloader = dataloader.get_test_data_loader(args.test_batch_size, args.test)

    logger.info("Calculating nll loss, accuracy of predictions against test data.")
    nll_loss, accuracy = test(model, testloader, device)
    

    report_dict = {
        "classification_metrics": {
            "nll_loss": {
                "value": nll_loss
            },
            "accuracy": {
                "value": accuracy
            },
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing out evaluation report with nll loss: %f", nll_loss)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))

