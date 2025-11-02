"""
Script Name: ParallelProcessing/gpus_with_DDP/main.py
Author: Ram Kumar Ramasamy Pandiaraj
Description:
This script serves as the entry point to run distributed training for the glaucoma classification model using DDP (Distributed Data Parallel) in GPUs. 
It initializes the distributed environment, passes the required hyperparameters, and calls the 'ddp_train_and_evaluate' function from the 'ddp_train' module to start the training process across multiple GPUs/nodes.
The script also sets the environment variables for the distributed setup, including the rank and world size.

Function Arguments:
- rank (int): The rank of the current process in the distributed setup. This is used to identify the specific GPU or node.
- world_size (int): The total number of processes across all nodes. It indicates how many GPUs will be used in the training process.
"""

import os
import socket
from ddp_train import ddp_train_and_evaluate

def main():
    """
    The main function that initializes the distributed environment and starts the training process.
    It reads the rank and world size from the environment variables, sets up the hyperparameters, 
    and invokes the `ddp_train_and_evaluate` function to perform the training.
    """
    # Read distributed rank and world size from environment variables.
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE"))
    
    data_dir = "../../preprocessed_glaucoma_data" 
    hyperparams = {
        "batch_size": 32,
        "num_epochs": 10,
        "initial_lr": 0.001,
        "weight_decay": 1e-4,
        "num_classes": 2,
        "save_model": True
    }
    
    # Start the training and evaluation process using DDP
    ddp_train_and_evaluate(rank, world_size, data_dir, hyperparams)

if __name__ == "__main__":
    main()
