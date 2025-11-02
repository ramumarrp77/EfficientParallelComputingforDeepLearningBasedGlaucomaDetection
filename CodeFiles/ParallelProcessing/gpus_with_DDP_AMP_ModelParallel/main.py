"""
Script Name: ParallelProcessing/gpus_with_DDP_AMP_ModelParallel/main.py
Author: Ram Kumar Ramasamy Pandiaraj
Description:
This script initializes the distributed environment and starts the training and evaluation 
process for the MedicalCNN model using Distributed Data Parallel (DDP), automatic mixed precision (AMP) and Model Parallelism.

Function Arguments:
- None: This script fetches environment variables for rank and world_size.
"""

import os
import socket
from ddp_train import ddp_train_and_evaluate

def main():
    """
    Main function to setup distributed environment and trigger training/evaluation.
    
    This function gets the hostname, retrieves rank and world size from environment 
    variables, sets hyperparameters, and calls the training function.
    """
    # Get the hostname of the machine
    hostname = socket.gethostname()
    print(f"Running on node: {hostname}")
    
    # Read distributed rank and world size from environment variables
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE"))
    print(f"Running with world_size: {world_size} (Rank: {rank})")
    
    # Path to the directory containing preprocessed glaucoma dataset
    data_dir = "../../preprocessed_glaucoma_data"
    
    # Define hyperparameters for the training
    hyperparams = {
        "batch_size": 32,
        "num_epochs": 10,
        "initial_lr": 0.001,
        "weight_decay": 1e-4,
        "num_classes": 2,
        "save_model": True
    }
    
    # Call the DDP training and evaluation function
    ddp_train_and_evaluate(rank, world_size, data_dir, hyperparams)

if __name__ == "__main__":
    main()
