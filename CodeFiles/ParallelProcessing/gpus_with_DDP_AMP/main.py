"""
Script Name:  ParallelProcessing/gpus_with_DDP_AMP/main.py
Author: Ram Kumar Ramasamy Pandiaraj
Description:
This script initializes the distributed environment and coordinates the training of a deep learning model using PyTorch's DistributedDataParallel (DDP) and AMP. 
It sets up logging, loads the preprocessed glaucoma data, and calls the 'ddp_train_and_evaluate' function to train the model on multiple GPUs with AMP (Automatic Mixed Precision) for faster computation.
The model is evaluated on both validation and test datasets, and results are logged for each rank.

Function Arguments:
- rank (int): The rank of the current process (GPU).
- world_size (int): The total number of processes participating in training (number of GPUs).
- data_dir (str): The directory where the preprocessed glaucoma data is stored.
- hyperparams (dict): A dictionary containing hyperparameters for training, such as:
  - batch_size (int): Batch size for training.
  - num_epochs (int): Number of epochs for training.
  - initial_lr (float): Initial learning rate for the optimizer.
  - weight_decay (float): Weight decay for the optimizer.
  - num_classes (int): Number of output classes for classification.
  - save_model (bool): Whether to save the best model during training.
"""

import os
import socket
import logging
from ddp_train import ddp_train_and_evaluate

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to set up the distributed environment and call the training function.
    It reads the rank and world size from environment variables and sets the hyperparameters for training.
    """
    hostname = socket.gethostname()
    logger.info(f"Running on node: {hostname}")
    
    # Read distributed rank and world size from environment variables
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE"))
    logger.info(f"Running with world_size - GPU with AMP: {world_size} (Rank: {rank})")
    
    data_dir = "../../preprocessed_glaucoma_data"  # Path to your data directory
    hyperparams = {
        "batch_size": 32,
        "num_epochs": 10,
        "initial_lr": 0.001,
        "weight_decay": 1e-4,
        "num_classes": 2,
        "save_model": True
    }
    
    # Call the function to train and evaluate using DistributedDataParallel (DDP)
    ddp_train_and_evaluate(rank, world_size, data_dir, hyperparams)

if __name__ == "__main__":
    main()
