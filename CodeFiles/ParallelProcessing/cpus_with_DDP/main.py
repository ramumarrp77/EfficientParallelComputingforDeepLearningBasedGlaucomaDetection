import torch
import torch.multiprocessing as mp
from ddp_train import ddp_train_and_evaluate, setup_logger
import argparse

def worker(rank, world_size, data_dir, hyperparams, num_threads, logger):
    ddp_train_and_evaluate(rank, world_size, data_dir, hyperparams, num_threads, logger)

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Distributed Training with DDP on CPUs')
    parser.add_argument('--world_size', type=int, default=2, help='Number of processes (world size)')
    parser.add_argument('--num_threads', type=int, default=4, help='Number of threads per process')

    # Parse arguments
    args = parser.parse_args()

    world_size = args.world_size
    num_threads = args.num_threads
    data_dir = "../../preprocessed_glaucoma_data"
    hyperparams = {
        "batch_size": 32,
        "num_epochs": 10,
        "initial_lr": 0.001,
        "weight_decay": 1e-4,
        "num_classes": 2,
        "save_model": True
    }

    # Set up logger with num_threads for logging in file and console
    logger = setup_logger(world_size,num_threads)

    processes = []

    # Create a separate process for each rank.
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank, world_size, data_dir, hyperparams, num_threads, logger))
        p.start()
        processes.append(p)
    
    # Ensure all processes complete.
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
