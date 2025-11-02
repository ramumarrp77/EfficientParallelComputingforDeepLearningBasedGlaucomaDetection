"""
Script Name: ParallelProcessing/gpus_with_DDP_AMP_ModelParallel/ddp_train.py
Author: Ram Kumar Ramasamy Pandiaraj
Description:
This script sets up Distributed Data Parallel (DDP) training for the MedicalCNN model.
It manages the training, validation, and testing processes, and supports parallelism across
multiple GPUs using model parallelism and automatic mixed precision (AMP).

Function Arguments:
- rank (int): The rank of the current process in the distributed setup.
- world_size (int): The total number of processes (GPUs) involved in distributed training.
- data_dir (str): Path to the directory containing the dataset.
- hyperparams (dict): A dictionary containing hyperparameters like batch size, number of epochs, learning rate, etc.
"""

import os
import time
import json
import socket
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datetime import datetime
import matplotlib.pyplot as plt
import logging
from model import (GlaucomaDataset, get_medical_transforms, MedicalCNN)

def setup_distributed(rank, world_size):
    """
    Setup the distributed environment.
    
    Arguments:
    - rank (int): The rank of the current process.
    - world_size (int): The total number of processes (GPUs).
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_distributed():
    """
    Cleanup the distributed environment after training.
    """
    dist.destroy_process_group()

def ddp_train_and_evaluate(rank, world_size, data_dir, hyperparams):
    """
    Train and evaluate the model using Distributed Data Parallel (DDP).
    
    Arguments:
    - rank (int): Rank of the current process.
    - world_size (int): Total number of processes (GPUs).
    - data_dir (str): Path to the dataset.
    - hyperparams (dict): Dictionary containing hyperparameters (e.g., batch_size, num_epochs, etc.)
    
    Returns:
    - result (dict): Dictionary containing final performance results.
    """
    try:
        setup_distributed(rank, world_size)
        
        # Setup logging
        log_file = f"logs/training_using_{world_size}_gpus.log"
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(), 
                logging.FileHandler(log_file) 
            ]
        )
        logger = logging.getLogger(__name__)

        # Use GPU for DDP
        device = torch.device(f"cuda:{rank}")
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        torch.set_num_threads(1)  
        torch.manual_seed(42)
        np.random.seed(42)

        logger.info(f"[Rank {rank}] Loading data from {data_dir}...")
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
        y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        logger.info(f"[Rank {rank}] Data loaded.")

        train_transform, val_transform = get_medical_transforms()
        train_dataset = GlaucomaDataset(X_train, y_train, transform=train_transform)
        val_dataset = GlaucomaDataset(X_val, y_val, transform=val_transform)
        test_dataset = GlaucomaDataset(X_test, y_test, transform=val_transform)

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        batch_size = hyperparams.get('batch_size', 32)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                  num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler,
                                num_workers=4, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler,
                                 num_workers=4, pin_memory=True, persistent_workers=True)

        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        class_weights = torch.FloatTensor([1.0 / count for count in class_counts])
        class_weights = class_weights / class_weights.sum()
        class_weights = class_weights.to(device)

        num_classes = hyperparams.get('num_classes', 2)
        model = MedicalCNN(num_classes=num_classes, device_list=[f'cuda:{i}' for i in range(world_size)])
        model = model.to(device)

        # Initialize DDP for GPU
        model = DDP(model, device_ids=[rank], output_device=rank)

        total_params = sum(p.numel() for p in model.parameters())
        total_layers = model.module.count_layers()
        if rank == 0:
            logger.info(f"Model architecture: MedicalCNN")
            logger.info(f"Total layers: {total_layers}")
            logger.info(f"Total parameters: {total_params:,}")

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        initial_lr = hyperparams.get('initial_lr', 0.001)
        weight_decay = hyperparams.get('weight_decay', 1e-4)
        optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
        num_epochs = hyperparams.get('num_epochs', 10)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        # Lists to record metrics for plotting
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        start_time = time.time()
        best_val_acc = 0.0
        best_model_state = None
        for epoch in range(num_epochs):
            train_sampler.set_epoch(epoch)
            model.train()
            epoch_train_loss = 0.0
            correct_train = 0
            total_train = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_train_loss += loss.item() * data.size(0)
                total_train += data.size(0)
                _, pred = torch.max(output, 1)
                correct_train += (pred == target).sum().item()
                if batch_idx % 10 == 0 and rank == 0:
                    logger.info(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

            scheduler.step()
            train_loss = epoch_train_loss / total_train
            train_acc = 100.0 * correct_train / total_train
            if rank == 0:
                train_losses.append(train_loss)
                train_accuracies.append(train_acc)
                logger.info(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

            model.eval()
            epoch_val_loss = 0.0
            correct_val = 0
            total_val = 0
            all_targets = []
            all_probs = []
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    epoch_val_loss += loss.item() * data.size(0)
                    total_val += data.size(0)
                    _, pred = torch.max(output, 1)
                    correct_val += (pred == target).sum().item()
                    all_targets.extend(target.cpu().numpy())
                    all_probs.extend(F.softmax(output, dim=1).cpu().numpy())
            val_loss = epoch_val_loss / total_val
            val_acc = 100.0 * correct_val / total_val
            if rank == 0:
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
                try:
                    roc_auc = roc_auc_score(np.array(all_targets), np.array(all_probs)[:, 1])
                    logger.info(f"Epoch [{epoch+1}/{num_epochs}] Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%, AUC-ROC: {roc_auc:.4f}")
                except Exception as e:
                    logger.info(f"Epoch [{epoch+1}/{num_epochs}] Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

            if rank == 0 and val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.module.state_dict().copy()
                logger.info(f"New best model at epoch {epoch+1} with val accuracy: {best_val_acc:.2f}%")

        total_time = time.time() - start_time

        # Testing phase
        model.eval()
        epoch_test_loss = 0.0
        correct_test = 0
        total_test = 0
        all_test_targets = []
        all_test_probs = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                epoch_test_loss += loss.item() * data.size(0)
                total_test += data.size(0)
                _, pred = torch.max(output, 1)
                correct_test += (pred == target).sum().item()
                all_test_targets.extend(target.cpu().numpy())
                all_test_probs.extend(F.softmax(output, dim=1).cpu().numpy())
        test_loss = epoch_test_loss / total_test
        test_acc = 100.0 * correct_test / total_test
        if rank == 0:
            try:
                test_auc = roc_auc_score(np.array(all_test_targets), np.array(all_test_probs)[:, 1])
                logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%, Test AUC-ROC: {test_auc:.4f}")
            except Exception as e:
                logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

        if rank == 0:
            result = {
                "world_size": world_size,
                "train_time": total_time,
                "avg_epoch_time": total_time / num_epochs,
                "val_accuracy": best_val_acc,
                "test_accuracy": test_acc,
                "test_loss": test_loss,
                "total_time": total_time,
                "train_losses": train_losses,
                "train_accuracies": train_accuracies,
                "val_losses": val_losses,
                "val_accuracies": val_accuracies,
                "hyperparameters": hyperparams
            }
            logger.info("\n===== Final Performance Results =====")
            logger.info(json.dumps(result, indent=4))

            # Save best model
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"models/training_using_gpus_{world_size}_best_model.pt"
            if best_model_state is not None:
                torch.save(best_model_state, model_filename)
                logger.info(f"Model saved to {model_filename}")

            return result

    except Exception as e:
        logger.error(f"[Rank {rank}] Error: {e}")
        raise e
    finally:
        cleanup_distributed()
