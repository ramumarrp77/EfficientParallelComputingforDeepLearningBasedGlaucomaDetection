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
from model import (GlaucomaDataset, get_medical_transforms, MedicalCNN)
import logging
import socket

def setup_logger(num_cpus, num_threads):
    # Set up logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create the log file with the name based on the number of CPUs
    log_file = f"logs/training_using_cpus_{num_cpus}_log.txt"
    hostname = socket.gethostname()

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    # Log the running node information
    logger.info(f"Running on node: {hostname}")
    logger.info(f"Using {num_threads} threads and {num_cpus} processes for training.")

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

def setup_distributed(rank, world_size, num_threads):
    # Set the necessary environment variables
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    # Set the number of threads for CPU
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)
    
    # Initialize the process group for DDP
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

def cleanup_distributed():
    # Destroy the process group when done
    dist.destroy_process_group()

def ddp_train_and_evaluate(rank, world_size, data_dir, hyperparams, num_threads, logger):
    try:
        setup_distributed(rank, world_size, num_threads)

        dist.barrier()
        logger.info(f"[Rank {rank}] Passed initial barrier")

        # Use the CPU for DDP
        device = torch.device("cpu")
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
        
        # Set up DistributedSampler for each dataset
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        
        batch_size = hyperparams.get('batch_size', 32)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                  num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=None)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler,
                                num_workers=4, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler,
                                 num_workers=4, pin_memory=True, persistent_workers=True)
        
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        class_weights = torch.FloatTensor([1.0 / count for count in class_counts])
        class_weights = class_weights / class_weights.sum()
        class_weights = class_weights.to(device)
        
        num_classes = hyperparams.get('num_classes', 2)
        model = MedicalCNN(num_classes=num_classes)
        model = model.to(device)
        
        # Initialize DDP for CPU
        model = DDP(model)

        # Print model details
        total_params = sum(p.numel() for p in model.parameters())
        total_layers = model.module.count_layers()
        if rank == 0:
            logger.info(f"Model architecture: MedicalCNN")
            logger.info(f"Total layers: {total_layers}")
            logger.info(f"Total parameters: {total_params:,}")
        
        # Loss and optimizer
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
            model_filename = f"models/training_using_cpus_{world_size}_best_model.pt"
            if best_model_state is not None:
                torch.save(best_model_state, model_filename)
                logger.info(f"Model saved to {model_filename}")
            
            # Plot training & validation loss curves
            epochs_range = range(1, num_epochs+1)
            plt.figure()
            plt.plot(epochs_range, train_losses, label='Train Loss')
            plt.plot(epochs_range, val_losses, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f"Training using CPUs_{world_size} Loss Curve")
            plt.legend()
            loss_plot_filename = f"plots/training_using_cpus_{world_size}_loss.png"
            plt.savefig(loss_plot_filename)
            plt.close()
            logger.info(f"Loss plot saved as {loss_plot_filename}")
            
            # Plot training & validation accuracy curves
            plt.figure()
            plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
            plt.plot(epochs_range, val_accuracies, label='Val Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.title(f"Training using CPUs_{world_size} Accuracy Curve")
            plt.legend()
            acc_plot_filename = f"plots/training_using_cpus_{world_size}_accuracy.png"
            plt.savefig(acc_plot_filename)
            plt.close()
            logger.info(f"Accuracy plot saved as {acc_plot_filename}")
            
            # Save runtime parameters and metrics as JSON
            params_filename = f"metrics/training_using_cpus_{world_size}_params.json"
            with open(params_filename, 'w') as f:
                json.dump(result, f, indent=4)
            logger.info(f"Runtime parameters saved as {params_filename}")
            
            return result

    except Exception as e:
        logger.error(f"[Rank {rank}] Error: {e}")
        raise e
    finally:
        cleanup_distributed()
