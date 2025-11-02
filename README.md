# 🔬 Efficient Parallel Computing for Deep Learning-Based Glaucoma Detection

<div align="center">

Developed for High-Performance ML/AI Research

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/Numpy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)


</div>

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Project Scope](#-project-scope)
- [Dataset](#-dataset)
- [Technical Architecture](#-technical-architecture)
- [Experiments & Methodology](#-experiments--methodology)
- [Parallel Processing Techniques](#-parallel-processing-techniques)
- [Results & Analysis](#-results--analysis)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Key Findings](#-key-findings)
- [References](#-references)

---

## 🎯 Project Overview

This project implements and analyzes **efficient parallel computing strategies** for training deep learning models on medical imaging tasks, specifically **glaucoma detection from fundus images**. The research systematically compares serial vs parallel execution across CPU and GPU architectures, evaluating the impact of **Distributed Data Parallel (DDP)**, **Automatic Mixed Precision (AMP)**, and **Model Parallelism** on training time, speedup, and efficiency.

**👨‍🎓 Author:** Ram Kumar Ramasamy Pandiaraj  
**👨‍🏫 Guide:** Dr. Handan Liu (Teaching Professor - Northeastern University)

---

## 🔬 Project Scope

### 🎯 Objectives

1. **Enhance Data Processing Efficiency**
   - Build custom preprocessing pipeline with CLAHE, gamma correction, and denoising
   - Implement class-aware augmentation to address dataset imbalance
   - Improve optic disc visibility and reduce noise for better model convergence

2. **Improve Scalability & Resource Utilization**
   - Design training pipeline utilizing parallel computing on CPUs and GPUs
   - Implement Distributed Data Parallelism (DDP) for multi-process training
   - Optimize resource allocation across compute nodes

3. **Optimize Deep Learning Model Training**
   - Train custom 129-layer CNN with 22M+ parameters
   - Integrate Squeeze-and-Excitation blocks and residual connections
   - Implement mixed precision training (AMP) for faster convergence

4. **Benchmark Performance & Scalability**
   - Compare training time across serial and parallel implementations
   - Analyze speedup and efficiency metrics for different configurations
   - Identify optimal hardware setup for medical image analysis

### 🔑 Key Research Questions

- How does parallelization impact training time for deep medical imaging models?
- What is the optimal configuration for CPU vs GPU-based training?
- How do DDP, AMP, and Model Parallelism affect speedup and efficiency?
- What are the trade-offs between adding more workers/GPUs and training efficiency?

---

## 📊 Dataset

**Dataset:** [SMDG-19 Fundus Image Dataset](https://www.kaggle.com/datasets/sabari50312/fundus-pytorch/)

### 📈 Dataset Characteristics

- **Total Images:** 17,242 high-resolution color fundus images
- **Resolution:** 512×512 pixels (preprocessed to 224×224)
- **Classes:** Binary classification (Normal vs Glaucoma)
- **Format:** PNG images with 3 RGB channels

### 📂 Data Splits

| Split | Normal (Class 0) | Glaucoma (Class 1) | Total |
|-------|------------------|---------------------|-------|
| **Train** | 5,293 (61%) | 3,328 (39%) | 8,621 |
| **Validation** | 3,539 (61%) | 2,208 (39%) | 5,747 |
| **Test** | 1,754 (61%) | 1,120 (39%) | 2,874 |

### 🔄 Preprocessing Pipeline

1. **Resizing:** 512×512 → 224×224 pixels
2. **Normalization:** Pixel values scaled to [0, 1]
3. **CLAHE:** Contrast enhancement on LAB color space
4. **Gamma Correction:** Brightness adjustment (γ = 1.2)
5. **Median Blur:** Noise reduction (3×3 kernel)
6. **Class-Aware Augmentation:** Random rotations, flips, brightness/contrast adjustments

---

## 🏗️ Technical Architecture

### 🧠 Model: MedicalCNN

Custom deep convolutional neural network designed specifically for fundus image analysis.

**Architecture Highlights:**
- **Total Layers:** 129
- **Parameters:** ~22.5 million
- **Key Components:**
  - Convolutional Stem (Conv → BN → ReLU → MaxPool)
  - 4 Residual Stages (ResNet-style blocks)
  - Squeeze-and-Excitation (SE) Attention Modules
  - Multi-Dilated Convolution Block (ASPP-inspired)
  - Global Average Pooling + Deep MLP Head with Dropout

**Design Principles:**
- **Residual Connections:** Combat vanishing gradients, enable deeper learning
- **SE Blocks:** Model channel-wise interdependencies, reweight feature importance
- **Multi-Dilated Convolutions:** Capture multi-scale retinal features (optic disc, vessels)
- **Regularization:** BatchNorm + Dropout to prevent overfitting

---

## 🧪 Experiments & Methodology

### 📋 Experimental Configurations

| Configuration | Hardware | Batch Size | Workers/GPUs |
|---------------|----------|------------|--------------|
| **Serial CPU** | 1 CPU core | 32 | 1 |
| **DDP CPU** | Multiple cores | 32 | 2, 4, 8, 16, 24, 32 |
| **Serial GPU** | P100, V100 | 16, 32, 64 | 1 |
| **DDP GPU** | V100 SXM-2 | 32 | 1, 2, 4 |
| **DDP + AMP** | V100 SXM-2 | 32 | 1, 2, 4 |
| **DDP + AMP + Model Parallel** | V100 SXM-2 | 32 | 2, 4 |

### 🔬 Experimental Setup

**Cluster Environment:** Northeastern University's OOD Explorer
- **CPU Node (d0010):** 32 CPU cores, 50GB memory
- **GPU Nodes:** NVIDIA P100, V100 SXM-2
- **Backend:** Gloo (CPU), NCCL (GPU)

**Training Configuration:**
- **Optimizer:** Adam with weight decay
- **Loss Function:** Cross-Entropy
- **Learning Rate Scheduler:** Cosine Annealing
- **Gradient Clipping:** Max norm = 1.0
- **Epochs:** 20

---

## ⚡ Parallel Processing Techniques

### 1️⃣ Distributed Data Parallel (DDP)

**Concept:** Replicate the model across multiple processes/devices, with each process handling a unique data partition.

**Implementation:**
- Each process maintains a full copy of the model
- `DistributedSampler` ensures non-overlapping data distribution
- Gradients are synchronized via `all-reduce` during backpropagation
- All processes converge to the same model state

**Benefits:**
- ✅ Linear speedup with small datasets
- ✅ No model modification required
- ✅ Efficient gradient averaging

**Code Example:**
```python
# Initialize process group
torch.distributed.init_process_group(
    backend='gloo',  # or 'nccl' for GPU
    init_method='env://',
    world_size=world_size,
    rank=rank
)

# Wrap model with DDP
model = MedicalCNN()
model = DistributedDataParallel(model)

# Use DistributedSampler
train_sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, sampler=train_sampler)
```

### 2️⃣ Automatic Mixed Precision (AMP)

**Concept:** Use FP16 (half precision) for most operations, maintaining FP32 for sensitive computations.

**Implementation:**
- `torch.cuda.amp.autocast()` automatically casts operations to FP16
- `GradScaler` prevents underflow during gradient updates
- Compatible with DDP for multi-GPU training

**Benefits:**
- ✅ 1.5-2x faster training on Tensor Core GPUs
- ✅ Reduced memory consumption (larger batch sizes possible)
- ✅ Minimal accuracy loss

**Code Example:**
```python
scaler = torch.cuda.amp.GradScaler()

for data, target in train_loader:
    optimizer.zero_grad()
    
    # Mixed precision forward pass
    with torch.cuda.amp.autocast():
        output = model(data)
        loss = criterion(output, target)
    
    # Scaled backward pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 3️⃣ Model Parallelism

**Concept:** Split the model layers across multiple GPUs when the model is too large for a single GPU.

**Implementation:**
- Partition MedicalCNN into stages assigned to different GPUs
- GPU 0: Stem, Stage2, Stage4, Global Pooling, FC Head
- GPU 1: Stage1, Stage3 (Residual + MultiDilated blocks)
- Forward pass moves data across GPUs sequentially

**Benefits:**
- ✅ Train larger models that don't fit in single GPU memory
- ✅ Balanced memory distribution

**Trade-offs:**
- ❌ Increased communication overhead
- ❌ Not necessary for models that fit in one GPU

---

## 📊 Results & Analysis

### 🎯 Optimal Batch Size Selection

| Batch Size | Training Time | Test Accuracy |
|------------|---------------|---------------|
| 16 | 9.84 min | 86.57% |
| **32** | **8.5 min** | **86.85%** ✅ |
| 64 | 7.8 min | 86.19% |

**Selected:** Batch size 32 (best balance of speed and accuracy)

### 💻 CPU Training Performance

| Workers | Training Time | Speedup | Efficiency |
|---------|---------------|---------|------------|
| 1 | 756.0 min | 1.00x | 1.00 |
| 2 | 383.4 min | 1.97x | 0.97 |
| 4 | 213.2 min | 3.55x | 0.89 |
| 8 | 115.4 min | 6.55x | 0.82 |
| 16 | 71.8 min | 10.53x | 0.66 |
| 24 | 64.5 min | 11.72x | 0.49 |
| **32** | **62.3 min** | **12.13x** | **0.38** |

**Key Insights:**
- 🚀 12x speedup with 32 workers vs serial execution
- 📉 Efficiency drops beyond 8 workers due to communication overhead
- ✅ **Optimal configuration: 8-16 workers** (best speedup/efficiency balance)

### 🎮 GPU Training Performance

#### Serial GPU Comparison

| GPU | Training Time | Speedup vs CPU |
|-----|---------------|----------------|
| P100 | 15.28 min | 49.5x |
| V100 SXM-2 | 8.56 min | 88.3x |
| **V100 + AMP** | **7.18 min** | **105.3x** ✅ |

#### Multi-GPU with DDP

| GPUs | V100 (min) | V100 + AMP (min) | Speedup (AMP) |
|------|------------|------------------|---------------|
| 1 | 8.56 | 7.18 | 1.00x |
| **2** | **4.66** | **3.82** | **1.88x** ✅ |
| 4 | 3.54 | 3.30 | 2.18x |

#### DDP + AMP + Model Parallel

| GPUs | Training Time | Speedup | Efficiency |
|------|---------------|---------|------------|
| 2 | 6.50 min | 1.39x | 0.69 |
| 4 | 4.20 min | 1.75x | 0.44 |

**Analysis:** Model Parallelism adds overhead for this project as the model fits in single GPU memory. **Not recommended** for this dataset size.

### 📈 Performance Comparison Summary

| Configuration | Training Time | Speedup | Best Use Case |
|---------------|---------------|---------|---------------|
| Serial CPU | 756.0 min | 1.00x | Baseline |
| 8 CPUs (DDP) | 115.4 min | 6.55x | CPU-only environments |
| V100 GPU | 8.56 min | 88.3x | Single GPU training |
| **2× V100 + AMP** | **3.82 min** | **197.9x** | **Optimal** ✅ |
| 4× V100 + AMP | 3.30 min | 229.1x | Marginal gains |

---

## 📁 Repository Structure

```
EfficientParallelComputingforDeepLearningBasedGlaucomaDetection/
│
├── 📂 Analysis/
│   ├── CPU-Comparison.ipynb              # CPU performance analysis
│   └── GPU-Comparison.ipynb              # GPU performance analysis
│
├── 📂 data_preprocessing/
│   └── data_preprocessing.ipynb          # Image preprocessing pipeline
│
├── 📂 SerialProcessing/
│   ├── 📂 cpu/
│   │   ├── SerialExecutionCPUs.ipynb     # Serial CPU training
│   │   └── logs/models/metrics/plots/     # Training artifacts
│   └── 📂 gpu/
│       ├── SerialExecutionGPUs.ipynb     # Serial GPU training
│       ├── SerialExecutionGPUs-BatchSize.ipynb
│       ├── SerialExecutionGPUs_AMP.ipynb
│       └── logs/models/metrics/plots/
│
├── 📂 ParallelProcessing/
│   ├── 📂 cpus_with_DDP/
│   │   ├── ddp_train.py                  # DDP CPU training script
│   │   ├── main.py                       # Main execution script
│   │   ├── model.py                      # Model architecture
│   │   ├── ParallelExecutionCPU.ipynb
│   │   └── logs/models/metrics/plots/
│   │
│   ├── 📂 gpus_with_DDP/
│   │   ├── ddp_train.py                  # DDP GPU training script
│   │   ├── main.py
│   │   ├── model.py
│   │   └── logs/models/metrics/plots/
│   │
│   ├── 📂 gpus_with_DDP_AMP/
│   │   ├── ddp_train.py                  # DDP + AMP training script
│   │   ├── main.py
│   │   ├── model.py
│   │   └── logs/models/metrics/plots/
│   │
│   └── 📂 gpus_with_DDP_AMP_ModelParallel/
│       ├── ddp_train.py                  # Full parallelism pipeline
│       ├── main.py
│       ├── model.py
│       └── logs/models/metrics/plots/
│
└── 📂 preprocessed_glaucoma_data/        # Processed dataset (not in repo)
```

### ⚙️ Installation & Setup

Move into the code_files folder and then follow below:

```bash
# Clone the repository
git clone https://github.com/ramumarrp77/EfficientParallelComputingforDeepLearningBasedGlaucomaDetection.git
cd EfficientParallelComputingforDeepLearningBasedGlaucomaDetection

# Install dependencies
pip install torch torchvision numpy pandas opencv-python matplotlib seaborn scikit-learn
```

### 🏃 Running Experiments

#### CPU Training (DDP)
```bash
cd ParallelProcessing/cpus_with_DDP
python main.py --num_workers 8 --batch_size 32 --epochs 20
```

#### GPU Training (DDP)
```bash
cd ParallelProcessing/gpus_with_DDP
python main.py --num_gpus 2 --batch_size 32 --epochs 20
```

#### GPU Training (DDP + AMP)
```bash
cd ParallelProcessing/gpus_with_DDP_AMP
python main.py --num_gpus 2 --batch_size 32 --amp --epochs 20
```

---

## 💡 Key Findings

### ✅ Recommendations

1. **For CPU-Only Training:**
   - Use 8-16 workers with DDP
   - Batch size: 32
   - Expected speedup: 6-10x over serial

2. **For GPU Training:**
   - **Optimal Setup:** 2× V100 GPUs + DDP + AMP
   - Batch size: 32
   - Expected speedup: ~200x over serial CPU
   - Training time: ~3.8 minutes for 20 epochs

3. **Avoid:**
   - More than 16 CPU workers (diminishing returns)
   - 4+ GPUs for small datasets (<10GB)
   - Model Parallelism for models that fit in single GPU

### 🎓 Lessons Learned

- ✅ DDP scales efficiently with proper data loading optimization
- ✅ AMP provides significant speedup on Tensor Core GPUs
- ✅ Communication overhead becomes bottleneck beyond certain parallelism
- ✅ Batch size selection impacts both speed and accuracy
- ❌ More GPUs ≠ always better (efficiency drops with overhead)

---

## 📚 References

1. **Glaucoma Statistics:** [PubMed - Global Prevalence](https://pubmed.ncbi.nlm.nih.gov/24974815/)
2. **ResNet:** He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
3. **SE Networks:** Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
4. **ASPP:** Chen et al., "DeepLab: Semantic Image Segmentation", PAMI 2017
5. **Dataset:** [SMDG-19 Kaggle Dataset](https://www.kaggle.com/datasets/sabari50312/fundus-pytorch/)
---

<div align="center">

</div>
