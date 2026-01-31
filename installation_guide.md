# Installation

## 1. Install dependencies

### Connect Colab to your local computer
Install colab extension

### Install PyTorch
Choose the appropriate version based on your system:

```bash
# For CUDA 12.1 (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or for CUDA 11.8 (older GPUs / laptops)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -q datasets scikit-learn seaborn matplotlib pandas numpy

# Verify on notebook
import torch
print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA version:", torch.version.cuda)