# Setup Guide
## USC Course Recommendation Model - Installation & Configuration

Complete step-by-step setup instructions for macOS with Apple Silicon.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Setup](#quick-setup)
3. [Detailed Installation](#detailed-installation)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)
6. [Next Steps](#next-steps)

---

## System Requirements

### Hardware
- **Required**: Apple Silicon Mac (M1, M2, M3, M4)
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: 5GB free space (10GB recommended)
- **OS**: macOS 13.0+ (Ventura or later)

### Software
- **Python**: 3.9 - 3.14 (3.14 tested)
- **Git**: For cloning repository (optional)
- **Command Line Tools**: Xcode Command Line Tools

### Check Your System
```bash
# Check macOS version
sw_vers

# Check architecture (should show arm64)
uname -m

# Check Python version
python3 --version

# Check available space
df -h
```

---

## Quick Setup

### One-Command Setup (Recommended)

```bash
# Clone or navigate to project directory
cd /path/to/PDF-Finetuning-Model

# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import mlx; import mlx_lm; print('✅ Setup complete!')"
```

**Time**: ~5-10 minutes (depends on download speed)

---

## Detailed Installation

### Step 1: Install Xcode Command Line Tools

```bash
# Install if not already installed
xcode-select --install

# Verify installation
xcode-select -p
# Should output: /Library/Developer/CommandLineTools
```

### Step 2: Verify Python Installation

```bash
# Check Python version
python3 --version
# Should be Python 3.9 or higher

# If Python is not installed or outdated:
# Option 1: Install from python.org
# Download from: https://www.python.org/downloads/

# Option 2: Install via Homebrew
brew install python@3.11
```

### Step 3: Create Project Directory

```bash
# Option A: Clone from Git (if available)
git clone https://github.com/your-username/PDF-Finetuning-Model.git
cd PDF-Finetuning-Model

# Option B: Create new directory
mkdir -p ~/Projects/PDF-Finetuning-Model
cd ~/Projects/PDF-Finetuning-Model
```

### Step 4: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Verify creation
ls -la venv/
# Should see: bin, include, lib directories
```

**Why virtual environment?**
- Isolates project dependencies
- Prevents conflicts with system Python
- Easy to recreate if needed

### Step 5: Activate Virtual Environment

```bash
# Activate (you'll see (venv) in prompt)
source venv/bin/activate

# Verify activation
which python
# Should output: /path/to/PDF-Finetuning-Model/venv/bin/python
```

**Note**: You must activate the virtual environment every time you open a new terminal:
```bash
cd /path/to/PDF-Finetuning-Model
source venv/bin/activate
```

### Step 6: Upgrade pip

```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Verify pip version
pip --version
# Should be pip 25.0 or higher
```

### Step 7: Install Dependencies

#### Option A: From requirements.txt (Recommended)

```bash
# Install all dependencies at once
pip install -r requirements.txt

# This installs:
# - mlx (0.30.3)
# - mlx-lm (0.30.2)
# - transformers (5.0.0rc1)
# - datasets (4.5.0)
# - pandas, numpy, tqdm
# - And all dependencies
```

#### Option B: Manual Installation

```bash
# Install core packages
pip install mlx==0.30.3
pip install mlx-lm==0.30.2
pip install transformers==5.0.0rc1
pip install datasets==4.5.0
pip install pandas numpy tqdm
pip install huggingface_hub
```

**Time**: ~5-10 minutes (downloads ~500MB)

### Step 8: Verify Installation

```bash
# Test Python imports
python3 << EOF
import mlx
import mlx_lm
from transformers import AutoTokenizer
from datasets import load_dataset
print("✅ All packages imported successfully!")
print(f"MLX version: {mlx.__version__}")
EOF
```

Expected output:
```
✅ All packages imported successfully!
MLX version: 0.30.3
```

---

## Verification

### Test 1: Check MLX

```bash
python3 << EOF
import mlx.core as mx
import mlx.nn as nn

# Test Metal acceleration
print(f"✅ MLX installed: {mx.__version__}")
print(f"✅ Metal available: {mx.metal.is_available()}")

# Simple computation test
a = mx.array([1, 2, 3])
b = mx.array([4, 5, 6])
c = a + b
print(f"✅ MLX computation works: {c}")
EOF
```

### Test 2: Check MLX-LM

```bash
python3 << EOF
from mlx_lm import load, generate

print("✅ MLX-LM installed successfully")
print("✅ Ready for fine-tuning!")
EOF
```

### Test 3: Check Dataset Access

```bash
python3 << EOF
from datasets import load_dataset

print("Testing dataset access...")
ds = load_dataset("USC/USC-Course-Catalog", split="train[:5]")
print(f"✅ Dataset loaded: {len(ds)} samples")
print(f"✅ Features: {list(ds.features.keys())}")
EOF
```

### Test 4: Run Exploration Script

```bash
# Test with exploration script
python explore_dataset.py

# Should display:
# - Dataset structure
# - Number of courses
# - Sample course data
```

### Complete Verification

```bash
# Run all verification checks
python3 << EOF
print("="*60)
print("INSTALLATION VERIFICATION")
print("="*60)

# 1. MLX
try:
    import mlx
    print("✅ MLX:", mlx.__version__)
except Exception as e:
    print("❌ MLX:", str(e))

# 2. MLX-LM
try:
    import mlx_lm
    print("✅ MLX-LM: Installed")
except Exception as e:
    print("❌ MLX-LM:", str(e))

# 3. Transformers
try:
    import transformers
    print("✅ Transformers:", transformers.__version__)
except Exception as e:
    print("❌ Transformers:", str(e))

# 4. Datasets
try:
    from datasets import __version__ as ds_version
    print("✅ Datasets:", ds_version)
except Exception as e:
    print("❌ Datasets:", str(e))

# 5. Pandas
try:
    import pandas as pd
    print("✅ Pandas:", pd.__version__)
except Exception as e:
    print("❌ Pandas:", str(e))

# 6. NumPy
try:
    import numpy as np
    print("✅ NumPy:", np.__version__)
except Exception as e:
    print("❌ NumPy:", str(e))

print("="*60)
print("✅ Setup Complete! Ready to use.")
print("="*60)
EOF
```

---

## Troubleshooting

### Issue 1: "externally-managed-environment" Error

**Error:**
```
error: externally-managed-environment
× This environment is externally managed
```

**Solution:**
Always use a virtual environment:
```bash
# Create venv if you haven't
python3 -m venv venv

# Activate it
source venv/bin/activate

# Then install
pip install -r requirements.txt
```

---

### Issue 2: MLX Installation Fails

**Error:**
```
ERROR: Could not find a version that satisfies the requirement mlx
```

**Causes & Solutions:**

1. **Not on Apple Silicon:**
   ```bash
   # Check architecture
   uname -m
   # Must show: arm64
   # If shows: x86_64, MLX won't work
   ```

2. **Wrong Python version:**
   ```bash
   # MLX requires Python 3.9+
   python3 --version
   ```

3. **Not in virtual environment:**
   ```bash
   # Always activate venv first
   source venv/bin/activate
   ```

---

### Issue 3: Import Errors

**Error:**
```python
ModuleNotFoundError: No module named 'mlx'
```

**Solution:**
```bash
# 1. Ensure virtual environment is activated
source venv/bin/activate
# You should see (venv) in your prompt

# 2. Verify mlx is installed
pip list | grep mlx

# 3. If not installed, reinstall
pip install mlx mlx-lm
```

---

### Issue 4: Slow Downloads

**Issue**: Package downloads taking too long

**Solution:**
```bash
# Use a different mirror
pip install -r requirements.txt -i https://pypi.org/simple

# Or upgrade pip first
pip install --upgrade pip

# Or install in parts
pip install mlx mlx-lm  # Install these first
pip install -r requirements.txt  # Then the rest
```

---

### Issue 5: Permission Denied

**Error:**
```
PermissionError: [Errno 13] Permission denied
```

**Solution:**
```bash
# DO NOT use sudo with pip!
# Instead, use virtual environment

# If you accidentally used sudo, recreate venv:
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### Issue 6: Disk Space

**Error:**
```
OSError: [Errno 28] No space left on device
```

**Solution:**
```bash
# Check available space
df -h

# Need at least 5GB free
# Clean up space:
# - Empty Trash
# - Remove old downloads
# - Delete unused apps

# Or install to external drive:
cd /Volumes/ExternalDrive/
mkdir PDF-Finetuning-Model
cd PDF-Finetuning-Model
# Continue setup from here
```

---

### Issue 7: Metal Not Available

**Issue**: `mx.metal.is_available()` returns False

**Possible Causes:**
1. Not on Apple Silicon
2. macOS version too old
3. Metal drivers not loaded

**Solution:**
```bash
# 1. Verify Apple Silicon
uname -m  # Must show: arm64

# 2. Check macOS version
sw_vers  # Should be 13.0+

# 3. Restart Mac (for driver issues)
sudo reboot

# 4. Reinstall MLX
pip uninstall mlx mlx-metal
pip install mlx mlx-lm
```

---

## Environment Management

### Activating Environment

```bash
# Every new terminal session, run:
cd /path/to/PDF-Finetuning-Model
source venv/bin/activate

# Create alias for convenience (add to ~/.zshrc or ~/.bash_profile):
echo 'alias mlx-env="cd ~/Projects/PDF-Finetuning-Model && source venv/bin/activate"' >> ~/.zshrc

# Then just run:
mlx-env
```

### Deactivating Environment

```bash
# When done working
deactivate
```

### Recreating Environment

```bash
# If environment is corrupted
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Updating Dependencies

```bash
# Update all packages
pip install --upgrade -r requirements.txt

# Update specific package
pip install --upgrade mlx mlx-lm

# Generate new requirements.txt
pip freeze > requirements.txt
```

---

## Next Steps

### After Setup, You Can:

1. **Explore the Dataset**
   ```bash
   python explore_dataset.py
   ```

2. **Prepare Training Data**
   ```bash
   python prepare_training_data.py
   ```

3. **Run Fine-tuning**
   ```bash
   python finetune_model.py
   ```

4. **Test the Model**
   ```bash
   python test_with_adapters.py
   ```

5. **Interactive Mode**
   ```bash
   python inference.py
   ```

---

## Additional Resources

### Documentation
- [README.md](../README.md) - Main documentation
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Quick start guide
- [IMPROVEMENT_GUIDE.md](IMPROVEMENT_GUIDE.md) - Enhancement strategies

### External Links
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX-LM Examples](https://github.com/ml-explore/mlx-examples)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets)

---

## Setup Checklist

- [ ] Apple Silicon Mac (M1/M2/M3/M4)
- [ ] macOS 13.0+ installed
- [ ] Python 3.9+ installed
- [ ] Project directory created
- [ ] Virtual environment created
- [ ] Virtual environment activated
- [ ] Dependencies installed via requirements.txt
- [ ] MLX installation verified
- [ ] MLX-LM installation verified
- [ ] Dataset access tested
- [ ] Ready to start fine-tuning! 🚀

---

## Quick Commands Reference

```bash
# Setup (one-time)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Daily workflow
source venv/bin/activate          # Start session
python explore_dataset.py         # Explore data
python prepare_training_data.py   # Prepare data
python finetune_model.py          # Train model
python test_with_adapters.py      # Test model
deactivate                        # End session

# Troubleshooting
pip install --upgrade pip         # Upgrade pip
pip list                          # List packages
pip show mlx                      # Check package info
python -c "import mlx; print(mlx.__version__)"  # Test import
```

---

**Setup Time**: ~15-20 minutes
**Disk Space**: ~5GB
**Last Updated**: January 2026
**Status**: ✅ Tested on macOS Sonoma with M3
