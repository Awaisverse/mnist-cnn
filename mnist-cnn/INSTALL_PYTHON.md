# 🐍 Python Installation Guide

## ✅ Recommended Python Version

For this MNIST CNN project, install:

### **Python 3.11** (Recommended) 🎯
- ✅ **Fully compatible** with PyTorch
- ✅ **Modern features** and performance improvements
- ✅ **Well-tested** with all ML libraries
- ✅ **Good balance** of stability and features

### **Python 3.10** (Alternative) 
- ✅ Also fully compatible
- ✅ Slightly older but very stable
- ✅ Good choice if you encounter issues with 3.11

### **Avoid:**
- ❌ Python 3.12+ (may have compatibility issues with some packages)
- ❌ Python 3.7 or older (outdated, security issues)
- ❌ Python 3.14 (too new, PyTorch may not support it yet)

---

## 📥 How to Download Python

### Option 1: Official Python Website (Recommended)

1. **Visit**: https://www.python.org/downloads/
2. **Download**: 
   - Click on **"Download Python 3.11.x"** (or latest 3.11 version)
   - For Windows: Download **Windows installer (64-bit)**
3. **Install**:
   - ⚠️ **IMPORTANT**: Check **"Add Python to PATH"** during installation
   - Click **"Install Now"**
   - Wait for installation to complete

### Option 2: Microsoft Store (Windows)

1. Open **Microsoft Store**
2. Search for **"Python 3.11"**
3. Click **"Install"**

---

## ✅ Verify Installation

Open PowerShell or Command Prompt and run:

```bash
python --version
```

You should see:
```
Python 3.11.x
```

Also check pip:
```bash
pip --version
```

---

## 📦 Install Project Dependencies

After installing Python, install the required packages:

```bash
cd "R:\Awais Akram\Career\mnist-cnn"
pip install -r requirements.txt
```

This will install:
- `torch` - PyTorch (deep learning framework)
- `torchvision` - Vision utilities and datasets
- `numpy` - Numerical computing
- `matplotlib` - Plotting and visualization

---

## 🚀 Optional: GPU Support (CUDA)

If you have an **NVIDIA GPU** and want GPU acceleration:

### Check GPU:
```bash
nvidia-smi
```

If you have an NVIDIA GPU, install PyTorch with CUDA:

1. Visit: https://pytorch.org/get-started/locally/
2. Select your configuration:
   - OS: **Windows**
   - Package: **Pip**
   - CUDA: **11.8** or **12.1** (based on your GPU)
3. Run the provided command, for example:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

**Note**: If you don't have a GPU or aren't sure, the CPU version works fine (just slower for training).

---

## 🧪 Test Your Installation

After installing everything, test it:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

Run this in Python or create a test file:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 📝 Quick Installation Checklist

- [ ] Download Python 3.11 from python.org
- [ ] ✅ Check "Add Python to PATH" during installation
- [ ] Verify installation: `python --version`
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Test PyTorch: `python -c "import torch; print(torch.__version__)"`
- [ ] (Optional) Test GPU: `python -c "import torch; print(torch.cuda.is_available())"`

---

## 🎯 After Installation

Once Python is installed, you can:

1. **Run the test notebook**: Open `test_model.ipynb` in VS Code/Jupyter
2. **Run inference script**: `python src/inference.py`
3. **Train locally** (slower than Kaggle): `python src/train.py` (if you create it)

---

## ❓ Troubleshooting

### "Python not found"
- **Solution**: Reinstall Python and check "Add Python to PATH"
- Or add Python manually to PATH in System Environment Variables

### "pip not found"
- **Solution**: Python 3.11 includes pip automatically. Try `python -m pip` instead

### "torch installation fails"
- **Solution**: Update pip first: `python -m pip install --upgrade pip`
- Then: `pip install torch torchvision numpy matplotlib`

### GPU not detected
- **Solution**: Install CUDA-enabled PyTorch (see GPU Support section above)
- Or use CPU version (works fine, just slower)

---

**Happy Coding! 🚀**

