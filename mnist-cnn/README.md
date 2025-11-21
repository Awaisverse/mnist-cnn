# MNIST CNN Classification Project 🎯

A beginner-friendly project for training a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset using PyTorch.

## 📚 Table of Contents

- [What is a CNN?](#what-is-a-cnn)
- [What is the MNIST Dataset?](#what-is-the-mnist-dataset)
- [Project Structure](#project-structure)
- [Workflow: Cursor → Kaggle → Cursor](#workflow-cursor--kaggle--cursor)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Results](#results)

---

## 🧠 What is a CNN?

**CNN** stands for **Convolutional Neural Network** - a type of deep learning model designed specifically for processing images and visual data.

### Key Concepts:

#### 1. **Convolutional Layers**
- Think of these as "feature detectors"
- They slide small filters (kernels) across the image to detect patterns like edges, curves, and shapes
- Each filter learns to recognize different visual features
- Example: One filter might detect horizontal lines, another detects curves

#### 2. **Pooling Layers**
- Reduces the size of the image while keeping important information
- Like zooming out from a detailed map - you lose some detail but keep the big picture
- Makes the model faster and more efficient
- Helps prevent overfitting (memorizing training data too much)

#### 3. **Fully Connected Layers**
- Takes all the detected features and combines them to make predictions
- Like a final decision-maker that says "Based on all these features, this is digit 7"

### Why CNNs for Images?

- **Spatial Understanding**: They understand that pixels close together are related
- **Translation Invariant**: Can recognize a "7" even if it's in different positions
- **Feature Hierarchy**: Detect simple patterns (edges) → complex patterns (shapes) → final prediction (digit)
- **Efficiency**: Much better than regular neural networks for image tasks

### Our CNN Architecture:

```
Input Image (28×28 grayscale)
    ↓
[Conv1: 32 filters] → [ReLU] → [MaxPool] → (14×14×32)
    ↓
[Conv2: 64 filters] → [ReLU] → [MaxPool] → (7×7×64)
    ↓
[Flatten] → (3136 features)
    ↓
[Fully Connected: 128] → [ReLU] → [Dropout]
    ↓
[Output: 10 classes] → (Predictions for digits 0-9)
```

---

## 📊 What is the MNIST Dataset?

**MNIST** (Modified National Institute of Standards and Technology) is one of the most famous datasets in machine learning, often called the "Hello World" of computer vision.

### Dataset Overview:

- **Purpose**: Handwritten digit recognition (0-9)
- **Training Images**: 60,000 grayscale images
- **Test Images**: 10,000 grayscale images
- **Image Size**: 28×28 pixels
- **Color**: Grayscale (1 channel, values 0-255)

### Why MNIST is Perfect for Learning:

✅ **Simple**: Small images (28×28), grayscale only  
✅ **Well-preprocessed**: Clean, centered digits  
✅ **Fast Training**: Can train in minutes, not hours  
✅ **Established Benchmark**: Easy to compare with others  
✅ **Real-World Relevance**: Foundation for more complex tasks  

### Sample Images:

```
MNIST contains images like these:
  0    1    2    3    4    5    6    7    8    9
  ┌─┐  ┌─┐  ┌─┐  ┌─┐  ┌─┐  ┌─┐  ┌─┐  ┌─┐  ┌─┐  ┌─┐
  │ │  │ │  ┌─┘  ┌─┐  │ │  ┌─┐  ┌─┐  ┌─┐  ┌─┐  ┌─┐
  │ │  │ │  └─┐  └─┘  │ │  └─┐  └─┘  └─┐  └─┐  └─┘
  └─┘  └─┘  └─┘  └─┘  └─┘  └─┘  └─┘  └─┘  └─┘  └─┘
```

Each digit is a 28×28 grid of pixels with values representing the darkness/intensity.

---

## 📁 Project Structure

```
mnist-cnn/
│
├── data/                      # Dataset storage directory
│   └── (MNIST data downloaded here)
│
├── models/                    # Saved trained models
│   └── mnist_cnn_model.pth   # Trained model weights
│
├── src/                       # Source code
│   ├── model.py              # CNN model architecture definition
│   └── train.py              # Training script (for local use)
│
├── mnist_training.ipynb      # Kaggle-ready Jupyter notebook
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🔄 Workflow: Cursor → Kaggle → Cursor

This project uses a hybrid workflow that combines local development (Cursor) with cloud training (Kaggle) for the best of both worlds.

### Step-by-Step Workflow:

```
┌─────────────────────────────────────────────────────────────┐
│                    STEP 1: CURSOR (Local)                   │
│  Create project structure, model architecture, and code     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    STEP 2: KAGGLE (Cloud)                   │
│  Upload notebook → Train model → Download trained model     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    STEP 3: CURSOR (Local)                   │
│  Use trained model for inference, evaluation, deployment    │
└─────────────────────────────────────────────────────────────┘
```

### Detailed Workflow:

#### 🖥️ **PHASE 1: Cursor (Local Development)**

**What you do here:**
1. **Setup Project Structure**
   - Create directories (`data/`, `models/`, `src/`)
   - Define model architecture in `src/model.py`
   - Write training code in `mnist_training.ipynb`

2. **Design & Plan**
   - Define your CNN architecture
   - Set hyperparameters (learning rate, batch size, epochs)
   - Write well-commented, beginner-friendly code

3. **Version Control**
   - Use Git to track changes
   - Organize code for easy sharing

**Why Cursor first?**
- ✨ Better code editing experience
- ✨ AI assistance for coding
- ✨ Easy project organization
- ✨ Version control integration

---

#### ☁️ **PHASE 2: Kaggle (Cloud Training)**

**What you do here:**
1. **Upload to Kaggle**
   - Upload `mnist_training.ipynb` to Kaggle
   - Kaggle automatically provides GPU access

2. **Run Training**
   - Execute the notebook cells sequentially
   - Kaggle downloads MNIST automatically
   - Model trains on powerful GPU (much faster than CPU!)

3. **Monitor Progress**
   - Watch training loss decrease
   - See validation accuracy improve
   - Visualize results with plots

4. **Save Model**
   - Model is saved to `models/mnist_cnn_model.pth`
   - Download the `.pth` file to your local machine

**Why Kaggle for training?**
- 🚀 **Free GPU access** - Train in minutes instead of hours
- 💰 **No setup costs** - No need to buy expensive GPUs
- 📊 **Built-in datasets** - MNIST already available
- 📈 **Easy visualization** - Plots display automatically
- 💾 **Cloud storage** - Models saved in Kaggle

**Steps on Kaggle:**
1. Go to [kaggle.com](https://www.kaggle.com)
2. Create new notebook
3. Upload `mnist_training.ipynb` or copy/paste code
4. Enable GPU (Settings → Accelerator → GPU T4 x2)
5. Run all cells
6. Download `models/mnist_cnn_model.pth` when training completes

---

#### 🖥️ **PHASE 3: Cursor (Local Usage)**

**What you do here:**
1. **Load Trained Model**
   - Place downloaded `mnist_cnn_model.pth` in `models/` folder
   - Load model using PyTorch

2. **Use for Inference**
   - Make predictions on new images
   - Test model performance
   - Build applications

3. **Further Development**
   - Experiment with different architectures
   - Fine-tune hyperparameters
   - Integrate into larger projects

**Why return to Cursor?**
- 🏠 **Local control** - Use model without internet
- 🔧 **Custom development** - Build applications
- 📦 **Deployment** - Package for production
- 🔄 **Iteration** - Improve and retrain

---

### Workflow Benefits:

| Phase | Benefit |
|-------|---------|
| **Cursor (Local)** | Better code editing, organization, version control |
| **Kaggle (Cloud)** | Free GPU, fast training, no hardware costs |
| **Cursor (Local)** | Local inference, deployment, further development |

**Best of both worlds:**
- ✅ Develop locally with great tools
- ✅ Train in the cloud with free GPU
- ✅ Use trained models locally

---

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- Git (optional, for version control)

### Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd mnist-cnn
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   This installs:
   - `torch` - PyTorch deep learning framework
   - `torchvision` - Datasets and image utilities
   - `numpy` - Numerical computing
   - `matplotlib` - Plotting and visualization

3. **Verify installation**
   ```python
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   ```

---

## 📖 Usage

### Option 1: Kaggle Notebook (Recommended)

1. **Upload `mnist_training.ipynb` to Kaggle**
   - Create new notebook at kaggle.com
   - Copy/paste notebook contents
   - Enable GPU in settings

2. **Run all cells**
   - Kaggle will download MNIST automatically
   - Model trains on GPU
   - Results and plots display automatically

3. **Download trained model**
   - After training, download `models/mnist_cnn_model.pth`
   - Save to your local `models/` directory

### Option 2: Local Training (Slower)

1. **Run the training script**
   ```bash
   python src/train.py
   ```

   Note: Training on CPU will be significantly slower (hours vs minutes)

2. **Or use Jupyter**
   ```bash
   jupyter notebook mnist_training.ipynb
   ```

### Loading and Using Trained Model

```python
import torch
from src.model import MNISTCNN

# Load model
model = MNISTCNN()
model.load_state_dict(torch.load('models/mnist_cnn_model.pth'))
model.eval()

# Make prediction
# (Your image preprocessing code here)
with torch.no_grad():
    output = model(image)
    predicted = torch.max(output, 1)[1]
    print(f"Predicted digit: {predicted.item()}")
```

---

## 📈 Results

After training for 10 epochs, you should expect:

- **Training Accuracy**: ~99%+
- **Validation Accuracy**: ~98-99%
- **Training Time**: 
  - GPU (Kaggle): ~2-5 minutes
  - CPU (Local): ~30-60 minutes

### Model Performance:

The CNN typically achieves:
- ✅ **High accuracy** on MNIST test set
- ✅ **Fast inference** (< 1ms per image)
- ✅ **Robust predictions** even with slight variations

### Training Metrics:

The notebook automatically tracks and plots:
- Training/Validation Loss (should decrease)
- Training/Validation Accuracy (should increase)
- Visual plots to monitor progress

---

## 🎓 Learning Resources

### Understanding CNNs:
- [CS231n Course Notes](http://cs231n.github.io/convolutional-networks/)
- [PyTorch Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

### MNIST Dataset:
- [Official MNIST Website](http://yann.lecun.com/exdb/mnist/)
- [MNIST on Wikipedia](https://en.wikipedia.org/wiki/MNIST_database)

### PyTorch:
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

---

## 🤝 Contributing

Feel free to:
- 🐛 Report bugs
- 💡 Suggest improvements
- 📝 Improve documentation
- 🔧 Add new features

---

## 📝 License

This project is open source and available for educational purposes.

---

## 🙏 Acknowledgments

- MNIST dataset creators (Yann LeCun et al.)
- PyTorch team for the excellent framework
- Kaggle for providing free GPU resources
- The open-source community

---

## 📧 Contact

For questions or feedback, please open an issue on the repository.

---

**Happy Learning! 🎉**

*Remember: This is a beginner-friendly project. Don't hesitate to experiment and modify the code to learn more!*

