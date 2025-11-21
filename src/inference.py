import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import MNISTCNN
import matplotlib.pyplot as plt
import numpy as np

def get_device():
    """Detect and return available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ GPU FOUND! Using: {device}")
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
        return device, True
    else:
        device = torch.device('cpu')
        print(f"⚠️  No GPU detected. Using: CPU")
        return device, False

def load_model(model_path='models/mnist_cnn_model.pth', device=None, use_gpu=False):
    """Load the trained model."""
    model = MNISTCNN()
    if use_gpu and device:
        model.load_state_dict(torch.load(model_path, map_location='cuda'))
        model = model.to(device)
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict(model, image_tensor, device):
    """Make prediction on a single image."""
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
        probabilities = F.softmax(output, dim=1)
        predicted = torch.argmax(output, dim=1)
        confidence = probabilities[0][predicted].item() * 100
    return predicted.item(), confidence

def test_on_samples(model, test_dataset, num_samples=10, device=None, use_gpu=False):
    """Test model on random samples."""
    model.eval()
    correct = 0
    
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    print(f"\nTesting on {num_samples} random samples:")
    if use_gpu:
        print(f"🚀 Using GPU: {device}")
    else:
        print(f"⚠️  Using CPU")
    print("=" * 60)
    
    for i, idx in enumerate(indices):
        image, true_label = test_dataset[idx]
        predicted, confidence = predict(model, image, device)
        is_correct = predicted == true_label
        if is_correct:
            correct += 1
        
        print(f"Sample {i+1}: True={true_label}, Predicted={predicted}, "
              f"Confidence={confidence:.2f}%, {'✓' if is_correct else '✗'}")
    
    accuracy = 100 * correct / num_samples
    print("=" * 60)
    print(f"Accuracy on {num_samples} samples: {accuracy:.2f}%")
    return accuracy

if __name__ == "__main__":
    print("=" * 60)
    print("🖥️  GPU DETECTION")
    print("=" * 60)
    
    device, use_gpu = get_device()
    
    model_path = 'models/mnist_cnn_model.pth'
    
    print("\n" + "=" * 60)
    print("📦 LOADING MODEL")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("\n📝 To get the model:")
        print("   1. Go to your Kaggle notebook")
        print("   2. Find the 'models/' folder in output")
        print("   3. Download 'mnist_cnn_model.pth'")
        print("   4. Place it in: models/mnist_cnn_model.pth")
        exit(1)
    
    model = load_model(model_path, device, use_gpu)
    if use_gpu:
        print(f"✅ Model loaded on GPU: {device}")
    else:
        print("✅ Model loaded on CPU")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("\n" + "=" * 60)
    print("📊 LOADING DATASET")
    print("=" * 60)
    
    test_dataset = datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=transform
    )
    
    print(f"✅ Test dataset loaded: {len(test_dataset)} samples")
    
    print("\n" + "=" * 60)
    print("🧪 TESTING MODEL")
    print("=" * 60)
    
    test_on_samples(model, test_dataset, num_samples=10, device=device, use_gpu=use_gpu)
    
    print("\n✅ Inference completed!")
    if use_gpu:
        print("🚀 GPU acceleration was used for fast inference!")

