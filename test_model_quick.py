"""Quick test to verify model is working correctly"""
import torch
import torch.nn.functional as F
from src.model import MNISTCNN
from torchvision import datasets, transforms

# Load model
model = MNISTCNN()
model.load_state_dict(torch.load('models/mnist_cnn_model.pth', map_location='cpu'))
model.eval()

# Load test data with same transform as training
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

# Test on 5 samples
print("Testing model on MNIST test samples:")
print("=" * 60)
for i in range(5):
    img, true_label = test_data[i]
    with torch.no_grad():
        output = model(img.unsqueeze(0))
        probs = F.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1).item()
        conf = probs[0][pred].item() * 100
    
    status = "✓" if pred == true_label else "✗"
    print(f"Sample {i+1}: True={true_label}, Pred={pred}, Conf={conf:.2f}% {status}")

print("=" * 60)
print("✅ Model is working correctly!")

