"""
Flask Web Application for MNIST Digit Classification

This web app allows users to upload images and get predictions from the trained CNN model.
Uses the exact same preprocessing as training for accurate predictions.
"""

import os
import sys
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from model import MNISTCNN

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global model variable
model = None
device = torch.device('cpu')  # CPU is fine for inference!


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def preprocess_image(image_path):
    """
    Preprocess uploaded image to EXACTLY match MNIST training format.
    
    Training uses:
    - transforms.ToTensor() - converts to [0, 1] range
    - transforms.Normalize((0.1307,), (0.3081,)) - normalizes with mean/std
    
    This function replicates that exactly for best accuracy.
    """
    try:
        # Open and convert to grayscale
        img = Image.open(image_path).convert('L')
        
        # Get original size
        original_size = img.size
        
        # Auto-crop to digit (remove excess white space)
        img_array = np.array(img, dtype=np.float32)
        
        # Find bounding box of digit (non-white areas)
        # Threshold: pixels darker than 240 are considered part of digit
        threshold = 240
        rows = np.any(img_array < threshold, axis=1)
        cols = np.any(img_array < threshold, axis=0)
        
        if rows.any() and cols.any():
            try:
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                
                # Add small padding (5% of size)
                h, w = img_array.shape
                pad_h = max(1, int((rmax - rmin) * 0.05))
                pad_w = max(1, int((cmax - cmin) * 0.05))
                
                rmin = max(0, rmin - pad_h)
                rmax = min(h, rmax + pad_h)
                cmin = max(0, cmin - pad_w)
                cmax = min(w, cmax + pad_w)
                
                # Crop to digit
                img_array = img_array[rmin:rmax, cmin:cmax]
                img = Image.fromarray(img_array.astype(np.uint8))
            except (IndexError, ValueError):
                # If cropping fails, use original
                pass
        
        # Resize to 28x28 (MNIST size) with high-quality resampling
        img = img.resize((28, 28), Image.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Check if we need to invert (MNIST has white digits on black background)
        # If image is mostly bright (white background), invert it
        mean_val = img_array.mean()
        if mean_val > 128:
            img_array = 255.0 - img_array
        
        # Apply ToTensor equivalent: scale to [0, 1]
        img_array = img_array / 255.0
        
        # Apply MNIST normalization: (x - 0.1307) / 0.3081
        # This EXACTLY matches the training normalization
        img_array = (img_array - 0.1307) / 0.3081
        
        # Convert to tensor: (28, 28) -> (1, 1, 28, 28)
        # Add channel dimension: (28, 28) -> (1, 28, 28)
        # Add batch dimension: (1, 28, 28) -> (1, 1, 28, 28)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
        
        return img_tensor
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")


def load_model():
    """Load the trained model from Kaggle."""
    global model, device
    
    model_path = 'models/mnist_cnn_model.pth'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}\n"
            "Please download the model from Kaggle and place it in the models/ directory."
        )
    
    # Load model on CPU
    model = MNISTCNN()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()  # Set to evaluation mode
    
    # Verify model is loaded correctly
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded successfully on {device}")
    print(f"   Total parameters: {total_params:,}")
    
    return model


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return prediction using the trained model."""
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image to match training format
        try:
            img_tensor = preprocess_image(filepath)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        
        # Make prediction using the trained model
        # THIS IS USING YOUR ACTUAL TRAINED MODEL FROM KAGGLE
        with torch.no_grad():
            # img_tensor shape: (1, 1, 28, 28)
            output = model(img_tensor)  # <-- YOUR MODEL PREDICTION HERE
            
            # Get probabilities using softmax
            probabilities = F.softmax(output, dim=1)
            
            # Get predicted class
            predicted = torch.argmax(output, dim=1).item()
            
            # Get confidence (probability of predicted class)
            confidence = probabilities[0][predicted].item() * 100
        
        # Get probabilities for all classes
        all_probs = probabilities[0].cpu().numpy() * 100
        
        # Debug output (visible in terminal) - PROVES IT'S USING THE MODEL
        top3_indices = torch.topk(probabilities[0], 3).indices.tolist()
        top3_probs = [probabilities[0][i].item() * 100 for i in top3_indices]
        print(f"🔍 MODEL PREDICTION: {predicted} | Confidence: {confidence:.2f}% | Top 3: {[(top3_indices[i], f'{top3_probs[i]:.1f}%') for i in range(3)]}")
        print(f"   Raw model output (first 3 values): {output[0][:3].tolist()}")
        print(f"   Model file used: models/mnist_cnn_model.pth")
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        # Return prediction
        return jsonify({
            'success': True,
            'prediction': int(predicted),
            'confidence': round(confidence, 2),
            'probabilities': {str(i): round(float(all_probs[i]), 2) for i in range(10)}
        })
    
    except Exception as e:
        import traceback
        error_msg = f'Prediction error: {str(e)}'
        print(f"❌ Error: {error_msg}")
        print(traceback.format_exc())
        return jsonify({'error': error_msg}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting MNIST CNN Web Application")
    print("=" * 60)
    print(f"📱 Device: {device} (CPU is perfect for inference!)")
    print("=" * 60)
    
    # Load model on startup
    try:
        load_model()
        print("\n✅ Model loaded successfully!")
        print("🌐 Starting web server...")
        print("📝 Open http://127.0.0.1:5000 in your browser")
        print("=" * 60)
        print("\n💡 Tips for best results:")
        print("   - Upload clear, handwritten digits")
        print("   - White or light background with dark digits")
        print("   - Centered digit in the image")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        sys.exit(1)
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)

