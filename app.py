# app.py
# Flask Web Server for Deepfake Detection Project
# For: Ephraim, FUTA
# Purpose: Receives images from the web browser, runs them through the
#          model, generates Grad-CAM and Uncertainty, and returns results.

import gradcam
import os
import json
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from torchvision import models, transforms
from PIL import Image

# Import the modules we wrote earlier
from gradcam import GradCAM, overlay_heatmap
from uncertainty import mc_dropout_predict, get_uncertainty_tier

# ── 1. FLASK APP SETUP ────────────────────────────────────────────
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH    = 'model/efficientnet_deepfake.pth'
DEVICE        = torch.device('cpu')  # We run the web app on CPU
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── 2. LOAD MODEL ONCE AT STARTUP ────────────────────────────────
def load_model():
    """
    Builds the EfficientNet-B0 architecture and loads the trained weights.
    This runs exactly once when the Flask server starts.
    """
    print("Loading trained model...")
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    # Replace the classifier head to match our 2 classes (REAL, FAKE)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 2)
    )
    # Load the weights from Colab
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()  # Set to evaluation mode
    
    # Initialize Grad-CAM with the loaded model
    global gradcam
    gradcam = GradCAM(model)
    print("Model loaded successfully!")
    return model

# Load the model before any routes are ready
model = load_model()

# ── 3. LOAD GENERALISATION RESULTS ────────────────────────────────
# Load the JSON file we created so it can be sent to the frontend
with open('generalisation_results.json', 'r') as f:
    gen_results = json.load(f)

# ── 4. PREPROCESSING PIPELINE ─────────────────────────────────────
# This must EXACTLY match the test_transforms in train.py
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

CLASS_NAMES = ['REAL', 'FAKE']

# ── 5. ROUTES ─────────────────────────────────────────────────────
@app.route('/')
def index():
    """
    Serves the main HTML page when the user visits the root URL.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives the uploaded image, runs the full pipeline, and returns
    the results as a JSON response to the browser.
    """
    # Check if an image was actually uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file  = request.files['image']
    fname = file.filename
    path  = os.path.join(UPLOAD_FOLDER, fname)
    file.save(path)

    # Open image with PIL and apply our preprocessing
    pil_img = Image.open(path).convert('RGB')
    tensor  = preprocess(pil_img).unsqueeze(0).to(DEVICE) # unsqueeze adds batch dimension

    # ── A. STANDARD PREDICTION ───────────────────────────────────
    with torch.no_grad():
        output     = model(tensor)
        probs      = torch.softmax(output, dim=1)[0]
        class_idx  = probs.argmax().item() # 0 or 1
        confidence = probs[class_idx].item()

    predicted_class = CLASS_NAMES[class_idx]

    # ── B. GRAD-CAM HEATMAP ──────────────────────────────────────
    # Generate the heatmap for the predicted class
    heatmap     = gradcam.generate(tensor, class_idx)
    # Overlay it on the original image
    overlay_img = overlay_heatmap(pil_img, heatmap)
    # Save the overlay image so the browser can display it
    heatmap_path = os.path.join(UPLOAD_FOLDER, 'heatmap_' + fname)
    overlay_img.save(heatmap_path)

    # ── C. MONTE CARLO DROPOUT UNCERTAINTY ───────────────────────
    mean_conf, uncertainty, _ = mc_dropout_predict(model, tensor, class_idx=class_idx)
    tier, colour, unc_msg     = get_uncertainty_tier(uncertainty)

    # ── D. USER AWARENESS PANEL (Contribution #4) ────────────────
    # This logic creates plain-English text for the user based on results.
    if predicted_class == 'FAKE':
        if confidence > 0.85:
            explanation = (
                "This image shows strong signs of AI generation. "
                "The highlighted regions in the heatmap indicate where "
                "the model detected synthetic patterns most strongly."
            )
        else:
            explanation = (
                "This image shows some signs of AI generation. "
                "The result is not fully conclusive — consider "
                "cross-checking with a reverse image search."
            )
        visual_cues = [
            "Check for unnatural blurring around facial edges",
            "Look for inconsistent lighting across the image",
            "Inspect eye reflections — they are often asymmetric in fakes",
            "Check ear geometry — AI often produces irregular ears",
            "Look for unusual skin texture patterns"
        ]
        actions = [
            "Perform a reverse image search (Google Images or TinEye)",
            "Report the image to the platform if you believe it is harmful",
            "Do not share this image until you have verified its source",
            "Check trusted fact-checking websites for related content"
        ]
    else:
        explanation = (
            "This image does not show strong signs of AI generation. "
            "The model classified it as likely real. "
            "However, no automated system is perfect — use your judgement."
        )
        visual_cues = [
            "Natural lighting and consistent shadows are good signs",
            "Consistent facial geometry suggests authenticity",
            "Look for natural background details"
        ]
        actions = [
            "You can share this image, but always verify the source",
            "Check the original source of the image for context"
        ]

    # ── E. SEND JSON RESPONSE BACK TO BROWSER ────────────────────
    return jsonify({
        'predicted_class':  predicted_class,
        'confidence':       round(mean_conf * 100, 2),
        'raw_confidence':   round(confidence * 100, 2),
        'uncertainty':      round(uncertainty, 4),
        'uncertainty_tier': tier,
        'uncertainty_colour': colour,
        'uncertainty_message': unc_msg,
        'heatmap_path':     '/' + heatmap_path,
        'original_path':    '/' + path,
        'explanation':      explanation,
        'visual_cues':      visual_cues,
        'actions':          actions,
        'generalisation':   gen_results
    })

# ── 6. START SERVER ───────────────────────────────────────────────
if __name__ == '__main__':
    # debug=True allows the server to auto-restart if you change the code
    app.run(debug=True, port=5000)