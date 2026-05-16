"""
Brain Tumor Detection - Flask API
Run: python app.py
Then open http://localhost:5000 in your browser.
"""

import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "brain_tumor_model.keras")
IMG_SIZE = (224, 224)

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
print("[*] Loading model...")
model = keras.models.load_model(MODEL_PATH)
print("[OK] Model loaded!")


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    image_bytes = file.read()

    try:
        img = preprocess_image(image_bytes)
        prediction = float(model.predict(img, verbose=0)[0][0])
        confidence = prediction if prediction > 0.5 else 1 - prediction
        has_tumor = prediction > 0.5

        return jsonify({
            'has_tumor': has_tumor,
            'confidence': round(confidence * 100, 1),
            'raw_score': round(prediction, 4),
            'label': 'TUMOR DETECTED' if has_tumor else 'NO TUMOR DETECTED'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("[*] Starting server at http://localhost:5000")
    app.run(debug=False, port=5000)
