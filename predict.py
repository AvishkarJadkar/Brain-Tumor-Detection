"""
Brain Tumor Prediction Script
==============================
Load the trained model and predict on individual MRI images.

Usage:
    python predict.py <image_path>
    python predict.py d:\brain_tumor_dataset\yes\Y1.jpg
"""

import sys
import os
import numpy as np
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import tensorflow as tf
from tensorflow import keras

MODEL_PATH = os.path.join(os.path.dirname(__file__), "brain_tumor_model.keras")
IMG_SIZE = (224, 224)


def load_and_preprocess(image_path: str) -> np.ndarray:
    """Load an image, resize to 224x224, and normalize."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)  # Add batch dimension


def predict(image_path: str):
    """Run prediction on a single image."""
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found: {image_path}")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print("   Please run train_model.py first to train the model.")
        sys.exit(1)

    print(f"🔄 Loading model from: {MODEL_PATH}")
    model = keras.models.load_model(MODEL_PATH)

    print(f"🖼️  Processing image: {image_path}")
    img = load_and_preprocess(image_path)

    prediction = model.predict(img, verbose=0)[0][0]
    confidence = prediction if prediction > 0.5 else 1 - prediction

    print("\n" + "=" * 50)
    if prediction > 0.5:
        print(f"  🔴 TUMOR DETECTED")
    else:
        print(f"  🟢 NO TUMOR DETECTED")
    print(f"  📊 Confidence: {confidence * 100:.1f}%")
    print(f"  📊 Raw score : {prediction:.4f}")
    print("=" * 50)

    return prediction


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py yes/Y1.jpg")
        sys.exit(1)

    predict(sys.argv[1])
