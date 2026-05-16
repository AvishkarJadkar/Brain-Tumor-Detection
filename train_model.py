"""
Brain Tumor Detection using Transfer Learning (MobileNetV2)
===========================================================
- Uses MobileNetV2 pretrained on ImageNet as the feature extractor
- Fine-tunes with data augmentation to handle the small dataset (~253 images)
- Saves the best model, training curves, and a classification report
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ──────────────────────────── CONFIG ────────────────────────────
DATASET_DIR  = r"d:\brain_tumor_dataset"
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 16
EPOCHS       = 30
LEARNING_RATE = 1e-4
SEED         = 42
MODEL_SAVE_PATH = os.path.join(DATASET_DIR, "brain_tumor_model.keras")
RESULTS_DIR  = os.path.join(DATASET_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ──────────────────────────── DATA LOADING ────────────────────────────
print("=" * 60)
print("  BRAIN TUMOR DETECTION - MODEL TRAINING")
print("=" * 60)

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,       # 80/20 train-val split
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# No augmentation for validation — only rescale
val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

print("\n📂 Loading training data...")
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    seed=SEED,
    classes=['no', 'yes'],       # 0 = no tumor, 1 = tumor
    shuffle=True
)

print("\n📂 Loading validation data...")
val_generator = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    seed=SEED,
    classes=['no', 'yes'],
    shuffle=False
)

print(f"\n✅ Training samples  : {train_generator.samples}")
print(f"✅ Validation samples: {val_generator.samples}")
print(f"📌 Class mapping     : {train_generator.class_indices}")

# ──────────────────────────── MODEL ARCHITECTURE ────────────────────────────
print("\n🧠 Building model (MobileNetV2 + Custom Head)...")

# Load pretrained MobileNetV2 (without top classification layers)
base_model = keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model initially
base_model.trainable = False

# Build classification head
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')   # Binary output
], name="BrainTumorDetector")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

model.summary()

# ──────────────────────────── CALLBACKS ────────────────────────────
cb_list = [
    callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# ──────────────────────────── PHASE 1: TRAIN HEAD ONLY ────────────────────────────
print("\n" + "=" * 60)
print("  PHASE 1: Training classification head (base frozen)")
print("=" * 60)

history1 = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    callbacks=cb_list,
    verbose=1
)

# ──────────────────────────── PHASE 2: FINE-TUNE TOP LAYERS ────────────────────────────
print("\n" + "=" * 60)
print("  PHASE 2: Fine-tuning top layers of MobileNetV2")
print("=" * 60)

# Unfreeze the last 30 layers of MobileNetV2
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with a lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

history2 = model.fit(
    train_generator,
    epochs=EPOCHS,
    initial_epoch=len(history1.history['loss']),
    validation_data=val_generator,
    callbacks=cb_list,
    verbose=1
)

# ──────────────────────────── MERGE HISTORIES ────────────────────────────
history = {}
for key in history1.history:
    history[key] = history1.history[key] + history2.history[key]

# ──────────────────────────── EVALUATION ────────────────────────────
print("\n" + "=" * 60)
print("  EVALUATION")
print("=" * 60)

# Reload best model
model = keras.models.load_model(MODEL_SAVE_PATH)

val_generator.reset()
val_loss, val_acc, val_auc = model.evaluate(val_generator, verbose=0)
print(f"\n📊 Validation Loss     : {val_loss:.4f}")
print(f"📊 Validation Accuracy : {val_acc:.4f}")
print(f"📊 Validation AUC      : {val_auc:.4f}")

# Generate predictions for classification report
val_generator.reset()
y_pred_probs = model.predict(val_generator, verbose=0)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()
y_true = val_generator.classes

print("\n📋 Classification Report:")
print("-" * 50)
report = classification_report(y_true, y_pred, target_names=['No Tumor', 'Tumor'])
print(report)

# Save classification report
with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
    f.write("Brain Tumor Detection - Classification Report\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Validation Loss     : {val_loss:.4f}\n")
    f.write(f"Validation Accuracy : {val_acc:.4f}\n")
    f.write(f"Validation AUC      : {val_auc:.4f}\n\n")
    f.write(report)

# ──────────────────────────── PLOTS ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Accuracy
axes[0].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Loss
axes[1].plot(history['loss'], label='Train Loss', linewidth=2)
axes[1].plot(history['val_loss'], label='Val Loss', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. AUC
axes[2].plot(history['auc'], label='Train AUC', linewidth=2)
axes[2].plot(history['val_auc'], label='Val AUC', linewidth=2)
axes[2].set_title('Model AUC', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('AUC')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=150, bbox_inches='tight')
print(f"\n📈 Training curves saved to: {RESULTS_DIR}/training_curves.png")

# Confusion Matrix
fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['No Tumor', 'Tumor'])
disp.plot(ax=ax_cm, cmap='Blues', values_format='d')
ax_cm.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150, bbox_inches='tight')
print(f"📊 Confusion matrix saved to: {RESULTS_DIR}/confusion_matrix.png")

print(f"\n✅ Model saved to: {MODEL_SAVE_PATH}")
print(f"✅ All results saved to: {RESULTS_DIR}/")
print("\n" + "=" * 60)
print("  TRAINING COMPLETE!")
print("=" * 60)
