"""
Train the CNN model for Brain Tumor classification using Transfer Learning.

Dataset path : datasets/Brain Tumor/Testing/
Classes       : glioma, meningioma, notumor, pituitary
Strategy      : Uses MobileNetV2 pretrained on ImageNet for transfer learning,
                since the dataset is small (~804 images). Splits 80/20 into
                training and validation.

Output:
  saved_models/cnn_model.keras
  saved_models/cnn_metrics.json
"""

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# ─── PATHS ────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, 'datasets', 'Brain Tumor', 'Testing')
MODEL_DIR    = os.path.join(BASE_DIR, 'saved_models')
MODEL_PATH   = os.path.join(MODEL_DIR, 'cnn_model.keras')
METRICS_PATH = os.path.join(MODEL_DIR, 'cnn_metrics.json')

os.makedirs(MODEL_DIR, exist_ok=True)

# ─── HYPERPARAMETERS ──────────────────────────────────────────────
IMG_SIZE           = (150, 150)
BATCH_SIZE         = 16          # smaller batch for small dataset
EPOCHS             = 30
VALIDATION_SPLIT   = 0.2
CLASSES            = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_LABELS       = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# ─── VERIFY DATA EXISTS ──────────────────────────────────────────
print("=" * 55)
print("  BRAIN TUMOR CNN TRAINING (Transfer Learning)")
print("=" * 55)
print(f"\n  Data directory : {DATA_DIR}")

if not os.path.exists(DATA_DIR):
    print(f"\n  [ERROR] Data directory not found: {DATA_DIR}")
    print("  Please place your brain tumor images in:")
    print(f"    {DATA_DIR}")
    print("  With subfolders: glioma/ meningioma/ notumor/ pituitary/")
    sys.exit(1)

# Count images per class
total = 0
for cls in CLASSES:
    cls_dir = os.path.join(DATA_DIR, cls)
    if os.path.exists(cls_dir):
        count = len([f for f in os.listdir(cls_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"  {cls:>12s} : {count} images")
        total += count
    else:
        print(f"  {cls:>12s} : MISSING!")

print(f"  {'TOTAL':>12s} : {total} images")
print(f"  Train/Val split : {int((1-VALIDATION_SPLIT)*100)}% / {int(VALIDATION_SPLIT*100)}%")
print(f"  Image size      : {IMG_SIZE}")
print(f"  Batch size      : {BATCH_SIZE}")
print(f"  Epochs          : {EPOCHS}")
print(f"  Base model      : MobileNetV2 (ImageNet)")
print()

# ─── IMPORT TENSORFLOW ────────────────────────────────────────────
print("Loading TensorFlow...")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

print(f"  TensorFlow version : {tf.__version__}")
if len(tf.config.list_physical_devices('GPU')) > 0:
    print(f"  GPU detected       : {tf.config.list_physical_devices('GPU')}")
else:
    print("  GPU                : Not available (training on CPU)")
print()

# ─── DATA GENERATORS ──────────────────────────────────────────────
print("Setting up data generators with augmentation...")

# Strong augmentation for small dataset
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=VALIDATION_SPLIT
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=VALIDATION_SPLIT
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASSES,
    subset='training',
    shuffle=True,
    seed=42
)

val_gen = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASSES,
    subset='validation',
    shuffle=False,
    seed=42
)

print(f"  Training samples   : {train_gen.samples}")
print(f"  Validation samples : {val_gen.samples}")
print()

# ─── BUILD MODEL WITH TRANSFER LEARNING ──────────────────────────
print("Building model with MobileNetV2 backbone...")

# Use MobileNetV2 pretrained on ImageNet (lightweight & effective)
base_model = keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model layers initially
base_model.trainable = False

model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(len(CLASSES), activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
print()

# ─── PHASE 1: TRAIN HEAD ONLY ────────────────────────────────────
print("=" * 55)
print("  PHASE 1: Training classification head (base frozen)")
print("=" * 55)

callbacks_phase1 = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=callbacks_phase1,
    verbose=1
)

# ─── PHASE 2: FINE-TUNE TOP LAYERS ───────────────────────────────
print()
print("=" * 55)
print("  PHASE 2: Fine-tuning top layers of MobileNetV2")
print("=" * 55)

# Unfreeze the last 30 layers of MobileNetV2 for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with lower learning rate for fine-tuning
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_phase2 = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks_phase2,
    verbose=1
)

# ─── COMBINE HISTORIES ───────────────────────────────────────────
train_acc  = history1.history['accuracy']  + history2.history['accuracy']
val_acc    = history1.history['val_accuracy'] + history2.history['val_accuracy']
train_loss = history1.history['loss'] + history2.history['loss']
val_loss   = history1.history['val_loss'] + history2.history['val_loss']

# ─── EVALUATE ─────────────────────────────────────────────────────
print("\nEvaluating on validation set...")

final_val_loss, final_val_acc = model.evaluate(val_gen, verbose=0)
print(f"  Validation Loss     : {final_val_loss:.4f}")
print(f"  Validation Accuracy : {final_val_acc*100:.2f}%")

# Classification report
val_gen.reset()
y_pred = model.predict(val_gen, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_gen.classes

print("\n  Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=CLASS_LABELS,
                            zero_division=0))

cm = confusion_matrix(y_true, y_pred_classes)
print("  Confusion Matrix:")
print(cm)

# ─── SAVE MODEL ───────────────────────────────────────────────────
print(f"\nSaving model to: {MODEL_PATH}")
model.save(MODEL_PATH)

# ─── SAVE METRICS ─────────────────────────────────────────────────
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred_classes, average='macro', zero_division=0
)

metrics = {
    'train_acc':  [float(v) for v in train_acc],
    'val_acc':    [float(v) for v in val_acc],
    'train_loss': [float(v) for v in train_loss],
    'val_loss':   [float(v) for v in val_loss],
    'accuracy': float(final_val_acc),
    'precision': float(precision),
    'recall': float(recall),
    'f1': float(f1),
    'final_val_accuracy': float(final_val_acc),
    'final_val_loss':     float(final_val_loss),
    'classes': CLASS_LABELS,
    'cm': cm.tolist(),
    'epochs_trained': len(train_acc),
}

print(f"Saving metrics to: {METRICS_PATH}")
with open(METRICS_PATH, 'w') as f:
    json.dump(metrics, f, indent=2)

# ─── SAVE TO DATABASE ─────────────────────────────────────────────
try:
    from app import create_app
    from database.models import db as app_db, ModelMetrics

    app = create_app()
    with app.app_context():
        # Calculate macro-averaged precision, recall, f1
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_classes, average='macro', zero_division=0
        )

        existing = ModelMetrics.query.filter_by(model_name='CNN').first()
        if existing:
            existing.accuracy  = final_val_acc
            existing.precision = float(precision)
            existing.recall    = float(recall)
            existing.f1_score  = float(f1)
        else:
            app_db.session.add(ModelMetrics(
                model_name='CNN',
                accuracy=final_val_acc,
                precision=float(precision),
                recall=float(recall),
                f1_score=float(f1)
            ))
        app_db.session.commit()
        print("\nMetrics saved to database.")
except Exception as e:
    print(f"\n[WARN] Could not save to DB: {e}")
    print("  (This is OK — the model and metrics JSON are saved.)")

# ─── DONE ─────────────────────────────────────────────────────────
print()
print("=" * 55)
print("  CNN TRAINING COMPLETE")
print("=" * 55)
print(f"  Accuracy  : {final_val_acc*100:.2f}%")
print(f"  Epochs    : {metrics['epochs_trained']}")
print(f"  Model     : {MODEL_PATH}")
print(f"  Metrics   : {METRICS_PATH}")
print("=" * 55)
print()
print("Next steps:")
print("  1. Run the app:  python app.py")
print("  2. Go to CNN Prediction page to classify brain MRI scans")
print()
