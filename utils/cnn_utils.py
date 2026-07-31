import os, json
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from config import Config

CNN_MODEL_PATH  = os.path.join(Config.SAVED_MODELS_FOLDER, 'cnn_model.keras')
CNN_METRICS_PATH = os.path.join(Config.SAVED_MODELS_FOLDER, 'cnn_metrics.json')
IMG_SIZE = (150, 150)
CLASSES  = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

_BG = 'rgba(0,0,0,0)'
_LAYOUT = dict(paper_bgcolor=_BG, plot_bgcolor=_BG,
               font=dict(family='Inter, sans-serif', size=12),
               margin=dict(t=10, b=30, l=40, r=10))


def _get_keras():
    from tensorflow import keras
    return keras


def build_cnn(num_classes: int = 4):
    keras = _get_keras()
    model = keras.Sequential([
        keras.layers.Input(shape=(*IMG_SIZE, 3)),
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def preprocess_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert('RGB').resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_cnn(image_path: str) -> dict:
    keras = _get_keras()
    if not os.path.exists(CNN_MODEL_PATH):
        raise FileNotFoundError('CNN model not trained yet.')
    model = keras.models.load_model(CNN_MODEL_PATH)
    img   = preprocess_image(image_path)
    probs = model.predict(img, verbose=0)[0]
    pred_idx   = int(np.argmax(probs))
    confidence = float(probs[pred_idx]) * 100

    explanations = {
        'Glioma':    'Glioma tumor detected. Occurs in brain/spinal cord. Immediate consultation required.',
        'Meningioma':'Meningioma detected. Tumor on brain membranes. Medical evaluation recommended.',
        'No Tumor':  'No tumor detected. MRI scan appears normal with no significant abnormalities.',
        'Pituitary': 'Pituitary tumor detected. Growth in pituitary gland. Consult a specialist.'
    }
    risk_map = {'Glioma': 'High', 'Meningioma': 'Medium', 'No Tumor': 'Low', 'Pituitary': 'Medium'}

    return {
        'prediction': CLASSES[pred_idx],
        'confidence': round(confidence, 2),
        'risk_level': risk_map[CLASSES[pred_idx]],
        'explanation': explanations[CLASSES[pred_idx]],
        'all_probs':  {cls: round(float(p) * 100, 2) for cls, p in zip(CLASSES, probs)}
    }


def _normalize_metrics(metrics: dict) -> dict:
    metrics = dict(metrics)

    if 'accuracy' not in metrics and 'final_val_accuracy' in metrics:
        metrics['accuracy'] = float(metrics['final_val_accuracy'])

    if any(key not in metrics for key in ('accuracy', 'precision', 'recall', 'f1')):
        cm = metrics.get('cm')
        if cm:
            cm_arr = np.array(cm, dtype=float)
            tp = np.diag(cm_arr)
            fp = cm_arr.sum(axis=0) - tp
            fn = cm_arr.sum(axis=1) - tp

            precision = np.where((tp + fp) > 0, tp / (tp + fp), 0)
            recall = np.where((tp + fn) > 0, tp / (tp + fn), 0)
            f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0)

            metrics['precision'] = float(np.mean(precision))
            metrics['recall'] = float(np.mean(recall))
            metrics['f1'] = float(np.mean(f1))
        else:
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
            metrics['f1'] = 0.0

    if 'accuracy' not in metrics:
        metrics['accuracy'] = float(metrics.get('final_val_accuracy', 0.0))

    metrics['accuracy'] = round(float(metrics.get('accuracy', 0.0)), 4)
    metrics['precision'] = round(float(metrics.get('precision', 0.0)), 4)
    metrics['recall'] = round(float(metrics.get('recall', 0.0)), 4)
    metrics['f1'] = round(float(metrics.get('f1', 0.0)), 4)
    return metrics


def load_cnn_metrics() -> dict | None:
    if not os.path.exists(CNN_METRICS_PATH):
        return None
    with open(CNN_METRICS_PATH) as f:
        return _normalize_metrics(json.load(f))


def build_cnn_charts(metrics: dict) -> dict:
    epochs = list(range(1, len(metrics['train_acc']) + 1))

    acc_fig = go.Figure()
    acc_fig.add_trace(go.Scatter(x=epochs, y=metrics['train_acc'],
                                  name='Train', line=dict(color='#6f42c1')))
    acc_fig.add_trace(go.Scatter(x=epochs, y=metrics['val_acc'],
                                  name='Validation', line=dict(color='#0dcaf0', dash='dash')))
    acc_fig.update_layout(**_LAYOUT, height=220,
                          legend=dict(orientation='h', y=1.1),
                          xaxis=dict(gridcolor='#f1f5f9'),
                          yaxis=dict(gridcolor='#f1f5f9'))

    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(x=epochs, y=metrics['train_loss'],
                                   name='Train', line=dict(color='#dc3545')))
    loss_fig.add_trace(go.Scatter(x=epochs, y=metrics['val_loss'],
                                   name='Validation', line=dict(color='#ffc107', dash='dash')))
    loss_fig.update_layout(**_LAYOUT, height=220,
                           legend=dict(orientation='h', y=1.1),
                           xaxis=dict(gridcolor='#f1f5f9'),
                           yaxis=dict(gridcolor='#f1f5f9'))

    charts = {'acc': acc_fig.to_json(), 'loss': loss_fig.to_json()}

    if metrics.get('cm'):
        labels = metrics.get('classes', CLASSES)
        cm_fig = px.imshow(metrics['cm'], text_auto=True,
                           color_continuous_scale='Purples',
                           x=labels, y=labels,
                           labels=dict(x='Predicted', y='Actual'))
        cm_fig.update_layout(paper_bgcolor=_BG,
                             margin=dict(t=10, b=60, l=80, r=10), height=280)
        charts['cm'] = cm_fig.to_json()

    return charts
