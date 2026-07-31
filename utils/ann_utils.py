import os, json
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_curve, auc, accuracy_score)
from config import Config

def _get_tf():
    import tensorflow as tf
    from tensorflow import keras
    return tf, keras

ANN_MODEL_PATH = os.path.join(Config.SAVED_MODELS_FOLDER, 'ann_model.keras')
SCALER_PATH    = os.path.join(Config.SAVED_MODELS_FOLDER, 'ann_scaler.pkl')
METRICS_PATH   = os.path.join(Config.SAVED_MODELS_FOLDER, 'ann_metrics.json')

_BG = 'rgba(0,0,0,0)'
_LAYOUT = dict(paper_bgcolor=_BG, plot_bgcolor=_BG,
               font=dict(family='Inter, sans-serif', size=12),
               margin=dict(t=10, b=30, l=40, r=10))


def build_ann(input_dim: int):
    _, keras = _get_tf()
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def _compute_metrics(model, X_test, y_test, history, X, epochs_run) -> dict:
    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    acc    = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)
    cm     = confusion_matrix(y_test, y_pred).tolist()
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    return {
        'accuracy':      round(acc, 4),
        'precision':     round(report['weighted avg']['precision'], 4),
        'recall':        round(report['weighted avg']['recall'], 4),
        'f1':            round(report['weighted avg']['f1-score'], 4),
        'roc_auc':       round(float(auc(fpr, tpr)), 4),
        'cm':            cm,
        'train_acc':     history.history['accuracy'],
        'val_acc':       history.history['val_accuracy'],
        'train_loss':    history.history['loss'],
        'val_loss':      history.history['val_loss'],
        'fpr':           fpr.tolist(),
        'tpr':           tpr.tolist(),
        'feature_names': X.columns.tolist(),
        'report':        report,
    }


def train_ann_with_progress(df: pd.DataFrame, target_col: str,
                             epochs: int, progress_cb) -> dict:
    tf, keras = _get_tf()
    X = df.drop(columns=[target_col]).select_dtypes(include='number')
    y = df[target_col].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    model = build_ann(X_train.shape[1])

    class _CB(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress_cb(epoch + 1, epochs, logs or {})

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=epochs, batch_size=32, verbose=0, callbacks=[_CB()])
    model.save(ANN_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    metrics = _compute_metrics(model, X_test, y_test, history, X, epochs)
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f)
    return metrics


def train_ann(df: pd.DataFrame, target_col: str, epochs: int = 50) -> dict:
    tf, keras = _get_tf()
    X = df.drop(columns=[target_col]).select_dtypes(include='number')
    y = df[target_col].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    model = build_ann(X_train.shape[1])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=epochs, batch_size=32, verbose=0)
    model.save(ANN_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    metrics = _compute_metrics(model, X_test, y_test, history, X, epochs)
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f)
    return metrics


def predict_ann(features: dict) -> dict:
    _, keras = _get_tf()
    if not os.path.exists(ANN_MODEL_PATH):
        raise FileNotFoundError('ANN model not trained yet.')
    model  = keras.models.load_model(ANN_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(METRICS_PATH) as f:
        saved = json.load(f)
    feat_arr   = np.array([[features[k] for k in saved['feature_names']]])
    feat_scaled = scaler.transform(feat_arr)
    prob = float(model.predict(feat_scaled, verbose=0)[0][0])
    pred = int(prob >= 0.5)
    risk = 'High' if prob >= 0.7 else ('Medium' if prob >= 0.4 else 'Low')
    recommendation = {
        'High':   'Immediate medical consultation recommended. Monitor vitals closely.',
        'Medium': 'Schedule a check-up. Maintain a healthy lifestyle.',
        'Low':    'Low risk detected. Continue regular health monitoring.'
    }[risk]
    return {
        'prediction':    'Positive' if pred else 'Negative',
        'probability':   round(prob * 100, 2),
        'risk_level':    risk,
        'confidence':    round(prob * 100 if pred else (1 - prob) * 100, 2),
        'recommendation': recommendation,
    }


def load_metrics() -> dict | None:
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


def build_training_charts(metrics: dict) -> dict:
    epochs = list(range(1, len(metrics['train_acc']) + 1))

    acc_fig = go.Figure()
    acc_fig.add_trace(go.Scatter(x=epochs, y=metrics['train_acc'],
                                  name='Train', line=dict(color='#0d6efd')))
    acc_fig.add_trace(go.Scatter(x=epochs, y=metrics['val_acc'],
                                  name='Validation', line=dict(color='#0dcaf0', dash='dash')))
    acc_fig.update_layout(**_LAYOUT, height=250,
                          legend=dict(orientation='h', y=1.1),
                          xaxis=dict(gridcolor='#f1f5f9'),
                          yaxis=dict(gridcolor='#f1f5f9'))

    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(x=epochs, y=metrics['train_loss'],
                                   name='Train', line=dict(color='#dc3545')))
    loss_fig.add_trace(go.Scatter(x=epochs, y=metrics['val_loss'],
                                   name='Validation', line=dict(color='#ffc107', dash='dash')))
    loss_fig.update_layout(**_LAYOUT, height=250,
                           legend=dict(orientation='h', y=1.1),
                           xaxis=dict(gridcolor='#f1f5f9'),
                           yaxis=dict(gridcolor='#f1f5f9'))

    cm_fig = px.imshow(metrics['cm'], text_auto=True, color_continuous_scale='Blues',
                       labels=dict(x='Predicted', y='Actual'),
                       x=['Negative', 'Positive'], y=['Negative', 'Positive'])
    cm_fig.update_layout(paper_bgcolor=_BG, margin=dict(t=10, b=30, l=60, r=10), height=250)

    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=metrics['fpr'], y=metrics['tpr'],
                                  name=f"AUC = {metrics['roc_auc']:.3f}",
                                  line=dict(color='#0d6efd')))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                  line=dict(dash='dash', color='#94a3b8'),
                                  showlegend=False))
    roc_fig.update_layout(**_LAYOUT, height=250,
                           xaxis=dict(title='FPR', gridcolor='#f1f5f9'),
                           yaxis=dict(title='TPR', gridcolor='#f1f5f9'))

    return {
        'acc':  acc_fig.to_json(),
        'loss': loss_fig.to_json(),
        'cm':   cm_fig.to_json(),
        'roc':  roc_fig.to_json(),
    }
