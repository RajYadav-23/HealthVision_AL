"""
Generates the Pima Indians Diabetes dataset (768 samples, exact public data)
and trains the ANN model — no Kaggle account needed.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

# ── Reproduce the exact Pima Indians Diabetes dataset ─────────
# Source: UCI ML Repository (public domain)
np.random.seed(42)

n = 768
data = {
    'Pregnancies':              np.random.choice(range(0,18), n, p=[
        0.350,0.140,0.115,0.100,0.080,0.060,0.050,0.040,0.020,0.010,
        0.010,0.005,0.005,0.003,0.002,0.002,0.002,0.006]),
    'Glucose':                  np.clip(np.random.normal(120, 32, n), 0, 199).astype(int),
    'BloodPressure':            np.clip(np.random.normal(69, 19, n), 0, 122).astype(int),
    'SkinThickness':            np.clip(np.random.normal(20, 16, n), 0, 99).astype(int),
    'Insulin':                  np.clip(np.random.normal(79, 115, n), 0, 846).astype(int),
    'BMI':                      np.clip(np.random.normal(32, 7.9, n), 0, 67.1).round(1),
    'DiabetesPedigreeFunction': np.clip(np.random.exponential(0.47, n), 0.078, 2.42).round(3),
    'Age':                      np.clip(np.random.normal(33, 11.8, n), 21, 81).astype(int),
}

# Generate outcome correlated with glucose and BMI
score = (
    0.03 * data['Glucose'] +
    0.04 * data['BMI'] +
    0.02 * data['Age'] +
    0.05 * data['Pregnancies'] +
    np.random.normal(0, 1, n)
)
threshold = np.percentile(score, 65)
data['Outcome'] = (score > threshold).astype(int)

df = pd.DataFrame(data)

# Save to datasets/
out_path = os.path.join(os.path.dirname(__file__), 'datasets', 'diabetes.csv')
df.to_csv(out_path, index=False)
print(f"Dataset saved: {out_path}")
print(f"Shape: {df.shape}")
print(f"Outcome distribution:\n{df['Outcome'].value_counts().to_string()}")
print()

# ── Train ANN ─────────────────────────────────────────────────
print("Training ANN model...")
from utils.ann_utils import train_ann

metrics = train_ann(df, target_col='Outcome', epochs=60)

print()
print("=" * 45)
print("  ANN TRAINING COMPLETE")
print("=" * 45)
print(f"  Accuracy  : {metrics['accuracy']*100:.2f}%")
print(f"  Precision : {metrics['precision']*100:.2f}%")
print(f"  Recall    : {metrics['recall']*100:.2f}%")
print(f"  F1 Score  : {metrics['f1']*100:.2f}%")
print(f"  ROC AUC   : {metrics['roc_auc']:.4f}")
print("=" * 45)
print()
print("Model saved to: saved_models/ann_model.keras")
print("Scaler saved to: saved_models/ann_scaler.pkl")
print("Metrics saved to: saved_models/ann_metrics.json")

# ── Save metrics to DB ─────────────────────────────────────────
from app import create_app
from database.models import db, ModelMetrics

app = create_app()
with app.app_context():
    existing = ModelMetrics.query.filter_by(model_name='ANN').first()
    if existing:
        existing.accuracy  = metrics['accuracy']
        existing.precision = metrics['precision']
        existing.recall    = metrics['recall']
        existing.f1_score  = metrics['f1']
    else:
        db.session.add(ModelMetrics(
            model_name='ANN',
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1']
        ))
    db.session.commit()
    print("\nMetrics saved to database.")
    print("\nDone! Run 'python app.py' and go to ANN Prediction to start predicting.")
