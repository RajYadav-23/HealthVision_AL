import os
import pandas as pd
from flask import Blueprint, render_template, request, flash, redirect, url_for, session, jsonify
from flask_login import login_required, current_user
from database.models import db, PredictionHistory, ModelMetrics
from utils.ann_utils import predict_ann, load_metrics, build_training_charts
from utils.training_manager import start_ann_training, get_state
from config import Config
import json

ann_bp = Blueprint('ann', __name__)

# Default Pima Diabetes feature set
DEFAULT_FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']


@ann_bp.route('/ann', methods=['GET'])
@login_required
def index():
    metrics = load_metrics()
    charts = build_training_charts(metrics) if metrics else None
    return render_template('modules/ann.html', metrics=metrics, charts=charts,
                           features=DEFAULT_FEATURES, result=None)


@ann_bp.route('/ann/train', methods=['POST'])
@login_required
def train():
    fname = session.get('explorer_file') or session.get('processed_file')
    if not fname:
        flash('Please upload a dataset first.', 'warning')
        return redirect(url_for('ann.index'))

    path = os.path.join(Config.UPLOAD_FOLDER, fname)
    if not os.path.exists(path):
        flash('Dataset file not found.', 'danger')
        return redirect(url_for('ann.index'))

    try:
        df = pd.read_csv(path)
        target = request.form.get('target_col', df.columns[-1])
        epochs = int(request.form.get('epochs', 50))

        # Pass current app context to background thread
        from flask import current_app
        _app = current_app._get_current_object()

        def on_done(m):
            """Save metrics to DB after training — runs in background thread."""
            with _app.app_context():
                existing = ModelMetrics.query.filter_by(model_name='ANN').first()
                if existing:
                    existing.accuracy  = m['accuracy']
                    existing.precision = m['precision']
                    existing.recall    = m['recall']
                    existing.f1_score  = m['f1']
                else:
                    db.session.add(ModelMetrics(
                        model_name='ANN', accuracy=m['accuracy'],
                        precision=m['precision'], recall=m['recall'],
                        f1_score=m['f1']
                    ))
                db.session.commit()

        started = start_ann_training(df, target, epochs, on_done)
        if not started:
            flash('Training is already running. Please wait.', 'warning')
        else:
            flash('Training started in background. Watch the progress bar below.', 'info')
    except Exception as e:
        flash(f'Error starting training: {e}', 'danger')

    return redirect(url_for('ann.index'))


@ann_bp.route('/ann/train/status')
@login_required
def train_status():
    """Polling endpoint — returns JSON training progress."""
    return jsonify(get_state())


# Valid input ranges for Pima Diabetes features
FEATURE_RANGES = {
    'Pregnancies':              (0, 20),
    'Glucose':                  (0, 300),
    'BloodPressure':            (0, 200),
    'SkinThickness':            (0, 100),
    'Insulin':                  (0, 900),
    'BMI':                      (0, 70),
    'DiabetesPedigreeFunction': (0.0, 3.0),
    'Age':                      (1, 120),
}


@ann_bp.route('/ann/predict', methods=['POST'])
@login_required
def predict():
    metrics = load_metrics()
    charts = build_training_charts(metrics) if metrics else None

    # Validate inputs
    errors = []
    features = {}
    for k in DEFAULT_FEATURES:
        raw = request.form.get(k, '').strip()
        if raw == '':
            errors.append(f'{k} is required.')
            continue
        try:
            val = float(raw)
        except ValueError:
            errors.append(f'{k} must be a number.')
            continue
        lo, hi = FEATURE_RANGES[k]
        if not (lo <= val <= hi):
            errors.append(f'{k} must be between {lo} and {hi}.')
            continue
        features[k] = val

    if errors:
        for e in errors:
            flash(e, 'danger')
        return render_template('modules/ann.html', metrics=metrics, charts=charts,
                               features=DEFAULT_FEATURES, result=None,
                               form_data=request.form)

    try:
        result = predict_ann(features)
        db.session.add(PredictionHistory(
            user_id=current_user.id,
            model_type='ANN',
            prediction=result['prediction'],
            confidence=result['confidence'] / 100,
            risk_level=result['risk_level'],
            input_data=json.dumps(features)
        ))
        db.session.commit()
    except FileNotFoundError:
        flash('ANN model not trained yet. Please train the model first.', 'warning')
        return redirect(url_for('ann.index'))
    except Exception as e:
        flash(f'Prediction error: {e}', 'danger')
        result = None

    return render_template('modules/ann.html', metrics=metrics, charts=charts,
                           features=DEFAULT_FEATURES, result=result,
                           form_data=request.form)
