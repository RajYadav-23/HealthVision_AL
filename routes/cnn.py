import os
from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from database.models import db, PredictionHistory, ModelMetrics
from utils.cnn_utils import predict_cnn, load_cnn_metrics, build_cnn_charts
from config import Config

cnn_bp = Blueprint('cnn', __name__)

ALLOWED = {'png', 'jpg', 'jpeg'}


def _allowed(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED


@cnn_bp.route('/cnn', methods=['GET'])
@login_required
def index():
    metrics = load_cnn_metrics()
    charts = build_cnn_charts(metrics) if metrics else None
    return render_template('modules/cnn.html', metrics=metrics, charts=charts, result=None)


@cnn_bp.route('/cnn/predict', methods=['POST'])
@login_required
def predict():
    metrics = load_cnn_metrics()
    charts = build_cnn_charts(metrics) if metrics else None
    result = None
    image_url = None

    file = request.files.get('image')
    if not file or not _allowed(file.filename):
        flash('Please upload a valid image (PNG, JPG, JPEG).', 'danger')
        return render_template('modules/cnn.html', metrics=metrics, charts=charts, result=None)

    filename = secure_filename(file.filename)
    save_path = os.path.join(Config.UPLOAD_FOLDER, filename)
    file.save(save_path)
    image_url = f'/uploads/{filename}'

    try:
        result = predict_cnn(save_path)

        db.session.add(PredictionHistory(
            user_id=current_user.id,
            model_type='CNN',
            prediction=result['prediction'],
            confidence=result['confidence'] / 100,
            risk_level=result['risk_level'],
            input_data=filename
        ))
        db.session.commit()
    except FileNotFoundError:
        flash('CNN model not trained yet. Please train the model first.', 'warning')
    except Exception as e:
        flash(f'Prediction error: {e}', 'danger')

    return render_template('modules/cnn.html', metrics=metrics, charts=charts,
                           result=result, image_url=image_url)
