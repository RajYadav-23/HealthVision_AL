from flask import Blueprint, render_template
from flask_login import login_required, current_user
from database.models import db, User, PredictionHistory, ModelMetrics
from sqlalchemy import func
from datetime import datetime, timedelta

dashboard_bp = Blueprint('dashboard', __name__)


@dashboard_bp.route('/dashboard')
@login_required
def index():
    total = PredictionHistory.query.count()
    healthy = PredictionHistory.query.filter_by(risk_level='Low').count()
    high_risk = PredictionHistory.query.filter_by(risk_level='High').count()
    users = User.query.count()

    ann_metrics = ModelMetrics.query.filter_by(model_name='ANN').first()
    cnn_metrics = ModelMetrics.query.filter_by(model_name='CNN').first()
    ann_acc = round(ann_metrics.accuracy * 100, 1) if ann_metrics else 0
    cnn_acc = round(cnn_metrics.accuracy * 100, 1) if cnn_metrics else 0

    recent = (PredictionHistory.query
              .order_by(PredictionHistory.created_at.desc())
              .limit(8).all())

    # Real daily prediction counts for last 7 days
    today = datetime.utcnow().date()
    days = [(today - timedelta(days=i)) for i in range(6, -1, -1)]
    day_labels = [d.strftime('%a') for d in days]

    daily_counts = {}
    rows = (db.session.query(
                func.date(PredictionHistory.created_at),
                func.count(PredictionHistory.id)
            )
            .filter(PredictionHistory.created_at >= datetime.utcnow() - timedelta(days=7))
            .group_by(func.date(PredictionHistory.created_at))
            .all())
    for date_str, count in rows:
        if isinstance(date_str, str):
            from datetime import date
            daily_counts[date.fromisoformat(date_str)] = count
        else:
            daily_counts[date_str] = count

    day_values = [daily_counts.get(d, 0) for d in days]

    return render_template('dashboard/index.html',
                           total=total, healthy=healthy, high_risk=high_risk,
                           users=users, ann_acc=ann_acc, cnn_acc=cnn_acc,
                           recent=recent,
                           day_labels=day_labels, day_values=day_values)
