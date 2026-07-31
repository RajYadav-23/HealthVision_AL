import csv, io
from flask import Blueprint, render_template, request, redirect, url_for, flash, make_response
from flask_login import login_required, current_user
from database.models import db, PredictionHistory

history_bp = Blueprint('history', __name__)


@history_bp.route('/history')
@login_required
def index():
    q = request.args.get('q', '').strip()
    model_filter = request.args.get('model', '')
    risk_filter = request.args.get('risk', '')
    page = request.args.get('page', 1, type=int)

    query = PredictionHistory.query.order_by(PredictionHistory.created_at.desc())

    if q:
        query = query.filter(PredictionHistory.prediction.ilike(f'%{q}%'))
    if model_filter:
        query = query.filter_by(model_type=model_filter)
    if risk_filter:
        query = query.filter_by(risk_level=risk_filter)

    pagination = query.paginate(page=page, per_page=15, error_out=False)
    return render_template('modules/history.html',
                           predictions=pagination.items,
                           pagination=pagination,
                           q=q, model_filter=model_filter, risk_filter=risk_filter)


@history_bp.route('/history/delete/<int:pid>', methods=['POST'])
@login_required
def delete(pid: int):
    p = PredictionHistory.query.get_or_404(pid)
    if p.user_id != current_user.id:
        flash('You are not authorized to delete this prediction.', 'danger')
        return redirect(url_for('history.index'))
    db.session.delete(p)
    db.session.commit()
    flash('Prediction deleted.', 'info')
    return redirect(url_for('history.index'))


@history_bp.route('/history/export')
@login_required
def export():
    predictions = PredictionHistory.query.order_by(PredictionHistory.created_at.desc()).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'User', 'Model', 'Prediction', 'Confidence', 'Risk Level', 'Date'])
    for p in predictions:
        writer.writerow([p.id, p.user.username, p.model_type, p.prediction,
                         f'{p.confidence*100:.1f}%', p.risk_level or 'N/A',
                         p.created_at.strftime('%Y-%m-%d %H:%M')])
    response = make_response(output.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=prediction_history.csv'
    response.headers['Content-Type'] = 'text/csv'
    return response
