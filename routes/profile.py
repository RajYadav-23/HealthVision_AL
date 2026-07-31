from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_login import login_required, current_user
from database.models import db, PredictionHistory
from werkzeug.security import check_password_hash

profile_bp = Blueprint('profile', __name__)


@profile_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def index():
    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'update_username':
            new_username = request.form.get('username', '').strip()
            if len(new_username) < 3:
                flash('Username must be at least 3 characters.', 'danger')
            elif new_username == current_user.username:
                flash('That is already your username.', 'info')
            else:
                from database.models import User
                if User.query.filter_by(username=new_username).first():
                    flash('Username already taken.', 'danger')
                else:
                    current_user.username = new_username
                    db.session.commit()
                    flash('Username updated successfully.', 'success')

        elif action == 'change_password':
            current_pw = request.form.get('current_password', '')
            new_pw = request.form.get('new_password', '')
            confirm_pw = request.form.get('confirm_password', '')

            if not current_user.check_password(current_pw):
                flash('Current password is incorrect.', 'danger')
            elif len(new_pw) < 6:
                flash('New password must be at least 6 characters.', 'danger')
            elif new_pw != confirm_pw:
                flash('Passwords do not match.', 'danger')
            else:
                current_user.set_password(new_pw)
                db.session.commit()
                flash('Password changed successfully.', 'success')

        return redirect(url_for('profile.index'))

    total_preds = PredictionHistory.query.filter_by(user_id=current_user.id).count()
    ann_preds   = PredictionHistory.query.filter_by(user_id=current_user.id, model_type='ANN').count()
    cnn_preds   = PredictionHistory.query.filter_by(user_id=current_user.id, model_type='CNN').count()
    high_risk   = PredictionHistory.query.filter_by(user_id=current_user.id, risk_level='High').count()

    return render_template('modules/profile.html',
                           total_preds=total_preds, ann_preds=ann_preds,
                           cnn_preds=cnn_preds, high_risk=high_risk)
