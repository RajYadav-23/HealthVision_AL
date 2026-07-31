from flask import Blueprint, render_template
from flask_login import login_required

about_bp = Blueprint('about', __name__)

@about_bp.route('/about')
@login_required
def index():
    return render_template('modules/about.html')
