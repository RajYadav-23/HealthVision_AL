import os, json
import pandas as pd
from flask import Blueprint, render_template, request, flash, redirect, url_for, session
from flask_login import login_required
from config import Config

explorer_bp = Blueprint('explorer', __name__)

ALLOWED = {'csv'}

def _allowed(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED

def _analyze(df: pd.DataFrame) -> dict:
    corr = df.select_dtypes(include='number').corr().round(3)
    return {
        'shape': df.shape,
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'missing_pct': (df.isnull().mean() * 100).round(2).to_dict(),
        'duplicates': int(df.duplicated().sum()),
        'stats': json.loads(df.describe(include='all').fillna('').to_json()),
        'preview': json.loads(df.head(10).to_json(orient='records')),
        'columns': df.columns.tolist(),
        'corr': corr.to_dict(),
        'corr_cols': corr.columns.tolist(),
    }


@explorer_bp.route('/explorer', methods=['GET', 'POST'])
@login_required
def index():
    analysis = None
    filename = session.get('explorer_file')

    if request.method == 'POST':
        file = request.files.get('csv_file')
        if not file or not _allowed(file.filename):
            flash('Please upload a valid CSV file.', 'danger')
            return redirect(url_for('explorer.index'))
        path = os.path.join(Config.UPLOAD_FOLDER, file.filename)
        file.save(path)
        session['explorer_file'] = file.filename
        filename = file.filename

    if filename:
        path = os.path.join(Config.UPLOAD_FOLDER, filename)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                analysis = _analyze(df)
            except Exception as e:
                flash(f'Error reading file: {e}', 'danger')

    return render_template('modules/explorer.html', analysis=analysis, filename=filename)
