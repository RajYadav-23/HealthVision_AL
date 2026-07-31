import os, json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from flask import Blueprint, render_template, request, flash, redirect, url_for, session
from flask_login import login_required
from config import Config

viz_bp = Blueprint('viz', __name__)


def _load_df() -> pd.DataFrame | None:
    fname = session.get('explorer_file')
    if not fname:
        return None
    path = os.path.join(Config.UPLOAD_FOLDER, fname)
    return pd.read_csv(path) if os.path.exists(path) else None


def _fig_json(fig) -> str:
    return json.dumps(fig, cls=go.Figure.__class__.__mro__[-1].__subclasses__()[0]
                      if False else None) if False else fig.to_json()


@viz_bp.route('/visualization', methods=['GET', 'POST'])
@login_required
def index():
    df = _load_df()
    chart_json = None
    chart_type = request.form.get('chart_type', 'histogram')
    col_x = request.form.get('col_x', '')
    col_y = request.form.get('col_y', '')
    col_color = request.form.get('col_color', '')

    num_cols, cat_cols, all_cols = [], [], []
    if df is not None:
        num_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = df.select_dtypes(exclude='number').columns.tolist()
        all_cols = df.columns.tolist()

        if request.method == 'POST' and col_x:
            try:
                fig = _build_chart(df, chart_type, col_x, col_y, col_color)
                chart_json = fig.to_json() if fig else None
            except Exception as e:
                flash(f'Chart error: {e}', 'danger')
    else:
        if request.method == 'POST':
            flash('Please upload a dataset in Dataset Explorer first.', 'warning')

    return render_template('modules/visualization.html',
                           chart_json=chart_json, chart_type=chart_type,
                           col_x=col_x, col_y=col_y, col_color=col_color,
                           num_cols=num_cols, cat_cols=cat_cols, all_cols=all_cols,
                           has_data=df is not None)


def _build_chart(df: pd.DataFrame, chart_type: str, col_x: str, col_y: str, col_color: str):
    color = col_color if col_color else None
    TEMPLATE = 'plotly_white'

    if chart_type == 'histogram':
        return px.histogram(df, x=col_x, color=color, template=TEMPLATE, nbins=30)
    elif chart_type == 'box':
        return px.box(df, x=color, y=col_x, template=TEMPLATE)
    elif chart_type == 'scatter':
        if not col_y:
            return None
        return px.scatter(df, x=col_x, y=col_y, color=color, template=TEMPLATE, opacity=0.7)
    elif chart_type == 'pie':
        counts = df[col_x].value_counts().reset_index()
        counts.columns = [col_x, 'count']
        return px.pie(counts, names=col_x, values='count', template=TEMPLATE, hole=0.4)
    elif chart_type == 'heatmap':
        corr = df.select_dtypes(include='number').corr()
        return px.imshow(corr, text_auto='.2f', template=TEMPLATE,
                         color_continuous_scale='RdBu_r', aspect='auto')
    elif chart_type == 'bar':
        if not col_y:
            counts = df[col_x].value_counts().reset_index()
            counts.columns = [col_x, 'count']
            return px.bar(counts, x=col_x, y='count', template=TEMPLATE)
        return px.bar(df, x=col_x, y=col_y, color=color, template=TEMPLATE)
    elif chart_type == 'line':
        if not col_y:
            return None
        return px.line(df, x=col_x, y=col_y, color=color, template=TEMPLATE)
    elif chart_type == 'violin':
        return px.violin(df, x=color, y=col_x, box=True, template=TEMPLATE)
    return None
