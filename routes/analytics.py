from collections import Counter
from datetime import datetime, timedelta
from flask import Blueprint, render_template
from flask_login import login_required
from database.models import PredictionHistory
import plotly.graph_objects as go

analytics_bp = Blueprint('analytics', __name__)

_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter, sans-serif', size=12, color='#64748b'),
    margin=dict(t=10, b=40, l=40, r=10), height=280
)


def _empty_fig(msg: str = 'No data yet') -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref='paper', yref='paper',
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=14, color='#94a3b8'))
    fig.update_layout(**_LAYOUT, xaxis_visible=False, yaxis_visible=False)
    return fig


@analytics_bp.route('/analytics')
@login_required
def index():
    all_preds = PredictionHistory.query.order_by(PredictionHistory.created_at).all()

    model_counts  = Counter(p.model_type       for p in all_preds)
    risk_counts   = Counter(p.risk_level or 'Unknown' for p in all_preds)
    outcome_counts = Counter(p.prediction      for p in all_preds)

    # ── Risk Distribution (Donut) ──────────────────────────────
    if risk_counts:
        RISK_COLORS = {'Low': '#198754', 'Medium': '#ffc107',
                       'High': '#dc3545', 'Unknown': '#94a3b8'}
        risk_fig = go.Figure(go.Pie(
            labels=list(risk_counts.keys()),
            values=list(risk_counts.values()),
            hole=0.55,
            marker_colors=[RISK_COLORS.get(k, '#94a3b8') for k in risk_counts],
            textinfo='percent', textfont_size=12
        ))
        risk_fig.update_layout(**_LAYOUT,
                               legend=dict(orientation='h', y=-0.15, font_size=11))
    else:
        risk_fig = _empty_fig()

    # ── Model Usage (Bar) ──────────────────────────────────────
    if model_counts:
        MODEL_COLORS = {'ANN': '#0d6efd', 'CNN': '#6f42c1'}
        model_fig = go.Figure(go.Bar(
            x=list(model_counts.keys()),
            y=list(model_counts.values()),
            marker_color=[MODEL_COLORS.get(k, '#94a3b8') for k in model_counts],
        ))
        model_fig.update_layout(**_LAYOUT, showlegend=False,
                                xaxis=dict(title='', gridcolor='#f1f5f9'),
                                yaxis=dict(title='Count', gridcolor='#f1f5f9', dtick=1))
    else:
        model_fig = _empty_fig()

    # ── Timeline (last 30 days) ────────────────────────────────
    today = datetime.utcnow().date()
    days = [(today - timedelta(days=i)) for i in range(29, -1, -1)]
    day_counts = Counter(p.created_at.date() for p in all_preds)
    timeline_fig = go.Figure(go.Scatter(
        x=[str(d) for d in days],
        y=[day_counts.get(d, 0) for d in days],
        fill='tozeroy', line=dict(color='#0d6efd'),
        fillcolor='rgba(13,110,253,0.1)'
    ))
    timeline_fig.update_layout(**_LAYOUT,
                               xaxis=dict(title='', gridcolor='#f1f5f9'),
                               yaxis=dict(title='Predictions', gridcolor='#f1f5f9', dtick=1))

    # ── Outcome Distribution (Bar) ─────────────────────────────
    if outcome_counts:
        OUTCOME_COLORS = {'Positive': '#dc3545', 'Negative': '#198754',
                          'No Tumor': '#198754', 'Glioma': '#dc3545',
                          'Meningioma': '#d97706', 'Pituitary': '#6f42c1'}
        outcome_fig = go.Figure(go.Bar(
            x=list(outcome_counts.keys()),
            y=list(outcome_counts.values()),
            marker_color=[OUTCOME_COLORS.get(k, '#0d6efd') for k in outcome_counts],
        ))
        outcome_fig.update_layout(**_LAYOUT, showlegend=False,
                                  xaxis=dict(title='', gridcolor='#f1f5f9'),
                                  yaxis=dict(title='Count', gridcolor='#f1f5f9', dtick=1))
    else:
        outcome_fig = _empty_fig()

    stats = {
        'total':          len(all_preds),
        'avg_confidence': round(sum(p.confidence for p in all_preds) / len(all_preds) * 100, 1) if all_preds else 0,
        'high_risk':      sum(1 for p in all_preds if p.risk_level == 'High'),
        'ann_count':      model_counts.get('ANN', 0),
        'cnn_count':      model_counts.get('CNN', 0),
    }

    return render_template('modules/analytics.html',
                           risk_chart=risk_fig.to_json(),
                           model_chart=model_fig.to_json(),
                           timeline_chart=timeline_fig.to_json(),
                           outcome_chart=outcome_fig.to_json(),
                           stats=stats)
