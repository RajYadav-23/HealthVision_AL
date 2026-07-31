import os, json
import pandas as pd
import numpy as np
from flask import Blueprint, render_template, request, flash, session
from flask_login import login_required
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from config import Config

preprocess_bp = Blueprint('preprocess', __name__)


def _load_df() -> pd.DataFrame | None:
    fname = session.get('explorer_file')
    if not fname:
        return None
    path = os.path.join(Config.UPLOAD_FOLDER, fname)
    return pd.read_csv(path) if os.path.exists(path) else None


@preprocess_bp.route('/preprocessing', methods=['GET', 'POST'])
@login_required
def index():
    df = _load_df()
    steps = []
    result_info = None

    if df is None:
        flash('Please upload a dataset in Dataset Explorer first.', 'warning')
        return render_template('modules/preprocessing.html', steps=steps, has_data=False)

    original_shape = df.shape

    if request.method == 'POST':
        actions = request.form.getlist('actions')

        if 'drop_missing' in actions:
            before = len(df)
            df = df.dropna()
            steps.append({
                'name': 'Drop Missing Values',
                'icon': 'fa-trash-can',
                'color': 'var(--danger)',
                'desc': f'Removed rows with null values.',
                'detail': f'Rows before: {before} → After: {len(df)} (removed {before - len(df)})'
            })

        if 'fill_missing' in actions:
            num_cols = df.select_dtypes(include='number').columns
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
            cat_cols = df.select_dtypes(exclude='number').columns
            df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0] if len(cat_cols) > 0 else df[cat_cols])
            steps.append({
                'name': 'Fill Missing Values',
                'icon': 'fa-fill-drip',
                'color': 'var(--warning)',
                'desc': 'Filled numeric nulls with median, categorical with mode.',
                'detail': f'Numeric columns filled: {len(num_cols)}'
            })

        if 'drop_duplicates' in actions:
            before = len(df)
            df = df.drop_duplicates()
            steps.append({
                'name': 'Remove Duplicates',
                'icon': 'fa-copy',
                'color': 'var(--teal)',
                'desc': 'Removed duplicate rows from the dataset.',
                'detail': f'Removed {before - len(df)} duplicate rows'
            })

        if 'label_encode' in actions:
            cat_cols = df.select_dtypes(include='object').columns.tolist()
            le = LabelEncoder()
            for col in cat_cols:
                df[col] = le.fit_transform(df[col].astype(str))
            steps.append({
                'name': 'Label Encoding',
                'icon': 'fa-tags',
                'color': '#6f42c1',
                'desc': 'Converted categorical text columns to numeric labels.',
                'detail': f'Encoded columns: {", ".join(cat_cols) if cat_cols else "None"}'
            })

        if 'scale' in actions:
            num_cols = df.select_dtypes(include='number').columns.tolist()
            scaler = StandardScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])
            steps.append({
                'name': 'Standard Scaling',
                'icon': 'fa-ruler',
                'color': 'var(--primary)',
                'desc': 'Scaled numeric features to mean=0, std=1.',
                'detail': f'Scaled {len(num_cols)} numeric columns'
            })

        if 'split' in actions:
            test_size = float(request.form.get('test_size', 0.2))
            steps.append({
                'name': 'Train/Test Split',
                'icon': 'fa-scissors',
                'color': 'var(--success)',
                'desc': f'Split dataset into training and testing sets.',
                'detail': f'Train: {int(len(df)*(1-test_size))} rows | Test: {int(len(df)*test_size)} rows ({int(test_size*100)}% test)'
            })

        result_info = {
            'original_shape': original_shape,
            'final_shape': df.shape,
            'preview': json.loads(df.head(5).to_json(orient='records')),
            'columns': df.columns.tolist()
        }

        # Save processed file
        processed_path = os.path.join(Config.UPLOAD_FOLDER, 'processed_dataset.csv')
        df.to_csv(processed_path, index=False)
        session['processed_file'] = 'processed_dataset.csv'

    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(exclude='number').columns.tolist()

    return render_template('modules/preprocessing.html',
                           steps=steps, result_info=result_info,
                           has_data=True, num_cols=num_cols, cat_cols=cat_cols,
                           shape=df.shape)
