# HealthVision AI

An intelligent healthcare analytics platform built with Flask, TensorFlow, and Scikit-learn.

> **Disclaimer:** This is an educational/portfolio project. Not intended for real medical diagnosis.

## Features

- **ANN Disease Prediction** – Diabetes risk prediction using a trained Artificial Neural Network
- **CNN Image Classification** – Brain Tumor detection using a Convolutional Neural Network
- **Dataset Explorer** – Upload and analyze CSV datasets with Pandas
- **Data Visualization** – 8 interactive chart types powered by Plotly
- **Data Preprocessing** – Missing values, encoding, scaling, train/test split
- **Prediction History** – Full history with search, filter, and CSV export
- **Analytics Dashboard** – Trends, distributions, and model usage charts
- **About AI** – Educational page explaining ANN & CNN concepts

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.12, Flask, Flask-Login, Flask-WTF |
| Database | SQLite, SQLAlchemy |
| ML/DL | TensorFlow/Keras, Scikit-learn |
| Data | Pandas, NumPy |
| Visualization | Plotly |
| Frontend | Bootstrap 5, Font Awesome, Jinja2 |

## Setup

### 1. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Dataset Setup

### ANN – Diabetes Prediction
Download the **Pima Indians Diabetes Dataset** from Kaggle:
- File: `diabetes.csv`
- Place in: `datasets/`
- Upload via Dataset Explorer in the app

### CNN – brain tumor
Download **Brain Tumor** from Kaggle:
- Place images in: `datasets/brain_tumor/`
- Run the training notebook in `notebooks/`

## Project Structure

```
HealthVision AI/
├── app.py                  # Flask app factory
├── config.py               # Configuration
├── requirements.txt
├── database/
│   └── models.py           # SQLAlchemy models
├── routes/
│   ├── auth.py             # Login/Register/Logout
│   ├── dashboard.py        # Main dashboard
│   ├── explorer.py         # Dataset Explorer
│   ├── visualization.py    # Plotly charts
│   ├── preprocessing.py    # Data preprocessing
│   ├── ann.py              # ANN training & prediction
│   ├── cnn.py              # CNN image classification
│   ├── history.py          # Prediction history
│   ├── analytics.py        # Analytics dashboard
│   └── about.py            # About AI page
├── utils/
│   ├── ann_utils.py        # ANN model logic
│   └── cnn_utils.py        # CNN model logic
├── templates/
│   ├── base.html           # Sidebar layout
│   ├── auth/               # Login & Register
│   ├── dashboard/          # Main dashboard
│   └── modules/            # All feature pages
├── static/
│   └── css/main.css        # Custom styles
├── saved_models/           # Trained .keras models
├── uploads/                # Uploaded files
└── datasets/               # Raw datasets
```

## Default Credentials

Register a new account on first run — no default credentials.

## License

MIT – Free to use for educational and portfolio purposes.
