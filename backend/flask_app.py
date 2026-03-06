"""
Flask Web Application and REST API
Frontend interface for Hydroponic ML Project
"""

from flask import Flask, render_template, request, jsonify, send_file

from flask_cors import CORS
import numpy as np
import pandas as pd

import json
import os
import importlib.util
from datetime import datetime
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

if importlib.util.find_spec('tensorflow'):
    import tensorflow as tf
else:
    tf = None

# Initialize Flask app
# Initialize Flask app
# Since the frontend is deployed separately on Vercel, the backend acts as a standalone API.
app = Flask(__name__)
CORS(app)

# Global variables for models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

baseline_model = None
optimized_model = None
scaler = None


def heuristic_probability(features):
    """Fallback probability when trained models are unavailable."""
    ph, tds, water_level, dht_temp, dht_humidity, water_temp = features

    score = 1.0
    if not (5.5 <= ph <= 6.5):
        score -= 0.25
    if not (900 <= tds <= 1600):
        score -= 0.20
    if not (1 <= water_level <= 2):
        score -= 0.20
    if not (18 <= dht_temp <= 30):
        score -= 0.15
    if not (55 <= dht_humidity <= 80):
        score -= 0.10
    if not (18 <= water_temp <= 24):
        score -= 0.10

    return float(np.clip(score, 0.01, 0.99))

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_models():
    """Load trained models from disk"""
    global baseline_model, optimized_model

    if tf is None:
        return

    try:
        baseline_path = os.path.join(MODELS_DIR, 'baseline_cnn.h5')
        if os.path.exists(baseline_path):
            baseline_model = tf.keras.models.load_model(baseline_path)
    except:
        pass

    try:
        optimized_path = os.path.join(MODELS_DIR, 'pso_optimized_cnn.h5')
        if os.path.exists(optimized_path):
            optimized_model = tf.keras.models.load_model(optimized_path)
    except:
        pass


def preprocess_input(data_array):
    """Preprocess input data for prediction"""
    # Normalize (assuming standard scaler with mean=0, std=1)
    data_array = (data_array - np.mean(data_array)) / (np.std(data_array) + 1e-8)
    return np.expand_dims(data_array, axis=-1)


# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    """API Root page"""
    return jsonify({
        "status": "online",
        "message": "Hydroponic ML API is running. Please use your Vercel frontend URL to access the user interface.",
        "endpoints": {
            "health": "/api/health",
            "predict": "/api/predict",
            "batch_predict": "/api/batch-predict",
            "metrics": "/api/metrics"
        }
    })


@app.route('/dashboard')
def dashboard():
    """Dashboard page"""
    if os.path.exists('dashboard.html'):
        return render_template('dashboard.html')
    return jsonify({'error': 'dashboard.html not found'}), 404


@app.route('/api/health')
def health_check():
    """API health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'tensorflow_available': tf is not None,
        'baseline_model_loaded': baseline_model is not None,
        'optimized_model_loaded': optimized_model is not None
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Make predictions using trained models
    Expects JSON: {
        'ph': float,
        'tds': float,
        'water_level': int,
        'dht_temp': float,
        'dht_humidity': float,
        'water_temp': float
    }
    """
    try:
        data = request.json

        # Extract features in correct order
        features = np.array([[
            data.get('ph', 6.0),
            data.get('tds', 1200),
            data.get('water_level', 1),
            data.get('dht_temp', 24),
            data.get('dht_humidity', 70),
            data.get('water_temp', 21)
        ]], dtype=float)

        # Preprocess
        features_processed = preprocess_input(features)

        predictions = {}

        # Baseline prediction
        if baseline_model:
            baseline_pred = baseline_model.predict(features_processed, verbose=0)
            predictions['baseline'] = {
                'probability': float(baseline_pred[0][0]),
                'health_status': 'Healthy' if baseline_pred[0][0] > 0.5 else 'Unhealthy',
                'confidence': float(max(baseline_pred[0][0], 1 - baseline_pred[0][0]))
            }

        # Optimized prediction
        if optimized_model:
            optimized_pred = optimized_model.predict(features_processed, verbose=0)
            predictions['optimized'] = {
                'probability': float(optimized_pred[0][0]),
                'health_status': 'Healthy' if optimized_pred[0][0] > 0.5 else 'Unhealthy',
                'confidence': float(max(optimized_pred[0][0], 1 - optimized_pred[0][0]))
            }

        if not predictions:
            fallback_prob = heuristic_probability(features[0])
            fallback_payload = {
                'probability': fallback_prob,
                'health_status': 'Healthy' if fallback_prob > 0.5 else 'Unhealthy',
                'confidence': float(max(fallback_prob, 1 - fallback_prob)),
                'fallback': True
            }
            predictions = {
                'baseline': fallback_payload,
                'optimized': fallback_payload
            }

        return jsonify({
            'success': True,
            'input_data': data,
            'predictions': predictions,
            'models_loaded': {
                'baseline': baseline_model is not None,
                'optimized': optimized_model is not None
            },
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction from CSV file
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        # Read CSV
        df = pd.read_csv(file)

        # Expected columns
        feature_cols = ['pH', 'TDS', 'water_level', 'DHT_temp', 'DHT_humidity', 'water_temp']

        if not all(col in df.columns for col in feature_cols):
            return jsonify({'error': 'Missing required columns'}), 400

        # Prepare features
        features = df[feature_cols].values.astype(float)
        X = preprocess_input(features)

        results = []

        if baseline_model:
            baseline_preds = baseline_model.predict(X, verbose=0)
            results.append({
                'model': 'baseline',
                'predictions': baseline_preds.flatten().tolist()
            })

        if optimized_model:
            optimized_preds = optimized_model.predict(X, verbose=0)
            results.append({
                'model': 'optimized',
                'predictions': optimized_preds.flatten().tolist()
            })

        if not results:
            fallback_preds = [heuristic_probability(row) for row in features]
            results.append({
                'model': 'baseline',
                'predictions': fallback_preds,
                'fallback': True
            })
            results.append({
                'model': 'optimized',
                'predictions': fallback_preds,
                'fallback': True
            })

        return jsonify({
            'success': True,
            'num_samples': len(df),
            'results': results
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/metrics')
def get_metrics():
    """Get model metrics from saved results"""
    try:
        metrics_files = [
            os.path.join(RESULTS_DIR, 'metrics_comparison.json'),
            os.path.join(RESULTS_DIR, 'metrics_sample.json')
        ]

        for metrics_file in metrics_files:
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                return jsonify({'success': True, 'metrics': metrics})

        return jsonify({'success': False, 'error': 'No metrics found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/generate-report')
def generate_report():
    """Generate comprehensive report"""
    try:
        report = {
            'timestamp': datetime.now().isoformat(),
            'project': 'Hybrid Metaheuristic Optimization of Deep Learning for Hydroponics',
            'models': {
                'baseline': 'CNN with default hyperparameters',
                'optimized': 'CNN with PSO-optimized hyperparameters'
            },
            'dataset': {
                'source': 'IoT Hydroponic Sensor Data',
                'total_samples': 50570,
                'features': ['pH', 'TDS', 'water_level', 'DHT_temp', 'DHT_humidity', 'water_temp'],
                'target': 'System Health (Binary Classification)'
            },
            'optimization_method': 'Particle Swarm Optimization (PSO)',
            'pso_config': {
                'particles': 5,
                'iterations': 10,
                'inertia_range': [0.4, 0.9],
                'cognitive_param': 2.0,
                'social_param': 2.0
            }
        }

        return jsonify({'success': True, 'report': report})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# =============================================================================
# RUN APPLICATION
# =============================================================================

# Load models on startup (supports both development and production servers like Gunicorn)
load_models()

if __name__ == '__main__':
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
