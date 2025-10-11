"""
Medical Report Analysis ML Model
Flask API for disease prediction based on medical reports and test values
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import re
import numpy as np
from datetime import datetime
import logging
import sys
import os

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

# Import ML models
try:
    from diabetes_model import DiabetesModel
    from mri_model import MRIModel
    from xray_model import XRayModel
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import ML models: {e}")
    MODELS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Initialize ML models
if MODELS_AVAILABLE:
    diabetes_model = DiabetesModel()
    mri_model = MRIModel()
    xray_model = XRayModel()
    
    # Load or train models
    try:
        diabetes_model.load_model()
        mri_model.load_model()
        xray_model.load_model()
        logger.info("All ML models loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load models, will train new ones: {e}")
        diabetes_model.train()
        mri_model.train()
        xray_model.train()
else:
    logger.warning("ML models not available, using fallback predictions only")

# Mock ML model for demonstration (fallback)
class MockDiseasePredictor:
    def __init__(self):
        self.disease_patterns = {
            'diabetes': {
                'keywords': ['glucose', 'blood sugar', 'diabetes', 'hyperglycemia'],
                'lab_values': {'glucose': {'high': 126, 'critical': 200}},
                'confidence_base': 0.85
            },
            'anemia': {
                'keywords': ['anemia', 'hemoglobin', 'hematocrit', 'iron deficiency'],
                'lab_values': {'hemoglobin': {'low': 12}, 'hematocrit': {'low': 36}},
                'confidence_base': 0.80
            },
            'kidney_disease': {
                'keywords': ['creatinine', 'bun', 'kidney', 'renal'],
                'lab_values': {'creatinine': {'high': 1.2}, 'bun': {'high': 20}},
                'confidence_base': 0.75
            },
            'liver_disease': {
                'keywords': ['alt', 'ast', 'bilirubin', 'liver', 'hepatitis'],
                'lab_values': {'alt': {'high': 56}, 'ast': {'high': 40}, 'bilirubin': {'high': 1.2}},
                'confidence_base': 0.80
            },
            'infection': {
                'keywords': ['wbc', 'white blood cell', 'infection', 'bacterial'],
                'lab_values': {'wbc': {'high': 11}},
                'confidence_base': 0.70
            },
            'hyperlipidemia': {
                'keywords': ['cholesterol', 'ldl', 'hdl', 'lipid'],
                'lab_values': {'cholesterol': {'high': 200}, 'ldl': {'high': 100}},
                'confidence_base': 0.75
            }
        }
    
    def predict(self, text, metrics):
        """
        Predict disease based on text and lab metrics
        """
        text_lower = text.lower()
        predictions = []
        
        # Analyze text for disease indicators
        for disease, config in self.disease_patterns.items():
            confidence = 0.0
            indicators = []
            
            # Check for keywords in text
            keyword_matches = sum(1 for keyword in config['keywords'] if keyword in text_lower)
            if keyword_matches > 0:
                confidence += 0.3 * (keyword_matches / len(config['keywords']))
            
            # Check lab values
            for metric in metrics:
                metric_name = metric['name'].lower()
                metric_value = metric['value']
                
                if isinstance(metric_value, str):
                    try:
                        metric_value = float(metric_value)
                    except ValueError:
                        continue
                
                if metric_name in config['lab_values']:
                    lab_config = config['lab_values'][metric_name]
                    
                    if 'high' in lab_config and metric_value > lab_config['high']:
                        confidence += 0.4
                        indicators.append(f"Elevated {metric_name}")
                    elif 'low' in lab_config and metric_value < lab_config['low']:
                        confidence += 0.4
                        indicators.append(f"Low {metric_name}")
            
            if confidence > 0.3:  # Minimum threshold
                predictions.append({
                    'disease': self._format_disease_name(disease),
                    'confidence': min(confidence, 0.95),  # Cap at 95%
                    'indicators': indicators
                })
        
        # Sort by confidence
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return predictions
    
    def _format_disease_name(self, disease_key):
        """Format disease names for display"""
        disease_names = {
            'diabetes': 'Type 2 Diabetes Mellitus',
            'anemia': 'Iron Deficiency Anemia',
            'kidney_disease': 'Chronic Kidney Disease',
            'liver_disease': 'Hepatitis or Liver Disease',
            'infection': 'Bacterial Infection',
            'hyperlipidemia': 'Hyperlipidemia'
        }
        return disease_names.get(disease_key, disease_key.replace('_', ' ').title())

# Initialize the predictor
predictor = MockDiseasePredictor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Medical Report ML Service',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'models_available': MODELS_AVAILABLE
    })

@app.route('/predict', methods=['POST'])
def predict_disease():
    """
    Main prediction endpoint
    Expects JSON with: text, metrics, file_name
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['text', 'metrics']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        text = data['text']
        metrics = data['metrics']
        file_name = data.get('file_name', 'unknown')
        
        logger.info(f"Processing prediction request for file: {file_name}")
        logger.info(f"Text length: {len(text)} characters")
        logger.info(f"Metrics count: {len(metrics)}")
        
        # Validate metrics format
        if not isinstance(metrics, list):
            return jsonify({'error': 'Metrics must be a list'}), 400
        
        # Run prediction
        predictions = predictor.predict(text, metrics)
        
        if not predictions:
            # No specific disease detected
            response = {
                'predicted_disease': 'Normal Laboratory Values',
                'confidence': 0.60,
                'key_indicators': ['All values within normal ranges'],
                'analysis_summary': 'The laboratory values appear to be within normal ranges. However, this analysis is limited and should not replace professional medical evaluation.',
                'alternative_diagnoses': []
            }
        else:
            # Get top prediction
            top_prediction = predictions[0]
            alternative_diagnoses = [
                {'disease': p['disease'], 'confidence': p['confidence']}
                for p in predictions[1:3]  # Top 2 alternatives
            ]
            
            response = {
                'predicted_disease': top_prediction['disease'],
                'confidence': top_prediction['confidence'],
                'key_indicators': top_prediction['indicators'],
                'analysis_summary': f"Based on laboratory analysis, the most likely condition is {top_prediction['disease']} with {top_prediction['confidence']:.1%} confidence. Key indicators include: {', '.join(top_prediction['indicators'])}. Please consult with a healthcare provider for proper diagnosis and treatment.",
                'alternative_diagnoses': alternative_diagnoses
            }
        
        logger.info(f"Prediction completed: {response['predicted_disease']} ({response['confidence']:.1%})")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/analyze', methods=['POST'])
def analyze_with_ml():
    """
    New endpoint for ML-based analysis using trained models
    Expects JSON with: extracted_values, symptoms, model_type
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['extracted_values', 'symptoms']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        extracted_values = data['extracted_values']
        symptoms = data['symptoms']
        model_type = data.get('model_type', 'heart_disease')
        
        logger.info(f"Processing ML analysis request")
        logger.info(f"Model type: {model_type}")
        logger.info(f"Extracted values: {list(extracted_values.keys())}")
        
        # Use appropriate ML model
        if MODELS_AVAILABLE:
            if model_type == 'diabetes':
                result = diabetes_model.predict(extracted_values)
            elif model_type == 'mri':
                result = mri_model.predict(extracted_values)
            elif model_type == 'xray':
                result = xray_model.predict(extracted_values)
            else:
                # Default to diabetes model
                result = diabetes_model.predict(extracted_values)
        else:
            # Fallback to mock prediction
            result = {
                'predicted_disease': 'Analysis Unavailable',
                'confidence': 0.5,
                'model_used': 'fallback',
                'parameters_analyzed': []
            }
        
        logger.info(f"ML analysis completed: {result['predicted_disease']} ({result['confidence']:.1%})")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in ML analysis: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/models', methods=['GET'])
def list_models():
    """List available disease patterns and ML models"""
    return jsonify({
        'available_patterns': list(predictor.disease_patterns.keys()),
        'total_patterns': len(predictor.disease_patterns),
        'ml_models_available': MODELS_AVAILABLE,
        'ml_models': ['diabetes', 'mri', 'xray'] if MODELS_AVAILABLE else []
    })

@app.route('/analyze-metrics', methods=['POST'])
def analyze_metrics():
    """
    Analyze specific lab metrics without full text analysis
    """
    try:
        data = request.get_json()
        metrics = data.get('metrics', [])
        
        if not metrics:
            return jsonify({'error': 'No metrics provided'}), 400
        
        # Simple metric analysis
        abnormalities = []
        for metric in metrics:
            name = metric['name'].lower()
            value = metric['value']
            status = metric.get('status', 'unknown')
            
            if status in ['high', 'low', 'critical']:
                abnormalities.append({
                    'metric': name,
                    'value': value,
                    'status': status,
                    'concern': f"{name} is {status}"
                })
        
        return jsonify({
            'abnormalities': abnormalities,
            'total_metrics': len(metrics),
            'abnormal_count': len(abnormalities)
        })
        
    except Exception as e:
        logger.error(f"Error in metric analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Medical Report ML Service...")
    logger.info("Available endpoints:")
    logger.info("  GET  /health - Health check")
    logger.info("  POST /predict - Disease prediction")
    logger.info("  POST /analyze - ML-based analysis")
    logger.info("  GET  /models - List available patterns")
    logger.info("  POST /analyze-metrics - Analyze lab metrics")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )