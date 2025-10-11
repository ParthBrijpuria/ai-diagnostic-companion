# Medical Report ML Service

This Python Flask service provides AI-powered analysis of medical reports and lab results for disease prediction.

## Features

- **Disease Prediction**: Analyzes medical text and lab values to predict possible conditions
- **Lab Value Extraction**: Parses common laboratory test results from medical reports
- **Confidence Scoring**: Provides confidence levels for predictions
- **Multiple Disease Support**: Recognizes patterns for diabetes, anemia, kidney disease, liver disease, infections, and more
- **RESTful API**: Clean JSON API for integration with frontend applications

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Navigate to the ML directory:**
   ```bash
   cd ml
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment:**
   
   **On Windows:**
   ```bash
   venv\Scripts\activate
   ```
   
   **On macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Start the service:**
   ```bash
   python diseasePredictor.py
   ```

The service will start on `http://localhost:5000`

### Using the Startup Scripts

**Windows:**
```bash
ml\start_ml_service.bat
```

**macOS/Linux:**
```bash
chmod +x ml/start_ml_service.sh
./ml/start_ml_service.sh
```

## API Endpoints

### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "Medical Report ML Service",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0"
}
```

### Disease Prediction
```http
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "text": "Patient presents with fatigue and weakness. Lab results show hemoglobin 8.5 g/dL, hematocrit 25%...",
  "metrics": [
    {
      "name": "HEMOGLOBIN",
      "value": 8.5,
      "unit": "g/dL",
      "status": "low",
      "reference_range": "12-16 g/dL"
    }
  ],
  "file_name": "lab_report.pdf"
}
```

**Response:**
```json
{
  "predicted_disease": "Iron Deficiency Anemia",
  "confidence": 0.85,
  "key_indicators": ["Low hemoglobin", "Reduced oxygen-carrying capacity"],
  "analysis_summary": "Based on laboratory analysis, the most likely condition is Iron Deficiency Anemia with 85.0% confidence...",
  "alternative_diagnoses": [
    {
      "disease": "Chronic Anemia",
      "confidence": 0.70
    }
  ]
}
```

### List Available Patterns
```http
GET /models
```

**Response:**
```json
{
  "available_patterns": ["diabetes", "anemia", "kidney_disease", "liver_disease", "infection", "hyperlipidemia"],
  "total_patterns": 6
}
```

### Analyze Metrics
```http
POST /analyze
Content-Type: application/json
```

**Request Body:**
```json
{
  "metrics": [
    {
      "name": "glucose",
      "value": 150,
      "status": "high"
    }
  ]
}
```

## Supported Disease Patterns

The service recognizes patterns for the following conditions:

1. **Diabetes** - Elevated glucose levels
2. **Anemia** - Low hemoglobin/hematocrit
3. **Kidney Disease** - Elevated creatinine/BUN
4. **Liver Disease** - Elevated ALT/AST/bilirubin
5. **Infection** - Elevated white blood cells
6. **Hyperlipidemia** - Elevated cholesterol/LDL

## Lab Value Recognition

The service automatically extracts and analyzes common lab values:

- **Blood Tests**: Hemoglobin, Hematocrit, WBC, RBC, Platelets
- **Chemistry**: Glucose, Creatinine, BUN, Sodium, Potassium
- **Lipid Panel**: Total Cholesterol, HDL, LDL
- **Liver Function**: ALT, AST, Bilirubin

## Integration with Frontend

The service is designed to work seamlessly with the Next.js frontend:

1. **File Upload**: Frontend uploads medical reports via `/api/reports`
2. **Text Extraction**: PDF/image text is extracted using `pdf-parse`
3. **ML Analysis**: Extracted data is sent to this Python service
4. **Results Display**: Predictions are returned and displayed in the UI

## Development

### Adding New Disease Patterns

To add support for new diseases, update the `disease_patterns` dictionary in `diseasePredictor.py`:

```python
'new_disease': {
    'keywords': ['keyword1', 'keyword2'],
    'lab_values': {'lab_test': {'high': 100, 'low': 50}},
    'confidence_base': 0.75
}
```

### Testing

Test the service using curl:

```bash
# Health check
curl http://localhost:5000/health

# Disease prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient has high glucose", "metrics": [{"name": "glucose", "value": 150, "unit": "mg/dL", "status": "high"}]}'
```

## Production Deployment

For production deployment:

1. **Use a production WSGI server** like Gunicorn:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 diseasePredictor:app
   ```

2. **Set environment variables**:
   ```bash
   export FLASK_ENV=production
   export FLASK_DEBUG=False
   ```

3. **Use a reverse proxy** like Nginx for better performance

## Troubleshooting

### Common Issues

1. **Port 5000 already in use:**
   ```bash
   # Find process using port 5000
   lsof -i :5000
   # Kill the process
   kill -9 <PID>
   ```

2. **Python dependencies not found:**
   ```bash
   # Ensure virtual environment is activated
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate    # Windows
   ```

3. **CORS errors:**
   - Ensure Flask-CORS is installed
   - Check that the frontend URL is allowed

### Logs

The service logs important events to the console. Check for:
- Successful prediction requests
- Error messages
- Health check responses

## License

This service is part of the AI Diagnostic Companion project and follows the same license terms.
