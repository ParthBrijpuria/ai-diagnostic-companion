/**
 * ML Analysis Service
 * Handles disease prediction using trained ML models
 */

import axios from 'axios';

// Define interfaces
export interface MLInput {
  extracted_values: Record<string, any>;
  symptoms: string;
  model_type: string;
  image_data?: any; // Added for image processing
}

export interface MLPrediction {
  predicted_disease: string;
  confidence: number;
  model_used: string;
  parameters_analyzed: string[];
  error?: string; // Added for error handling
}

// Configuration for ML service
const ML_SERVICE_CONFIG = {
  baseUrl: process.env.ML_SERVICE_URL || 'http://localhost:5000',
  timeout: 30000,
  retries: 3
};

/**
 * Analyze test values using ML models
 */
export async function analyzeWithML(extractedValues: Record<string, any>, symptoms: string, modelType: string): Promise<MLPrediction> {
  console.log(`🤖 Sending data to ML service for analysis`);
  
  try {
    // Determine which ML model to use based on symptoms and available values
    const finalModelType = modelType || determineModelType(extractedValues, symptoms);
    
    // Prepare data for ML model
    const mlInput = {
      extracted_values: extractedValues,
      symptoms: symptoms.toLowerCase(),
      model_type: finalModelType,
      timestamp: new Date().toISOString()
    };

    // Try to connect to Python ML service
    const response = await axios.post(
      `${ML_SERVICE_CONFIG.baseUrl}/analyze`,
      mlInput,
      {
        timeout: ML_SERVICE_CONFIG.timeout,
        headers: {
          'Content-Type': 'application/json',
          'User-Agent': 'AI-Diagnostic-Companion/1.0'
        }
      }
    );

    if (response.status === 200 && response.data) {
      console.log(`✅ ML prediction received: ${response.data.predicted_disease}`);
      return response.data as MLPrediction;
    } else {
      throw new Error(`ML service returned invalid response: ${response.status}`);
    }

  } catch (error) {
    console.warn(`⚠️ ML service unavailable, using fallback prediction:`, error);
    
    // Fallback to rule-based prediction when ML service is unavailable
    return generateFallbackPrediction(extractedValues, symptoms, modelType);
  }
}

/**
 * Determine which ML model to use based on available data
 */
function determineModelType(extractedValues: Record<string, any>, symptoms: string): string {
  const symptomsLower = symptoms.toLowerCase();
  
  // Check for diabetes-related symptoms and values
  if (symptomsLower.includes('diabetes') || 
      symptomsLower.includes('blood sugar') || 
      symptomsLower.includes('glucose') ||
      extractedValues.blood_sugar ||
      extractedValues.bmi ||
      extractedValues.Pregnancies ||
      extractedValues.Glucose ||
      extractedValues.BloodPressure) {
    return 'diabetes';
  }
  
  // Check for MRI-related symptoms
  if (symptomsLower.includes('brain') || 
      symptomsLower.includes('neurological') || 
      symptomsLower.includes('headache') ||
      symptomsLower.includes('seizure') ||
      symptomsLower.includes('tumor')) {
    return 'mri';
  }
  
  // Check for X-ray related symptoms
  if (symptomsLower.includes('lung') || 
      symptomsLower.includes('chest') || 
      symptomsLower.includes('respiratory') ||
      symptomsLower.includes('breathing') ||
      symptomsLower.includes('pneumonia')) {
    return 'xray';
  }
  
  // Default to diabetes model if no specific indicators
  return 'diabetes';
}

/**
 * Generate fallback prediction using rule-based logic
 */
function generateFallbackPrediction(extractedValues: Record<string, any>, symptoms: string, modelType: string): MLPrediction {
  console.log("🔄 Using fallback rule-based prediction");
  
  const symptomsLower = symptoms.toLowerCase();
  const predictions: Array<{ disease: string; confidence: number; parameters: string[] }> = [];
  
  // Diabetes Detection
  if (modelType === 'diabetes' || 
      extractedValues.blood_sugar || extractedValues.bmi || 
      extractedValues.Glucose || extractedValues.BMI ||
      symptomsLower.includes('diabetes') || symptomsLower.includes('glucose')) {
    
    let confidence = 0.6;
    const parameters: string[] = [];
    
    if (extractedValues.blood_sugar || extractedValues.Glucose) {
      parameters.push('glucose_level');
      confidence += 0.2;
    }
    if (extractedValues.bmi || extractedValues.BMI) {
      parameters.push('bmi');
      confidence += 0.1;
    }
    if (extractedValues.Pregnancies) {
      parameters.push('pregnancies');
      confidence += 0.1;
    }
    if (symptomsLower.includes('diabetes')) {
      parameters.push('diabetes_symptoms');
      confidence += 0.1;
    }
    
    predictions.push({
      disease: 'Type 2 Diabetes Mellitus',
      confidence: Math.min(confidence, 0.9),
      parameters
    });
  }
  
  // MRI Detection
  if (modelType === 'mri' || 
      symptomsLower.includes('brain') || 
      symptomsLower.includes('headache') ||
      symptomsLower.includes('seizure') ||
      symptomsLower.includes('tumor')) {
    
    let confidence = 0.7;
    const parameters: string[] = [];
    
    if (symptomsLower.includes('headache')) {
      parameters.push('headache');
      confidence += 0.1;
    }
    if (symptomsLower.includes('seizure')) {
      parameters.push('seizure');
      confidence += 0.1;
    }
    if (symptomsLower.includes('tumor')) {
      parameters.push('tumor_symptoms');
      confidence += 0.1;
    }
    
    predictions.push({
      disease: 'Brain Tumor Assessment Required',
      confidence: Math.min(confidence, 0.9),
      parameters
    });
  }
  
  // X-Ray Detection
  if (modelType === 'xray' || 
      symptomsLower.includes('lung') || 
      symptomsLower.includes('chest') ||
      symptomsLower.includes('breathing') ||
      symptomsLower.includes('pneumonia')) {
    
    let confidence = 0.7;
    const parameters: string[] = [];
    
    if (symptomsLower.includes('cough')) {
      parameters.push('cough');
      confidence += 0.1;
    }
    if (symptomsLower.includes('chest pain')) {
      parameters.push('chest_pain');
      confidence += 0.1;
    }
    if (symptomsLower.includes('breathing')) {
      parameters.push('breathing_difficulty');
      confidence += 0.1;
    }
    
    predictions.push({
      disease: 'Respiratory Condition Assessment Required',
      confidence: Math.min(confidence, 0.9),
      parameters
    });
  }
  
  // Sort predictions by confidence
  predictions.sort((a, b) => b.confidence - a.confidence);
  
  // Return the top prediction or a default
  if (predictions.length > 0) {
    const topPrediction = predictions[0];
    return {
      predicted_disease: topPrediction.disease,
      confidence: topPrediction.confidence,
      model_used: `fallback_${modelType}`,
      parameters_analyzed: topPrediction.parameters
    };
  } else {
    // No specific abnormalities detected
    return {
      predicted_disease: 'General Health Assessment Required',
      confidence: 0.5,
      model_used: `fallback_${modelType}`,
      parameters_analyzed: Object.keys(extractedValues)
    };
  }
}

/**
 * Health check for ML service
 */
export async function checkMLServiceHealth(): Promise<boolean> {
  try {
    const response = await axios.get(`${ML_SERVICE_CONFIG.baseUrl}/health`, {
      timeout: 5000
    });
    return response.status === 200;
  } catch (error) {
    console.warn('ML service health check failed:', error);
    return false;
  }
}
