/**
 * ML Service for Disease Prediction
 * Communicates with Python ML model for medical report analysis
 */

import axios from 'axios';

// Define interfaces for ML communication
export interface MLInput {
  text: string;
  metrics: Array<{
    name: string;
    value: number | string;
    unit: string;
    status: string;
    reference_range?: string;
  }>;
  fileName: string;
}

export interface MLPrediction {
  predicted_disease: string;
  confidence: number;
  key_indicators: string[];
  analysis_summary: string;
  alternative_diagnoses?: Array<{
    disease: string;
    confidence: number;
  }>;
}

// Configuration for ML service
const ML_SERVICE_CONFIG = {
  baseUrl: process.env.ML_SERVICE_URL || 'http://localhost:5000',
  timeout: 30000, // 30 seconds
  retries: 3
};

/**
 * Send data to Python ML model for disease prediction
 */
export async function predictDisease(input: MLInput): Promise<MLPrediction> {
  console.log(`🤖 Sending data to ML service at ${ML_SERVICE_CONFIG.baseUrl}`);
  
  try {
    // Prepare data for ML model
    const mlInput = {
      text: input.text,
      metrics: input.metrics,
      file_name: input.fileName,
      timestamp: new Date().toISOString()
    };

    // Try to connect to Python ML service
    const response = await axios.post(
      `${ML_SERVICE_CONFIG.baseUrl}/predict`,
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
    return generateFallbackPrediction(input);
  }
}

/**
 * Generate fallback prediction using rule-based logic
 * This is used when the Python ML service is unavailable
 */
function generateFallbackPrediction(input: MLInput): MLPrediction {
  console.log("🔄 Using fallback rule-based prediction");
  
  const metrics = input.metrics;
  // const text = input.text.toLowerCase();
  
  // Rule-based disease prediction based on lab values
  const predictions: Array<{ disease: string; confidence: number; indicators: string[] }> = [];
  
  // Anemia detection
  const hemoglobin = metrics.find(m => m.name === 'HEMOGLOBIN');
  const hematocrit = metrics.find(m => m.name === 'HEMATOCRIT');
  // const rbc = metrics.find(m => m.name === 'RBC');
  
  if (hemoglobin && typeof hemoglobin.value === 'number' && hemoglobin.value < 12) {
    predictions.push({
      disease: 'Iron Deficiency Anemia',
      confidence: 0.85,
      indicators: ['Low hemoglobin', 'Reduced oxygen-carrying capacity']
    });
  }
  
  if (hematocrit && typeof hematocrit.value === 'number' && hematocrit.value < 36) {
    predictions.push({
      disease: 'Anemia',
      confidence: 0.80,
      indicators: ['Low hematocrit', 'Reduced blood volume']
    });
  }
  
  // Diabetes detection
  const glucose = metrics.find(m => m.name === 'GLUCOSE');
  if (glucose && typeof glucose.value === 'number' && glucose.value > 126) {
    predictions.push({
      disease: 'Type 2 Diabetes Mellitus',
      confidence: 0.90,
      indicators: ['Elevated blood glucose', 'Hyperglycemia']
    });
  }
  
  // Kidney disease detection
  const creatinine = metrics.find(m => m.name === 'CREATININE');
  const bun = metrics.find(m => m.name === 'BUN');
  
  if (creatinine && typeof creatinine.value === 'number' && creatinine.value > 1.2) {
    predictions.push({
      disease: 'Chronic Kidney Disease',
      confidence: 0.75,
      indicators: ['Elevated creatinine', 'Reduced kidney function']
    });
  }
  
  if (bun && typeof bun.value === 'number' && bun.value > 20) {
    predictions.push({
      disease: 'Kidney Dysfunction',
      confidence: 0.70,
      indicators: ['Elevated BUN', 'Nitrogen waste accumulation']
    });
  }
  
  // Liver disease detection
  const alt = metrics.find(m => m.name === 'ALT');
  const ast = metrics.find(m => m.name === 'AST');
  const bilirubin = metrics.find(m => m.name === 'BILIRUBIN');
  
  if (alt && typeof alt.value === 'number' && alt.value > 56) {
    predictions.push({
      disease: 'Hepatitis or Liver Disease',
      confidence: 0.80,
      indicators: ['Elevated ALT', 'Liver enzyme elevation']
    });
  }
  
  if (ast && typeof ast.value === 'number' && ast.value > 40) {
    predictions.push({
      disease: 'Liver Dysfunction',
      confidence: 0.75,
      indicators: ['Elevated AST', 'Hepatic enzyme elevation']
    });
  }
  
  if (bilirubin && typeof bilirubin.value === 'number' && bilirubin.value > 1.2) {
    predictions.push({
      disease: 'Jaundice or Liver Disease',
      confidence: 0.85,
      indicators: ['Elevated bilirubin', 'Bile pigment accumulation']
    });
  }
  
  // Infection detection
  const wbc = metrics.find(m => m.name === 'WBC');
  if (wbc && typeof wbc.value === 'number' && wbc.value > 11) {
    predictions.push({
      disease: 'Bacterial Infection',
      confidence: 0.70,
      indicators: ['Elevated white blood cells', 'Immune system response']
    });
  }
  
  // Hyperlipidemia detection
  const cholesterol = metrics.find(m => m.name === 'CHOLESTEROL');
  const ldl = metrics.find(m => m.name === 'LDL');
  
  if (cholesterol && typeof cholesterol.value === 'number' && cholesterol.value > 200) {
    predictions.push({
      disease: 'Hyperlipidemia',
      confidence: 0.80,
      indicators: ['Elevated total cholesterol', 'Cardiovascular risk']
    });
  }
  
  if (ldl && typeof ldl.value === 'number' && ldl.value > 100) {
    predictions.push({
      disease: 'High LDL Cholesterol',
      confidence: 0.75,
      indicators: ['Elevated LDL cholesterol', 'Atherosclerosis risk']
    });
  }
  
  // Electrolyte imbalance detection
  const sodium = metrics.find(m => m.name === 'SODIUM');
  const potassium = metrics.find(m => m.name === 'POTASSIUM');
  
  if (sodium && typeof sodium.value === 'number' && sodium.value < 136) {
    predictions.push({
      disease: 'Hyponatremia',
      confidence: 0.85,
      indicators: ['Low sodium levels', 'Electrolyte imbalance']
    });
  }
  
  if (potassium && typeof potassium.value === 'number' && potassium.value < 3.5) {
    predictions.push({
      disease: 'Hypokalemia',
      confidence: 0.80,
      indicators: ['Low potassium levels', 'Muscle weakness risk']
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
      key_indicators: topPrediction.indicators,
      analysis_summary: `Based on laboratory values, the most likely condition is ${topPrediction.disease}. This assessment is based on ${topPrediction.indicators.join(' and ')}. Please consult with a healthcare provider for proper diagnosis and treatment.`,
      alternative_diagnoses: predictions.slice(1, 3).map(p => ({
        disease: p.disease,
        confidence: p.confidence
      }))
    };
  } else {
    // No specific abnormalities detected
    return {
      predicted_disease: 'Normal Laboratory Values',
      confidence: 0.60,
      key_indicators: ['All values within normal ranges'],
      analysis_summary: 'The laboratory values appear to be within normal ranges. However, this analysis is limited and should not replace professional medical evaluation. Please consult with a healthcare provider for comprehensive assessment.'
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
