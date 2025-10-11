/**
 * Diabetes Analysis API Route
 * Handles diabetes parameter analysis using local RandomForest model
 */

import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';

// Define interfaces
interface DiabetesAnalysisRequest {
  manualValues: {
    Gender: string;
    Pregnancies: string;
    Glucose: string;
    BloodPressure: string;
    SkinThickness: string;
    Insulin: string;
    BMI: string;
    DiabetesPedigreeFunction: string;
    Age: string;
  };
}

interface DiabetesAnalysisResponse {
  predicted_class: string;
  confidence: number;
  analysis_type: 'diabetes_parameters';
}

export async function POST(req: NextRequest): Promise<NextResponse<DiabetesAnalysisResponse | { error: string }>> {
  console.log("\n--- /api/diabetes-analysis endpoint was called ---");

  try {
    const body = await req.json();
    const { manualValues } = body as DiabetesAnalysisRequest;

    if (!manualValues) {
      return NextResponse.json(
        { error: "No diabetes parameters provided." },
        { status: 400 }
      );
    }

    // Validate required fields
    const requiredFields = ['Glucose', 'BloodPressure', 'BMI', 'Age'];
    const missingFields = requiredFields.filter(field => !manualValues[field as keyof typeof manualValues]);
    
    if (missingFields.length > 0) {
      return NextResponse.json(
        { error: `Missing required fields: ${missingFields.join(', ')}` },
        { status: 400 }
      );
    }

    console.log(`🔍 Processing diabetes parameters for ${manualValues.Gender}`);

    // Call Python diabetes model
    const prediction = await analyzeDiabetesParameters(manualValues);

    const response: DiabetesAnalysisResponse = {
      predicted_class: prediction.predicted_class,
      confidence: prediction.confidence,
      analysis_type: 'diabetes_parameters'
    };

    console.log(`✅ Diabetes analysis complete: ${prediction.predicted_class} (${(prediction.confidence * 100).toFixed(1)}%)`);

    return NextResponse.json(response);

  } catch (error) {
    console.error("🔴 Error in /api/diabetes-analysis:", error);

    let errorMessage = "An error occurred while analyzing your diabetes parameters.";
    let statusCode = 500;

    if (error instanceof Error) {
      if (error.message.includes("parameters") || error.message.includes("validation")) {
        errorMessage = "Invalid parameters provided. Please check your input values.";
        statusCode = 422;
      } else if (error.message.includes("model") || error.message.includes("prediction")) {
        errorMessage = "Diabetes analysis service temporarily unavailable. Please try again later.";
        statusCode = 503;
      } else {
        errorMessage = error.message;
      }
    }

    return NextResponse.json(
      { error: errorMessage },
      { status: statusCode }
    );
  }
}

/**
 * Analyze diabetes parameters using Python RandomForest model
 */
async function analyzeDiabetesParameters(manualValues: any): Promise<{ predicted_class: string; confidence: number }> {
  return new Promise((resolve, reject) => {
    console.log(`🐍 Calling Python diabetes model for parameters`);

    // Create Python script to analyze the parameters
    const pythonScript = `
import sys
import os
sys.path.append('.')

try:
    from models.diabetes.diabetes_classifier import DiabetesClassifier
    
    # Initialize and load the model
    classifier = DiabetesClassifier()
    classifier.load_model('models/diabetes/diabetes_model.pkl')
    
    # Prepare input data
    input_data = {
        'Pregnancies': ${manualValues.Pregnancies || 0},
        'Glucose': ${manualValues.Glucose || 0},
        'BloodPressure': ${manualValues.BloodPressure || 0},
        'SkinThickness': ${manualValues.SkinThickness || 0},
        'Insulin': ${manualValues.Insulin || 0},
        'BMI': ${manualValues.BMI || 0},
        'DiabetesPedigreeFunction': ${manualValues.DiabetesPedigreeFunction || 0},
        'Age': ${manualValues.Age || 0}
    }
    
    # Make prediction
    result = classifier.predict(input_data)
    
    import json
    print(json.dumps(result))
    
except Exception as e:
    print(f"ERROR: {str(e)}", file=sys.stderr)
    sys.exit(1)
`;

    // Write Python script to temporary file
    const scriptPath = path.join(process.cwd(), 'temp', `diabetes_analysis_${Date.now()}.py`);
    fs.writeFileSync(scriptPath, pythonScript);

    // Execute Python script
    const pythonProcess = spawn('python', [scriptPath], {
      cwd: process.cwd(),
      stdio: ['pipe', 'pipe', 'pipe']
    });

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    pythonProcess.on('close', (code) => {
      // Clean up script file
      try {
        fs.unlinkSync(scriptPath);
      } catch (cleanupError) {
        console.warn(`⚠️ Failed to delete script file: ${cleanupError}`);
      }

      if (code === 0) {
        const predictionData = JSON.parse(stdout.trim());
        if (predictionData && predictionData.predicted_class && predictionData.confidence) {
          console.log(`✅ Python model prediction: ${predictionData.predicted_class} (${(predictionData.confidence * 100).toFixed(1)}%)`);
          resolve(predictionData);
        } else {
          console.error(`🔴 Invalid prediction from Python model: ${stdout}`);
          reject(new Error(`Invalid prediction: ${stdout}`));
        }
      } else {
        console.error(`🔴 Python script failed with code ${code}: ${stderr}`);
        reject(new Error(`Diabetes analysis failed: ${stderr}`));
      }
    });

    pythonProcess.on('error', (error) => {
      console.error(`🔴 Failed to start Python process: ${error}`);
      reject(new Error(`Failed to start diabetes analysis: ${error.message}`));
    });

    // Set timeout
    setTimeout(() => {
      pythonProcess.kill();
      reject(new Error('Diabetes analysis timeout'));
    }, 30000); // 30 second timeout
  });
}

// Health check endpoint
export async function GET(): Promise<NextResponse<{ status: string; service: string }>> {
  return NextResponse.json({
    status: "healthy",
    service: "Diabetes Analysis API"
  });
}
