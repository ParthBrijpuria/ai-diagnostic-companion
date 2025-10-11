/**
 * X-Ray Analysis API Route
 * Handles X-ray image uploads and analysis using local GradientBoosting model
 */

import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';

// Define interfaces
// interface XRayAnalysisRequest {
//   file: File;
// }

interface XRayAnalysisResponse {
  predicted_class: string;
  confidence: number;
  analysis_type: 'xray_image';
}

export async function POST(req: NextRequest): Promise<NextResponse<XRayAnalysisResponse | { error: string }>> {
  console.log("\n--- /api/xray-analysis endpoint was called ---");

  try {
    const formData = await req.formData();
    const file = formData.get('file') as File;

    if (!file) {
      return NextResponse.json(
        { error: "No file uploaded. Please select an X-ray image file." },
        { status: 400 }
      );
    }

    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/png', 'image/jpg'];
    const fileName = file.name.toLowerCase();
    const hasValidExtension = fileName.endsWith('.jpg') || fileName.endsWith('.jpeg') || fileName.endsWith('.png');
    
    if (!allowedTypes.includes(file.type) && !hasValidExtension) {
      return NextResponse.json(
        { error: "Invalid file type. Please upload a JPEG or PNG image file." },
        { status: 400 }
      );
    }

    // Validate file size (10MB max)
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
      return NextResponse.json(
        { error: "File too large. Please upload a file smaller than 10MB." },
        { status: 400 }
      );
    }

    console.log(`🔍 Processing X-ray image: ${file.name}`);

    // Create temporary file
    const tempDir = path.join(process.cwd(), 'temp');
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir, { recursive: true });
    }

    const tempFilePath = path.join(tempDir, `xray_${Date.now()}_${file.name}`);
    const fileBuffer = Buffer.from(await file.arrayBuffer());
    fs.writeFileSync(tempFilePath, fileBuffer);

    console.log(`📁 Temporary file created: ${tempFilePath}`);

    // Call Python X-ray model
    const prediction = await analyzeXRayImage(tempFilePath);

    // Clean up temporary file
    try {
      fs.unlinkSync(tempFilePath);
      console.log(`🗑️ Temporary file deleted: ${tempFilePath}`);
    } catch (cleanupError) {
      console.warn(`⚠️ Failed to delete temporary file: ${cleanupError}`);
    }

    const response: XRayAnalysisResponse = {
      predicted_class: prediction.predicted_class,
      confidence: prediction.confidence,
      analysis_type: 'xray_image'
    };

    console.log(`✅ X-ray analysis complete: ${prediction.predicted_class} (${(prediction.confidence * 100).toFixed(1)}%)`);

    return NextResponse.json(response);

  } catch (error) {
    console.error("🔴 Error in /api/xray-analysis:", error);

    let errorMessage = "An error occurred while analyzing your X-ray image.";
    let statusCode = 500;

    if (error instanceof Error) {
      if (error.message.includes("file") || error.message.includes("upload")) {
        errorMessage = "File processing error. Please ensure your image is readable and try again.";
        statusCode = 422;
      } else if (error.message.includes("model") || error.message.includes("prediction")) {
        errorMessage = "X-ray analysis service temporarily unavailable. Please try again later.";
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
 * Analyze X-ray image using Python GradientBoosting model
 */
async function analyzeXRayImage(imagePath: string): Promise<{ predicted_class: string; confidence: number }> {
  return new Promise((resolve, reject) => {
    console.log(`🐍 Calling Python X-ray model for: ${imagePath}`);

    // Create Python script to analyze the image
    const pythonScript = `
import sys
import os
sys.path.append('.')

try:
    from models.xray.xray_pneumonia_classifier import XRayPneumoniaClassifier
    
    # Initialize and load the model
    classifier = XRayPneumoniaClassifier()
    classifier.load_model('models/xray/xray_pneumonia_model.pkl')
    
    # Make prediction
    result = classifier.predict('${imagePath.replace(/\\/g, '\\\\')}')
    
    import json
    print(json.dumps(result))
    
except Exception as e:
    print(f"ERROR: {str(e)}", file=sys.stderr)
    sys.exit(1)
`;

    // Write Python script to temporary file
    const scriptPath = path.join(process.cwd(), 'temp', `xray_analysis_${Date.now()}.py`);
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
        reject(new Error(`X-ray analysis failed: ${stderr}`));
      }
    });

    pythonProcess.on('error', (error) => {
      console.error(`🔴 Failed to start Python process: ${error}`);
      reject(new Error(`Failed to start X-ray analysis: ${error.message}`));
    });

    // Set timeout
    setTimeout(() => {
      pythonProcess.kill();
      reject(new Error('X-ray analysis timeout'));
    }, 30000); // 30 second timeout
  });
}

// Health check endpoint
export async function GET(): Promise<NextResponse<{ status: string; service: string }>> {
  return NextResponse.json({
    status: "healthy",
    service: "X-Ray Analysis API"
  });
}
