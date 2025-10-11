/**
 * Medical Reports Analysis API Route
 * Handles file uploads and processes medical reports for disease prediction
 */

import { NextRequest, NextResponse } from 'next/server';
import formidable from 'formidable';
import fs from 'fs';
import path from 'path';
import { parsePDF } from '@/services/reportParser';
import { predictDisease } from '@/services/mlService';

// Define interfaces for type safety
interface ReportAnalysisRequest {
  file?: File;
}

interface ReportAnalysisResponse {
  predicted_disease: string;
  confidence: number;
  key_indicators: string[];
  extracted_metrics: Record<string, any>;
  analysis_summary: string;
}

interface ErrorResponse {
  error: string;
  details?: string;
}

export async function POST(req: NextRequest): Promise<NextResponse<ReportAnalysisResponse | ErrorResponse>> {
  console.log("\n--- /api/reports endpoint was called ---");

  try {
    // Parse multipart form data
    const formData = await req.formData();
    const file = formData.get('file') as File;

    // Validate file upload
    if (!file) {
      return NextResponse.json(
        { error: "No file uploaded. Please select a medical report file." },
        { status: 400 }
      );
    }

    // Validate file type
    const allowedTypes = ['application/pdf', 'image/jpeg', 'image/png', 'image/jpg'];
    if (!allowedTypes.includes(file.type)) {
      return NextResponse.json(
        { 
          error: "Invalid file type. Please upload a PDF or image file (JPEG, PNG).",
          details: `Received file type: ${file.type}`
        },
        { status: 400 }
      );
    }

    // Validate file size (max 10MB)
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
      return NextResponse.json(
        { 
          error: "File too large. Please upload a file smaller than 10MB.",
          details: `File size: ${(file.size / 1024 / 1024).toFixed(2)}MB`
        },
        { status: 400 }
      );
    }

    console.log(`📄 Processing file: ${file.name} (${file.type}, ${(file.size / 1024).toFixed(2)}KB)`);

    // Convert file to buffer for processing
    const fileBuffer = Buffer.from(await file.arrayBuffer());

    // Extract text and metrics from the report
    console.log("🔍 Extracting text and metrics from report...");
    const extractedData = await parsePDF(fileBuffer, file.type);

    if (!extractedData.text || extractedData.text.trim().length === 0) {
      return NextResponse.json(
        { 
          error: "Unable to extract text from the uploaded file.",
          details: "The file may be corrupted, password-protected, or contain only images without OCR."
        },
        { status: 422 }
      );
    }

    console.log(`📊 Extracted ${extractedData.metrics.length} health metrics`);
    console.log(`📝 Text length: ${extractedData.text.length} characters`);

    // Send data to ML model for disease prediction
    console.log("🤖 Sending data to ML model for analysis...");
    const mlPrediction = await predictDisease({
      text: extractedData.text,
      metrics: extractedData.metrics,
      fileName: file.name
    });

    // Prepare response
    const response: ReportAnalysisResponse = {
      predicted_disease: mlPrediction.predicted_disease,
      confidence: mlPrediction.confidence,
      key_indicators: mlPrediction.key_indicators,
      extracted_metrics: extractedData.metrics.reduce((acc, metric) => {
        acc[metric.name] = {
          value: metric.value,
          unit: metric.unit,
          status: metric.status,
          reference_range: metric.reference_range
        };
        return acc;
      }, {} as Record<string, any>),
      analysis_summary: mlPrediction.analysis_summary
    };

    console.log(`✅ Analysis complete: ${response.predicted_disease} (${(response.confidence * 100).toFixed(1)}% confidence)`);

    return NextResponse.json(response);

  } catch (error) {
    console.error("🔴 Error in /api/reports:", error);

    let errorMessage = "An error occurred while processing your medical report.";
    let statusCode = 500;

    if (error instanceof Error) {
      if (error.message.includes("file") || error.message.includes("upload")) {
        errorMessage = "File processing error. Please ensure your file is readable and try again.";
        statusCode = 422;
      } else if (error.message.includes("ML") || error.message.includes("prediction")) {
        errorMessage = "Analysis service temporarily unavailable. Please try again later.";
        statusCode = 503;
      } else {
        errorMessage = error.message;
      }
    }

    return NextResponse.json(
      { 
        error: errorMessage,
        details: error instanceof Error ? error.message : "Unknown error"
      },
      { status: statusCode }
    );
  }
}

// Health check endpoint for the reports service
export async function GET(): Promise<NextResponse<{ status: string; service: string }>> {
  return NextResponse.json({
    status: "healthy",
    service: "Medical Reports Analysis API"
  });
}
