/**
 * Test Analysis API Route
 * Handles test report uploads and manual value analysis for disease prediction
 */

import { NextRequest, NextResponse } from 'next/server';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { parsePDF } from '@/services/reportParser';
import { analyzeWithML } from '@/services/mlAnalysis';

// Define interfaces
interface TestAnalysisRequest {
  file?: File;
  manualValues?: {
    bloodPressure: string;
    bloodSugar: string;
    cholesterol: string;
    bmi: string;
  };
  symptoms: string;
  duration: string;
  modelType: string;
}

interface TestAnalysisResponse {
  disease_name: string;
  confidence: number;
  future_steps: string;
  top_medicines: Array<{
    name: string;
    description: string;
  }>;
  lifestyle_changes: string;
  analysis_type: 'file' | 'manual';
  extracted_values?: Record<string, string | number | { value: string; unit: string; status: string }>;
}

export async function POST(req: NextRequest): Promise<NextResponse<TestAnalysisResponse | { error: string }>> {
  console.log("\n--- /api/test-analysis endpoint was called ---");

  try {
    const contentType = req.headers.get('content-type');
    let analysisData: TestAnalysisRequest;

    if (contentType?.includes('multipart/form-data')) {
      // Handle file upload
      const formData = await req.formData();
      const file = formData.get('file') as File;
      const symptoms = formData.get('symptoms') as string;
      const duration = formData.get('duration') as string;
      const modelType = formData.get('modelType') as string;

      if (!file) {
        return NextResponse.json(
          { error: "No file uploaded. Please select a test report file." },
          { status: 400 }
        );
      }

      // Validate file type
      const allowedTypes = ['application/pdf', 'image/jpeg', 'image/png', 'image/jpg'];
      if (!allowedTypes.includes(file.type)) {
        return NextResponse.json(
          { error: "Invalid file type. Please upload a PDF or image file." },
          { status: 400 }
        );
      }

      analysisData = {
        file,
        symptoms: symptoms || '',
        duration: duration || '',
        modelType: modelType || 'heart_disease'
      };
    } else {
      // Handle manual values
      const body = await req.json();
      analysisData = {
        manualValues: body.manualValues,
        symptoms: body.symptoms || '',
        duration: body.duration || '',
        modelType: body.modelType || 'heart_disease'
      };
    }

    console.log(`🔍 Processing ${analysisData.file ? 'file upload' : 'manual values'} analysis`);

    let extractedValues: Record<string, string | number | { value: string; unit: string; status: string }> = {};
    let analysisType: 'file' | 'manual' = 'manual';

    if (analysisData.file) {
      try {
        // Extract text and values from uploaded file
        console.log(`📄 Processing file: ${analysisData.file.name}, type: ${analysisData.file.type}`);
        const fileBuffer = Buffer.from(await analysisData.file.arrayBuffer());
        console.log(`📄 File buffer size: ${fileBuffer.length} bytes`);
        
        const extractedData = await parsePDF(fileBuffer, analysisData.file.type);
        console.log(`📄 Extracted text length: ${extractedData.text.length} characters`);
        console.log(`📄 Extracted metrics: ${extractedData.metrics.length} metrics`);
        
        if (extractedData.text.length === 0) {
          throw new Error('No text could be extracted from the uploaded file. The file may be corrupted, password-protected, or contain only images.');
        }
        
        extractedValues = extractedData.metrics.reduce((acc, metric) => {
          acc[metric.name.toLowerCase()] = {
            value: String(metric.value),
            unit: metric.unit,
            status: metric.status
          };
          return acc;
        }, {} as Record<string, { value: string; unit: string; status: string }>);
        
        // Store the extracted text for Gemini analysis
        extractedValues.extracted_text = extractedData.text;
        
        analysisType = 'file';
      } catch (error) {
        console.error('❌ File processing error:', error);
        return NextResponse.json(
          { error: `File processing failed: ${error instanceof Error ? error.message : 'Unknown error'}` },
          { status: 400 }
        );
      }
    } else if (analysisData.manualValues) {
      // Use manual values
      extractedValues = {
        blood_pressure: analysisData.manualValues.bloodPressure,
        blood_sugar: analysisData.manualValues.bloodSugar,
        cholesterol: analysisData.manualValues.cholesterol,
        bmi: analysisData.manualValues.bmi
      };
    }

    // Run ML analysis
    const mlResult = await analyzeWithML(extractedValues, analysisData.symptoms, analysisData.modelType);

    // Use Gemini for final analysis and recommendations
    // Use Gemini Pro for other report analysis, Flash for others
    const apiKey = analysisData.modelType === 'other' 
      ? "AIzaSyCL3I6-whCE00pi66zwO9VVnBvFl2_WI_0"  // Gemini Pro API key
      : "AIzaSyCA_d4JYpI22_300sPKPkKCkkBT2DE1DxI"; // Gemini Flash API key
    
    const genAI = new GoogleGenerativeAI(apiKey);
    const model = genAI.getGenerativeModel({ 
      model: "gemini-2.0-flash" 
    });

    const systemInstruction = `You are Dr. Sarah Chen, a board-certified physician specializing in diagnostic medicine. You excel at detailed medical analysis and data interpretation. Provide a comprehensive, thorough medical assessment based on all available information.

ANALYSIS CONTEXT:
- ML Prediction: ${mlResult.predicted_disease} (${(mlResult.confidence * 100).toFixed(1)}% confidence)
- Patient Symptoms: ${analysisData.symptoms}
- Duration: ${analysisData.duration || 'Not specified'}
- Test Values: ${JSON.stringify(extractedValues)}

ANALYSIS REQUIREMENTS:
1. **Deep Data Analysis**: Examine every test value, symptom, and data point provided
2. **Clinical Correlation**: Connect symptoms with test results and ML predictions
3. **Risk Assessment**: Provide detailed risk stratification and urgency levels
4. **Differential Diagnosis**: Consider multiple possible conditions
5. **Evidence-Based Reasoning**: Base all recommendations on medical evidence
6. **Comprehensive Coverage**: Address all aspects of the patient's condition

RESPONSE REQUIREMENTS:
Provide ONLY valid JSON format with the following structure:
{
  "disease_name": "Primary suspected condition with detailed explanation",
  "confidence": 0.85,
  "description": "Comprehensive explanation of the condition, pathophysiology, and clinical presentation",
  "reasoning": "Detailed clinical reasoning based on all available data points, including symptom analysis, test result interpretation, and ML prediction correlation",
  "risk_level": "Low/Moderate/High/Critical with justification",
  "urgency": "Routine/Urgent/Emergent with timeline recommendations",
  "future_steps": "Specific, actionable medical recommendations for next steps with timelines and priorities",
  "top_medicines": [
    {
      "name": "Medicine Name",
      "dosage": "Recommended dosage",
      "description": "Detailed explanation of why this medicine is recommended, how it works, and what to monitor"
    }
  ],
  "lifestyle_changes": "Specific lifestyle modifications and health recommendations with implementation details and expected outcomes"
}

QUALITY STANDARDS:
- Provide detailed, comprehensive analysis that demonstrates deep medical knowledge
- Include specific numbers, ranges, and clinical parameters
- Give actionable, evidence-based recommendations
- Consider patient safety as the top priority
- Provide clear reasoning for all conclusions
- Include both immediate and long-term management strategies
- Address potential complications and warning signs
- Ensure recommendations are practical and implementable`;

    const fullPrompt = `${systemInstruction}\n\nPatient Symptoms: ${analysisData.symptoms || 'Not provided'}\nDuration: ${analysisData.duration || 'Not specified'}\n\nTest Results:\n${JSON.stringify(extractedValues, null, 2)}${extractedValues.extracted_text ? `\n\nExtracted Report Text:\n${extractedValues.extracted_text}` : ''}\n\nPlease provide your medical assessment based on the above information.`;

    const result = await model.generateContent(fullPrompt);
    const responseText = result.response.text();

    // Clean and parse response
    let cleanedText = responseText.trim();
    if (cleanedText.startsWith('```json')) {
      cleanedText = cleanedText.replace(/^```json\s*/, '');
    } else if (cleanedText.startsWith('```')) {
      cleanedText = cleanedText.replace(/^```\s*/, '');
    }
    if (cleanedText.endsWith('```')) {
      cleanedText = cleanedText.replace(/\s*```$/, '');
    }
    cleanedText = cleanedText.trim();

    let parsedResponse: TestAnalysisResponse;
    try {
      parsedResponse = JSON.parse(cleanedText);
    } catch (parseError) {
      console.error("🔴 JSON Parse Error:", parseError);
      
      // Fallback response
      parsedResponse = {
        disease_name: mlResult.predicted_disease,
        confidence: mlResult.confidence,
        future_steps: "Please consult with a healthcare provider for proper diagnosis and treatment based on your test results.",
        top_medicines: [],
        lifestyle_changes: "Maintain a healthy lifestyle with regular exercise and balanced nutrition. Follow your healthcare provider's recommendations.",
        analysis_type: analysisType,
        extracted_values: extractedValues
      };
    }

    // Add analysis metadata
    parsedResponse.analysis_type = analysisType;
    parsedResponse.extracted_values = extractedValues;

    console.log(`✅ Test analysis complete: ${parsedResponse.disease_name} (${(parsedResponse.confidence * 100).toFixed(1)}% confidence)`);

    return NextResponse.json(parsedResponse);

  } catch (error) {
    console.error("🔴 Error in /api/test-analysis:", error);

    let errorMessage = "An error occurred while analyzing your test results.";
    let statusCode = 500;

    if (error instanceof Error) {
      if (error.message.includes("file") || error.message.includes("upload")) {
        errorMessage = "File processing error. Please ensure your file is readable and try again.";
        statusCode = 422;
      } else if (error.message.includes("ML") || error.message.includes("analysis")) {
        errorMessage = "Analysis service temporarily unavailable. Please try again later.";
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

// Health check endpoint
export async function GET(): Promise<NextResponse<{ status: string; service: string }>> {
  return NextResponse.json({
    status: "healthy",
    service: "Test Analysis API"
  });
}
