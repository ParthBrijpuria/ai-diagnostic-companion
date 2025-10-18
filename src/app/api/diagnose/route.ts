// Production-ready backend API route for Google Gemini integration
import { GoogleGenerativeAI } from '@google/generative-ai';
import { NextRequest, NextResponse } from 'next/server';

// Define interfaces for type safety
interface DiagnoseRequestBody {
    symptoms?: string;
    duration?: string;
    severity?: string;
}

interface Disease {
    name: string;
    probability: string;
    why: string;
}

interface RecommendedTest {
    test: string;
    why: string;
}

interface ApiResponse {
    possible_diseases: Disease[];
    recommended_tests: RecommendedTest[];
    home_remedies: string;
}

export async function POST(req: NextRequest) {
    console.log("\n--- /api/diagnose endpoint was called ---");

    // Use the provided API key
    const apiKey = "AIzaSyDeZZoP13Q3ordTwXn9ytX5AUBf6YnMI7M";
    
    // Use real AI instead of mock data
    const useMockData = false;
    
    console.log("🔶 Using Google Gemini AI for analysis");

    try {

            // Parse and validate request body
            const body: DiagnoseRequestBody = await req.json();
            const { symptoms, duration, severity } = body;

            if (!symptoms || symptoms.trim().length === 0) {
                return NextResponse.json(
                    { error: "Symptoms are required and cannot be empty." },
                    { status: 400 }
                );
            }

        let parsedResponse: ApiResponse;

        if (useMockData) {
            // Fallback mock response (should not be used with real API key)
            parsedResponse = {
                possible_diseases: [
                    {
                        name: "Analysis Unavailable",
                        probability: "Unknown",
                        why: "Unable to process the symptoms at this time. Please consult a healthcare professional."
                    }
                ],
                recommended_tests: [
                    {
                        test: "Professional Medical Consultation",
                        why: "A healthcare provider can properly assess your symptoms and provide appropriate care."
                    }
                ],
                home_remedies: "Please consult a healthcare professional for proper evaluation of your symptoms."
            };
        } else {
            // Real Google Gemini AI integration
            const genAI = new GoogleGenerativeAI(apiKey);
            const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

            // Enhanced system instruction for medical analysis
            const systemInstruction = `You are Dr. Sarah Chen, a board-certified internal medicine physician with 15 years of experience in diagnostic medicine. You specialize in symptom analysis and differential diagnosis.

MEDICAL EXPERTISE:
- Expert in internal medicine, emergency medicine, and diagnostic reasoning
- Extensive knowledge of symptom patterns, disease progression, and diagnostic testing
- Focus on evidence-based medicine and patient safety
- Experience with both common and rare medical conditions

ANALYSIS METHODOLOGY:
1. SYSTEMATIC APPROACH: Analyze symptoms using a structured diagnostic framework
2. DIFFERENTIAL DIAGNOSIS: Consider multiple possibilities ranked by likelihood
3. RISK ASSESSMENT: Identify red flags and urgent conditions
4. EVIDENCE-BASED: Base recommendations on current medical literature and guidelines
5. PATIENT-CENTERED: Provide clear, actionable advice tailored to the patient's situation

RESPONSE REQUIREMENTS:
- Provide ONLY valid JSON format (no markdown, no explanations outside JSON)
- Use precise medical terminology while remaining accessible
- Consider symptom severity, duration, and associated factors
- Include both common and serious conditions when appropriate
- Always prioritize patient safety and recommend professional consultation

JSON STRUCTURE:
{
  "possible_diseases": [
    {
      "name": "Specific Medical Condition (Common Name)",
      "probability": "High/Medium/Low",
      "why": "Detailed medical reasoning based on symptom analysis, including pathophysiology and clinical presentation"
    }
  ],
  "recommended_tests": [
    {
      "test": "Specific Diagnostic Test",
      "why": "Medical justification explaining what this test will help diagnose or rule out"
    }
  ],
  "home_remedies": "Specific, actionable home remedies and natural treatments with clear instructions"
}

IMPORTANT REQUIREMENTS:
- LIMIT to TOP 5 diseases only (most likely conditions based on symptoms)
- For complex medical names, ALWAYS include common name in brackets: "Medical Name (Common Name)"
- Prioritize diseases by likelihood and clinical relevance
- Exclude low-probability conditions to keep recommendations focused

CLINICAL GUIDELINES:
- Always consider the most common causes first
- Assess for red flag symptoms that require immediate attention
- Consider age, gender, medical history, and risk factors
- Suggest appropriate diagnostic workup based on clinical presentation
- Provide clear guidance on when to seek immediate medical care
- Emphasize the importance of professional medical evaluation`;

                const fullPrompt = `${systemInstruction}\n\nPatient Symptoms: ${symptoms.trim()}${duration ? `\nDuration: ${duration.trim()}` : ''}${severity ? `\nSeverity Level: ${severity}` : ''}`;

            // Generate content with Gemini
            const result = await model.generateContent(fullPrompt);
            const responseText = result.response.text();

            console.log("--- Raw response from Gemini ---");
            console.log(responseText);

            // Clean and parse response
            let cleanedText = responseText.trim();
            
            // Remove markdown code blocks if present
            if (cleanedText.startsWith('```json')) {
                cleanedText = cleanedText.replace(/^```json\s*/, '');
            } else if (cleanedText.startsWith('```')) {
                cleanedText = cleanedText.replace(/^```\s*/, '');
            }
            
            if (cleanedText.endsWith('```')) {
                cleanedText = cleanedText.replace(/\s*```$/, '');
            }
            
            cleanedText = cleanedText.trim();

            console.log("--- Cleaned response ---");
            console.log(cleanedText);

            // Parse JSON response
            try {
                parsedResponse = JSON.parse(cleanedText);
            } catch (parseError) {
                console.error("🔴 JSON Parse Error:", parseError);
                console.error("🔴 Raw text that failed to parse:", cleanedText);
                
                // Fallback response if JSON parsing fails
                parsedResponse = {
                    possible_diseases: [
                        {
                            name: "Analysis Unavailable",
                            probability: "Unknown",
                            why: "Unable to process the symptoms at this time. Please consult a healthcare professional."
                        }
                    ],
                    recommended_tests: [
                        {
                            test: "Professional Medical Consultation",
                            why: "A healthcare provider can properly assess your symptoms and provide appropriate care."
                        }
                    ],
                    home_remedies: "Please consult a healthcare professional for proper evaluation of your symptoms."
                };
            }

            // Validate response structure
            if (!parsedResponse.possible_diseases || !parsedResponse.recommended_tests || !parsedResponse.home_remedies) {
                throw new Error("Invalid response structure from AI model");
            }
        }

        return NextResponse.json(parsedResponse);

    } catch (error) {
        console.error("🔴 FULL ERROR OBJECT in /api/diagnose:", JSON.stringify(error, null, 2));
        
        let errorMessage = "An error occurred while processing your request.";
        let statusCode = 500;

        if (error instanceof Error) {
            if (error.message.includes("API key")) {
                errorMessage = "Server configuration error.";
                statusCode = 500;
            } else if (error.message.includes("quota") || error.message.includes("limit")) {
                errorMessage = "Service temporarily unavailable. Please try again later.";
                statusCode = 503;
            } else if (error.message.includes("network") || error.message.includes("timeout")) {
                errorMessage = "Network error. Please check your connection and try again.";
                statusCode = 503;
            } else {
                errorMessage = error.message;
            }
        }

        return NextResponse.json({ error: errorMessage }, { status: statusCode });
    }
}

