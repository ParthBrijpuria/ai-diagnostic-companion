/**
 * Find Doctor API Route
 * Finds doctors based on symptoms and location using Gemini API
 */

import { NextRequest, NextResponse } from 'next/server';
import { GoogleGenerativeAI } from '@google/generative-ai';

// Define interfaces
interface FindDoctorRequest {
  symptoms: string;
  city: string;
}

interface Doctor {
  name: string;
  specialty: string;
  hospital: string;
  experience: string;
  rating: string;
  location: string;
  phone: string;
  consultation_fee: string;
}

interface FindDoctorResponse {
  doctors: Doctor[];
  analysis_type: 'find_doctor';
}

export async function POST(req: NextRequest): Promise<NextResponse<FindDoctorResponse | { error: string }>> {
  console.log("\n--- /api/find-doctor endpoint was called ---");

  try {
    const body = await req.json();
    const { symptoms, city } = body as FindDoctorRequest;

    if (!symptoms || !city) {
      return NextResponse.json(
        { error: "Both symptoms and city are required." },
        { status: 400 }
      );
    }

    console.log(`🔍 Finding doctors for symptoms: ${symptoms} in city: ${city}`);

    // Use Gemini API to find doctors
    const apiKey = "AIzaSyCA_d4JYpI22_300sPKPkKCkkBT2DE1DxI";
    const genAI = new GoogleGenerativeAI(apiKey);
    const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

    const systemInstruction = `You are a medical directory assistant that helps patients find the right doctors based on their symptoms and location.

TASK: Find 3-5 best doctors for the given symptoms in the specified city.

RESPONSE REQUIREMENTS:
Provide ONLY valid JSON format with the following structure:
{
  "doctors": [
    {
      "name": "Dr. [Full Name]",
      "specialty": "[Medical Specialty]",
      "hospital": "[Hospital/Clinic Name]",
      "experience": "[X years of experience]",
      "rating": "[X.X/5.0]",
      "location": "[Address in the city]",
      "phone": "[Phone number]",
      "consultation_fee": "[Fee amount]"
    }
  ]
}

GUIDELINES:
- Match symptoms to appropriate medical specialties
- Include doctors from different hospitals/clinics
- Provide realistic ratings (3.5-5.0)
- Include complete contact information
- Make consultation fees reasonable for the location
- Ensure all doctors are in the specified city
- Focus on doctors who specialize in treating the described symptoms

EXAMPLE SPECIALTIES BY SYMPTOMS:
- Chest pain, heart issues → Cardiologist
- Breathing problems → Pulmonologist
- Headaches, neurological issues → Neurologist
- Joint pain, arthritis → Rheumatologist
- Skin problems → Dermatologist
- Eye problems → Ophthalmologist
- Mental health issues → Psychiatrist
- General symptoms → General Practitioner/Internal Medicine`;

    const prompt = `Find the best doctors for these symptoms: "${symptoms}" in the city: "${city}".

Please provide 3-5 doctors with their complete details including specialty, hospital, experience, rating, location, phone, and consultation fee.`;

    try {
      const result = await model.generateContent([systemInstruction, prompt]);
      const response = await result.response;
      const text = response.text();

      console.log("--- Raw response from Gemini ---");
      console.log(text);

      // Parse JSON response
      let doctorsData;
      try {
        // Extract JSON from response (remove any markdown formatting)
        const jsonMatch = text.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          doctorsData = JSON.parse(jsonMatch[0]);
        } else {
          throw new Error("No valid JSON found in response");
        }
      } catch (parseError) {
        console.error("JSON parsing error:", parseError);
        throw new Error("Failed to parse doctor recommendations");
      }

      // Validate response structure
      if (!doctorsData.doctors || !Array.isArray(doctorsData.doctors)) {
        throw new Error("Invalid response structure from AI");
      }

      const response_data: FindDoctorResponse = {
        doctors: doctorsData.doctors,
        analysis_type: 'find_doctor'
      };

      console.log(`✅ Found ${doctorsData.doctors.length} doctors for ${symptoms} in ${city}`);

      return NextResponse.json(response_data);

    } catch (geminiError) {
      console.error("Gemini API error:", geminiError);
      throw new Error("Failed to get doctor recommendations from AI service");
    }

  } catch (error) {
    console.error("🔴 Error in /api/find-doctor:", error);

    let errorMessage = "An error occurred while finding doctors.";
    let statusCode = 500;

    if (error instanceof Error) {
      if (error.message.includes("symptoms") || error.message.includes("city")) {
        errorMessage = "Please provide both symptoms and city.";
        statusCode = 400;
      } else if (error.message.includes("AI") || error.message.includes("Gemini")) {
        errorMessage = "Doctor search service temporarily unavailable. Please try again later.";
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
    service: "Find Doctor API"
  });
}
