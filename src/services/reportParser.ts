/**
 * Medical Report Parser Service
 * Extracts text and health metrics from PDF and image files
 */

import { PDFParse } from 'pdf-parse';
import sharp from 'sharp';

// Define interfaces for extracted data
export interface HealthMetric {
  name: string;
  value: number | string;
  unit: string;
  status: 'normal' | 'high' | 'low' | 'critical' | 'unknown';
  reference_range?: string;
}

export interface ExtractedData {
  text: string;
  metrics: HealthMetric[];
  file_type: string;
  processing_time: number;
}

// Common lab test patterns and their reference ranges
const LAB_PATTERNS = {
  // Blood tests
  hemoglobin: {
    pattern: /hemoglobin|hgb|hb\s*:?\s*(\d+\.?\d*)\s*(g\/dl|g\/l|gm\/dl)/i,
    unit: 'g/dL',
    normal_range: { min: 12, max: 16 },
    critical_low: 7,
    critical_high: 20
  },
  hematocrit: {
    pattern: /hematocrit|hct\s*:?\s*(\d+\.?\d*)\s*(%|percent)/i,
    unit: '%',
    normal_range: { min: 36, max: 46 },
    critical_low: 21,
    critical_high: 60
  },
  wbc: {
    pattern: /white\s*blood\s*cell|wbc|leukocyte\s*:?\s*(\d+\.?\d*)\s*(k\/ul|×10³\/μl|10³\/μl)/i,
    unit: 'K/μL',
    normal_range: { min: 4.5, max: 11.0 },
    critical_low: 2.0,
    critical_high: 30.0
  },
  rbc: {
    pattern: /red\s*blood\s*cell|rbc|erythrocyte\s*:?\s*(\d+\.?\d*)\s*(m\/ul|×10⁶\/μl|10⁶\/μl)/i,
    unit: 'M/μL',
    normal_range: { min: 4.2, max: 5.4 },
    critical_low: 2.0,
    critical_high: 7.0
  },
  platelet: {
    pattern: /platelet|plt\s*:?\s*(\d+\.?\d*)\s*(k\/ul|×10³\/μl|10³\/μl)/i,
    unit: 'K/μL',
    normal_range: { min: 150, max: 450 },
    critical_low: 50,
    critical_high: 1000
  },
  
  // Chemistry tests
  glucose: {
    pattern: /glucose|blood\s*sugar|bs\s*:?\s*(\d+\.?\d*)\s*(mg\/dl|mmol\/l)/i,
    unit: 'mg/dL',
    normal_range: { min: 70, max: 100 },
    critical_low: 40,
    critical_high: 400
  },
  creatinine: {
    pattern: /creatinine\s*:?\s*(\d+\.?\d*)\s*(mg\/dl|μmol\/l)/i,
    unit: 'mg/dL',
    normal_range: { min: 0.6, max: 1.2 },
    critical_low: 0.3,
    critical_high: 5.0
  },
  bun: {
    pattern: /blood\s*urea\s*nitrogen|bun\s*:?\s*(\d+\.?\d*)\s*(mg\/dl|mmol\/l)/i,
    unit: 'mg/dL',
    normal_range: { min: 7, max: 20 },
    critical_low: 2,
    critical_high: 100
  },
  sodium: {
    pattern: /sodium|na\s*:?\s*(\d+\.?\d*)\s*(meq\/l|mmol\/l)/i,
    unit: 'mEq/L',
    normal_range: { min: 136, max: 145 },
    critical_low: 120,
    critical_high: 160
  },
  potassium: {
    pattern: /potassium|k\s*:?\s*(\d+\.?\d*)\s*(meq\/l|mmol\/l)/i,
    unit: 'mEq/L',
    normal_range: { min: 3.5, max: 5.0 },
    critical_low: 2.5,
    critical_high: 7.0
  },
  
  // Lipid panel
  cholesterol: {
    pattern: /total\s*cholesterol|cholesterol\s*:?\s*(\d+\.?\d*)\s*(mg\/dl|mmol\/l)/i,
    unit: 'mg/dL',
    normal_range: { min: 0, max: 200 },
    critical_low: 0,
    critical_high: 300
  },
  hdl: {
    pattern: /hdl|high\s*density\s*lipoprotein\s*:?\s*(\d+\.?\d*)\s*(mg\/dl|mmol\/l)/i,
    unit: 'mg/dL',
    normal_range: { min: 40, max: 100 },
    critical_low: 20,
    critical_high: 150
  },
  ldl: {
    pattern: /ldl|low\s*density\s*lipoprotein\s*:?\s*(\d+\.?\d*)\s*(mg\/dl|mmol\/l)/i,
    unit: 'mg/dL',
    normal_range: { min: 0, max: 100 },
    critical_low: 0,
    critical_high: 200
  },
  
  // Liver function
  alt: {
    pattern: /alt|alanine\s*aminotransferase\s*:?\s*(\d+\.?\d*)\s*(u\/l|iu\/l)/i,
    unit: 'U/L',
    normal_range: { min: 7, max: 56 },
    critical_low: 0,
    critical_high: 500
  },
  ast: {
    pattern: /ast|aspartate\s*aminotransferase\s*:?\s*(\d+\.?\d*)\s*(u\/l|iu\/l)/i,
    unit: 'U/L',
    normal_range: { min: 10, max: 40 },
    critical_low: 0,
    critical_high: 500
  },
  bilirubin: {
    pattern: /bilirubin\s*:?\s*(\d+\.?\d*)\s*(mg\/dl|μmol\/l)/i,
    unit: 'mg/dL',
    normal_range: { min: 0.3, max: 1.2 },
    critical_low: 0,
    critical_high: 10
  }
};

/**
 * Main function to parse PDF or image files
 */
export async function parsePDF(fileBuffer: Buffer, fileType: string): Promise<ExtractedData> {
  const startTime = Date.now();
  
  try {
    let extractedText = '';

    if (fileType === 'application/pdf') {
      // Extract text from PDF
      try {
        const pdfData = await new PDFParse(fileBuffer);
        extractedText = pdfData.text;
        
        if (!extractedText || extractedText.trim().length === 0) {
          throw new Error('PDF appears to be empty or contains only images without extractable text');
        }
        
        console.log(`📄 Successfully extracted ${extractedText.length} characters from PDF`);
      } catch (pdfError) {
        console.error('PDF parsing error:', pdfError);
        throw new Error(`Failed to extract text from PDF: ${pdfError instanceof Error ? pdfError.message : 'Unknown PDF parsing error'}`);
      }
    } else if (fileType.startsWith('image/')) {
      // For images, we would need OCR (Optical Character Recognition)
      // For now, return a placeholder that indicates OCR is needed
      extractedText = '[IMAGE FILE - OCR processing required]';
    } else {
      throw new Error(`Unsupported file type: ${fileType}`);
    }

    // Extract health metrics from the text
    const metrics = extractHealthMetrics(extractedText);

    const processingTime = Date.now() - startTime;

    return {
      text: extractedText,
      metrics,
      file_type: fileType,
      processing_time: processingTime
    };

  } catch (error) {
    console.error('Error parsing file:', error);
    throw new Error(`Failed to parse file: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

/**
 * Extract health metrics from text using regex patterns
 */
function extractHealthMetrics(text: string): HealthMetric[] {
  const metrics: HealthMetric[] = [];

  // Process each lab test pattern
  Object.entries(LAB_PATTERNS).forEach(([testName, config]) => {
    const matches = text.match(config.pattern);
    
    if (matches && matches[1]) {
      const value = parseFloat(matches[1]);
      
      if (!isNaN(value)) {
        const status = determineStatus(value, config);
        const referenceRange = `${config.normal_range.min}-${config.normal_range.max} ${config.unit}`;
        
        metrics.push({
          name: testName.toUpperCase(),
          value,
          unit: config.unit,
          status,
          reference_range: referenceRange
        });
      }
    }
  });

  return metrics;
}

/**
 * Determine the status of a lab value (normal, high, low, critical)
 */
function determineStatus(value: number, config: any): HealthMetric['status'] {
  if (value < config.critical_low || value > config.critical_high) {
    return 'critical';
  } else if (value < config.normal_range.min) {
    return 'low';
  } else if (value > config.normal_range.max) {
    return 'high';
  } else {
    return 'normal';
  }
}

/**
 * Extract additional context from the report text
 */
export function extractReportContext(text: string): {
  patient_age?: number;
  patient_gender?: string;
  report_date?: string;
  lab_name?: string;
} {
  const context: any = {};

  // Extract age
  const ageMatch = text.match(/(?:age|aged?)\s*:?\s*(\d+)/i);
  if (ageMatch) {
    context.patient_age = parseInt(ageMatch[1]);
  }

  // Extract gender
  const genderMatch = text.match(/(?:sex|gender)\s*:?\s*(male|female|m|f)/i);
  if (genderMatch) {
    context.patient_gender = genderMatch[1].toLowerCase();
  }

  // Extract date
  const dateMatch = text.match(/(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})/);
  if (dateMatch) {
    context.report_date = dateMatch[1];
  }

  // Extract lab name
  const labMatch = text.match(/(?:lab|laboratory|clinic|hospital)\s*:?\s*([a-zA-Z\s]+)/i);
  if (labMatch) {
    context.lab_name = labMatch[1].trim();
  }

  return context;
}
