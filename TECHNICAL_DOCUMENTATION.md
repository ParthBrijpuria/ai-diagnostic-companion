# 🏥 AI Diagnostic Companion - Complete Technical Documentation

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture Deep Dive](#architecture-deep-dive)
3. [File-by-File Analysis](#file-by-file-analysis)
4. [API Endpoints Documentation](#api-endpoints-documentation)
5. [Machine Learning Models](#machine-learning-models)
6. [Data Flow Architecture](#data-flow-architecture)
7. [Security & Privacy](#security--privacy)
8. [Performance Optimization](#performance-optimization)
9. [Deployment Strategy](#deployment-strategy)
10. [Development Workflow](#development-workflow)

---

## 🎯 Project Overview

The AI Diagnostic Companion is a comprehensive medical AI platform that combines:
- **Local Machine Learning Models** for medical image analysis
- **Google Gemini AI** for symptom analysis and medical reasoning
- **Next.js 15.5.4** with TypeScript for the frontend
- **Python Flask** for ML service backend
- **Vercel** for cloud deployment

### Core Capabilities
1. **Symptom Analysis** - AI-powered medical diagnosis using Gemini
2. **Medical Image Analysis** - MRI brain scans, Chest X-rays
3. **Health Parameter Analysis** - Diabetes prediction from lab values
4. **Doctor Recommendations** - Location-based specialist search
5. **Report Processing** - PDF/image medical report analysis

---

## 🏗️ Architecture Deep Dive

### Technology Stack
```
Frontend: Next.js 15.5.4 + React 19.1.0 + TypeScript 5.0+
Backend: Next.js API Routes + Python Flask
AI/ML: Google Gemini AI + Scikit-learn + OpenCV
Database: File-based (temporary processing)
Deployment: Vercel + Serverless Functions
Styling: Tailwind CSS + Custom CSS Modules
```

### System Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   API Routes     │    │   ML Models     │
│   (Next.js)     │◄──►│   (Next.js)      │◄──►│   (Python)      │
│                 │    │                  │    │                 │
│ • Symptom UI    │    │ • /api/diagnose  │    │ • MRI Analysis  │
│ • Report Upload │    │ • /api/mri-*     │    │ • X-ray Analysis│
│ • Doctor Search │    │ • /api/xray-*    │    │ • Diabetes ML   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Gemini AI     │    │   File System    │    │   Datasets      │
│   (Cloud)       │    │   (Temporary)    │    │   (Local)       │
│                 │    │                  │    │                 │
│ • Symptom AI    │    │ • Temp uploads   │    │ • MRI images    │
│ • Medical AI    │    │ • Auto cleanup   │    │ • X-ray images  │
│ • Doctor AI     │    │ • Security       │    │ • Health data   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

---

## 📁 File-by-File Analysis

### 🔧 Configuration Files

#### `package.json` - Project Dependencies & Scripts
**Purpose**: Defines Node.js project configuration, dependencies, and build scripts
**Key Dependencies**:
- `next: 15.5.4` - React framework with App Router
- `react: 19.1.0` - Latest React with concurrent features
- `@google/generative-ai: ^0.21.0` - Gemini AI integration
- `axios: ^1.12.2` - HTTP client for API calls
- `pdf-parse: ^2.2.6` - PDF text extraction
- `formidable: ^3.5.4` - File upload handling
- `sharp: ^0.34.4` - Image processing

**Scripts**:
- `dev` - Development server with Turbopack
- `build` - Production build with Turbopack
- `start` - Production server
- `lint` - ESLint code quality check

#### `tsconfig.json` - TypeScript Configuration
**Purpose**: Configures TypeScript compiler for strict type checking
**Key Settings**:
- `strict: true` - Enables all strict type checks
- `target: ES2020` - Modern JavaScript features
- `module: ESNext` - Latest module system
- `jsx: preserve` - Preserves JSX for Next.js

#### `next.config.ts` - Next.js Configuration
**Purpose**: Configures Next.js build and runtime behavior
**Key Features**:
- Turbopack for faster development
- Image optimization settings
- API route configuration
- Build optimization

#### `tailwind.config.ts` - Styling Configuration
**Purpose**: Configures Tailwind CSS for responsive design
**Customizations**:
- Medical-themed color palette
- Custom spacing and typography
- Responsive breakpoints
- Component-specific styles

#### `vercel.json` - Deployment Configuration
**Purpose**: Configures Vercel deployment with security and performance settings
**Security Headers**:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: origin-when-cross-origin`

**Performance Settings**:
- API timeout: 30 seconds for diagnose endpoint
- Static asset caching: 1 year
- API route caching: no-cache

### 🎨 Frontend Application (`src/`)

#### `src/app/page.tsx` - Main Application Component (1025 lines)
**Purpose**: Central React component implementing the complete UI
**Key Features**:
- **State Management**: Uses React hooks for complex state
- **Three Main Sections**:
  1. Symptom Analysis with severity slider
  2. Report Upload (MRI, X-ray, Diabetes)
  3. Doctor Finder with location search
- **Responsive Design**: Mobile-first approach with conditional rendering
- **API Integration**: Connects to all backend endpoints
- **Error Handling**: Comprehensive error states and user feedback

**State Variables**:
```typescript
const [symptoms, setSymptoms] = useState<string>('');
const [severity, setSeverity] = useState<number>(1);
const [analysis, setAnalysis] = useState<ApiResponse | null>(null);
const [loading, setLoading] = useState<boolean>(false);
// ... 15+ more state variables
```

**Key Functions**:
- `handleSymptomAnalysis()` - Processes symptom input with Gemini AI
- `handleFileUpload()` - Manages file uploads for different analysis types
- `handleDoctorSearch()` - Integrates symptom data with doctor search

#### `src/app/layout.tsx` - Root Layout
**Purpose**: Defines the root HTML structure and global providers
**Features**:
- Metadata configuration for SEO
- Font loading (Inter font family)
- Global CSS imports
- Error boundary setup

#### `src/app/globals.css` - Global Styles
**Purpose**: Defines global CSS styles and Tailwind imports
**Key Styles**:
- Medical-themed color scheme
- Responsive typography
- Custom animations and transitions
- Accessibility-focused design

### 🔌 API Routes (`src/app/api/`)

#### `src/app/api/diagnose/route.ts` - Symptom Analysis API (220 lines)
**Purpose**: Processes symptom analysis using Google Gemini AI
**Input Interface**:
```typescript
interface DiagnoseRequestBody {
    symptoms?: string;
    duration?: string;
    severity?: string;
}
```

**AI Prompt Structure**:
- **System Instruction**: Dr. Sarah Chen persona with medical expertise
- **Analysis Methodology**: 5-step diagnostic approach
- **Response Format**: Structured JSON with diseases, tests, and recommendations
- **Safety Guidelines**: Patient safety and professional consultation emphasis

**Key Features**:
- Comprehensive medical reasoning (131 lines of system instruction)
- JSON response parsing with error handling
- Fallback mechanisms for API failures
- Rate limiting and timeout handling

#### `src/app/api/test-analysis/route.ts` - Comprehensive Test Analysis
**Purpose**: Integrates ML predictions with Gemini AI for detailed medical assessment
**Processing Flow**:
1. File upload or manual value input
2. ML model prediction
3. Gemini AI analysis with ML context
4. Comprehensive medical assessment

**Response Structure**:
```typescript
{
  disease_name: string;
  confidence: number;
  description: string;
  reasoning: string;
  risk_level: string;
  urgency: string;
  future_steps: string;
  top_medicines: Medicine[];
  lifestyle_changes: string;
}
```

#### `src/app/api/mri-analysis/route.ts` - MRI Brain Scan Analysis
**Purpose**: Processes MRI images using local Python ML models
**Technical Implementation**:
- **File Handling**: Temporary file creation and cleanup
- **Python Integration**: Dynamic script generation and execution
- **Model Loading**: `mri_enhanced_predict.py` with ensemble model
- **Feature Extraction**: 218 features (statistical + texture + pixels)

**Security Measures**:
- File type validation (JPEG, PNG only)
- File size limits (10MB max)
- Temporary file auto-cleanup
- Process timeout (30 seconds)

#### `src/app/api/xray-analysis/route.ts` - Chest X-ray Analysis
**Purpose**: Detects pneumonia in chest X-ray images
**Model Details**:
- **Algorithm**: GradientBoosting classifier
- **Accuracy**: 75.16% on test dataset
- **Features**: 113 features (5 basic + 8 texture + 100 pixels)
- **Classes**: Normal vs Pneumonic

#### `src/app/api/diabetes-analysis/route.ts` - Diabetes Prediction
**Purpose**: Predicts diabetes risk from health parameters
**Input Parameters**:
- Gender, Pregnancies, Glucose, Blood Pressure
- Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age

**Model**: RandomForest classifier with parameter validation

#### `src/app/api/find-doctor/route.ts` - Doctor Recommendation Service
**Purpose**: Uses Gemini AI to find doctors based on symptoms and location
**AI Prompt Features**:
- Medical specialty matching
- Location-based search
- Realistic doctor data generation
- Contact information and ratings

**Response Format**:
```typescript
{
  doctors: [{
    name: string;
    specialty: string;
    hospital: string;
    experience: string;
    rating: string;
    location: string;
    phone: string;
    consultation_fee: string;
  }];
}
```

### 🧩 React Components (`src/components/`)

#### `src/components/ReportUpload.tsx` - Report Upload Interface (821 lines)
**Purpose**: Comprehensive file upload and analysis interface
**Features**:
- **Drag & Drop**: File upload with visual feedback
- **Multiple Formats**: PDF, JPEG, PNG support
- **Analysis Types**: MRI, X-ray, Diabetes, General reports
- **Manual Input**: Direct parameter entry for diabetes
- **Results Display**: Confidence scores and detailed analysis
- **Loading States**: Progress indicators and error handling

**Key Functions**:
- `handleFileUpload()` - Processes file uploads
- `handleManualSubmit()` - Processes manual diabetes parameters
- `displayAnalysisResults()` - Renders analysis results

#### `src/components/FindDoc.tsx` - Doctor Search Component
**Purpose**: Location-based doctor search with symptom integration
**Features**:
- Symptom data transfer from analysis
- Location input with validation
- Doctor recommendation display
- Contact information and ratings
- Responsive design for mobile

### 🔧 Services (`src/services/`)

#### `src/services/mlAnalysis.ts` - ML Analysis Service
**Purpose**: Interfaces with Python ML service for disease prediction
**Key Functions**:
- `analyzeWithML()` - Main analysis function
- `determineModelType()` - Model selection logic
- `generateFallbackPrediction()` - Fallback when ML service unavailable

**Configuration**:
```typescript
const ML_SERVICE_CONFIG = {
  baseUrl: 'http://localhost:5000',
  timeout: 30000,
  retries: 3
};
```

#### `src/services/mlService.ts` - ML Service Wrapper
**Purpose**: Provides unified interface for ML model predictions
**Features**:
- Multiple model type support
- Error handling and retries
- Fallback prediction generation
- Service health monitoring

#### `src/services/reportParser.ts` - Medical Report Parser
**Purpose**: Extracts and processes medical report data
**Capabilities**:
- PDF text extraction using pdf-parse
- Image text extraction (OCR capabilities)
- Health metric parsing and validation
- Normal/abnormal status determination

---

## 🤖 Machine Learning Models

### 🧠 MRI Analysis Models (`models/mri/`)

#### `models/mri/mri_boosted_ensemble.py` - Best Performing MRI Model
**Purpose**: Brain tumor classification with 78.18% accuracy
**Architecture**: Ensemble of 6 algorithms
- RandomForest Classifier
- GradientBoosting Classifier
- ExtraTrees Classifier
- SVM (RBF kernel)
- K-Nearest Neighbors
- Logistic Regression

**Feature Engineering**:
- **Basic Features** (5): Mean, std, min, max, median
- **Texture Features** (8): Sobel gradients, Laplacian
- **LBP Features** (5): Local Binary Patterns
- **Pixel Features** (200): First 200 pixels
- **Total**: 218 features

**Training Process**:
1. Image preprocessing (64x64 grayscale)
2. Feature extraction
3. Feature scaling with StandardScaler
4. Ensemble training with voting classifier
5. Model serialization with joblib

#### `models/mri/mri_enhanced_predict.py` - Prediction Script
**Purpose**: Loads trained model and makes predictions
**Key Functions**:
- `predict_mri_image(image_path)` - Main prediction function
- `extract_features_enhanced()` - Feature extraction
- Model loading and preprocessing
- Confidence score calculation

**Classes**: Glioma, Meningioma, Pituitary, No Tumor

### 🫁 X-ray Analysis Models (`models/xray/`)

#### `models/xray/xray_enhanced_predict.py` - X-ray Prediction
**Purpose**: Pneumonia detection with 75.16% accuracy
**Model**: GradientBoosting Classifier
**Features**: 113 features (5 basic + 8 texture + 100 pixels)
**Classes**: Normal vs Pneumonic

**Feature Extraction**:
```python
def extract_features_quick(img_array):
    # Basic statistical features
    basic_features = [mean, std, min, max, median]
    # Texture features (Sobel, Laplacian)
    texture_features = [8 texture measures]
    # Pixel features
    pixel_features = img_array[:100]
    return basic_features + texture_features + pixel_features
```

### 🩺 Diabetes Analysis Models (`models/diabetes/`)

#### `models/diabetes/diabetes_classifier.py` - Diabetes Prediction
**Purpose**: Predicts diabetes risk from health parameters
**Model**: RandomForest Classifier
**Parameters**: 8 health metrics
**Accuracy**: 90%+ on test data

**Input Parameters**:
- Gender, Pregnancies, Glucose, Blood Pressure
- Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age

---

## 📊 Data Flow Architecture

### Symptom Analysis Flow
```
User Input → Frontend → /api/diagnose → Gemini AI → JSON Response → UI Display
```

### Medical Image Analysis Flow
```
File Upload → Temporary Storage → Python Script → ML Model → Prediction → Cleanup → Response
```

### Report Analysis Flow
```
PDF/Image → Text Extraction → ML Analysis → Gemini AI → Comprehensive Report → UI Display
```

### Doctor Search Flow
```
Symptoms + Location → /api/find-doctor → Gemini AI → Doctor Data → UI Display
```

---

## 🔒 Security & Privacy

### Data Protection
- **No Data Storage**: All processing is temporary
- **Auto Cleanup**: Temporary files deleted after processing
- **Input Validation**: File type and size restrictions
- **API Security**: Environment variables for sensitive data

### Security Headers (vercel.json)
- Content Security Policy
- XSS Protection
- Frame Options
- Content Type Options
- Referrer Policy

### API Security
- Rate limiting on sensitive endpoints
- Input sanitization and validation
- Error message sanitization
- Timeout protection

---

## ⚡ Performance Optimization

### Frontend Optimization
- **Next.js 15.5.4**: Latest performance improvements
- **Turbopack**: Faster development builds
- **Code Splitting**: Automatic route-based splitting
- **Image Optimization**: Next.js Image component
- **CSS Optimization**: Tailwind CSS purging

### Backend Optimization
- **Serverless Functions**: Vercel edge functions
- **Caching**: Static asset caching
- **API Timeouts**: 30-second limits
- **Memory Management**: Temporary file cleanup

### ML Model Optimization
- **Feature Selection**: Reduced feature dimensions
- **Model Compression**: Efficient serialization
- **Batch Processing**: Multiple image handling
- **GPU Acceleration**: OpenCV optimizations

---

## 🚀 Deployment Strategy

### Vercel Deployment
- **Framework**: Next.js with App Router
- **Regions**: US East (iad1) for optimal performance
- **Functions**: Serverless API routes
- **Static Assets**: CDN distribution
- **Environment**: Production-optimized

### Environment Variables
```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_PRO_API_KEY=your_gemini_pro_api_key
ML_SERVICE_URL=http://localhost:5000
NODE_ENV=production
```

### Build Process
1. **Dependencies**: npm install
2. **Type Checking**: TypeScript compilation
3. **Linting**: ESLint code quality
4. **Building**: Next.js production build
5. **Deployment**: Vercel automatic deployment

---

## 🔄 Development Workflow

### Local Development
1. **Setup**: `npm install` + `pip install -r ml/requirements.txt`
2. **Environment**: Copy `env.example` to `.env.local`
3. **ML Service**: Start Python Flask service (`python ml/diseasePredictor.py`)
4. **Frontend**: `npm run dev` (with Turbopack)
5. **Testing**: Access `http://localhost:3000`

### Code Quality
- **TypeScript**: Strict type checking
- **ESLint**: Code quality and consistency
- **Prettier**: Code formatting
- **Git Hooks**: Pre-commit validation

### Testing Strategy
- **Unit Tests**: Individual component testing
- **Integration Tests**: API endpoint testing
- **ML Tests**: Model accuracy validation
- **E2E Tests**: Complete user flow testing

---

## 📈 Performance Metrics

### Model Accuracy
- **MRI Analysis**: 78.18% (4-class classification)
- **X-ray Analysis**: 75.16% (binary classification)
- **Diabetes Prediction**: 90%+ (binary classification)

### API Performance
- **Symptom Analysis**: ~3-5 seconds (Gemini AI)
- **Image Analysis**: ~2-3 seconds (local ML)
- **Report Analysis**: ~5-8 seconds (ML + AI)
- **Doctor Search**: ~2-4 seconds (Gemini AI)

### Frontend Performance
- **First Load**: ~2-3 seconds
- **Subsequent Loads**: ~1-2 seconds
- **Image Upload**: ~1-2 seconds
- **Analysis Results**: ~3-5 seconds

---

## 🛠️ Maintenance & Updates

### Regular Maintenance
- **Dependencies**: Monthly updates
- **Security**: Security patch monitoring
- **Performance**: Regular performance audits
- **Models**: Quarterly accuracy reviews

### Update Process
1. **Testing**: Comprehensive testing in development
2. **Staging**: Deploy to staging environment
3. **Validation**: Performance and accuracy validation
4. **Production**: Deploy to production with monitoring

### Monitoring
- **Error Tracking**: Vercel error monitoring
- **Performance**: Core Web Vitals tracking
- **Usage**: API usage analytics
- **Accuracy**: Model performance monitoring

---

## 📚 Additional Resources

### Documentation
- **README.md**: Project overview and setup
- **models/README.md**: ML model documentation
- **ml/README.md**: Python service documentation
- **API Documentation**: Inline code documentation

### External Dependencies
- **Google Gemini AI**: Medical analysis and reasoning
- **Scikit-learn**: Machine learning algorithms
- **OpenCV**: Image processing
- **Next.js**: React framework
- **Vercel**: Deployment platform

### Development Tools
- **TypeScript**: Type safety and development experience
- **Tailwind CSS**: Utility-first CSS framework
- **ESLint**: Code quality and consistency
- **Git**: Version control and collaboration

---

## 🎯 Future Enhancements

### Planned Features
- **Real-time Analysis**: WebSocket integration
- **Mobile App**: React Native implementation
- **Advanced ML**: Deep learning models
- **Multi-language**: Internationalization support
- **Telemedicine**: Video consultation integration

### Technical Improvements
- **Database Integration**: Persistent data storage
- **Caching Layer**: Redis for performance
- **Microservices**: Service decomposition
- **CI/CD Pipeline**: Automated testing and deployment
- **Monitoring**: Advanced observability

---

**This documentation provides a comprehensive understanding of every file, component, and system in the AI Diagnostic Companion project. Each section can be expanded based on specific requirements or questions.**
