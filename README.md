# AI Diagnostic Companion 🏥💖

A comprehensive AI-powered medical diagnostic platform that combines machine learning models with advanced AI analysis to provide accurate health assessments and medical recommendations.

![AI Diagnostic Companion](https://img.shields.io/badge/AI-Medical%20Diagnostic-blue?style=for-the-badge&logo=heart)
![Next.js](https://img.shields.io/badge/Next.js-15.5.4-black?style=for-the-badge&logo=next.js)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue?style=for-the-badge&logo=typescript)

## 🌟 Features

### 🔍 Symptom Analysis
- **AI-Powered Diagnosis**: Advanced Gemini AI analyzes symptoms and provides detailed medical insights
- **Severity Assessment**: 5-level severity bar (Normal → Mild → Noticeable → Serious → Critical)
- **Comprehensive Reports**: Detailed disease analysis with probabilities, reasoning, and recommended tests
- **Emergency Integration**: Indian emergency numbers (112 for medical emergency, 108 for ambulance)

### 📊 Report Upload & Analysis
- **MRI Brain Scan Analysis**: Local machine learning model for tumor classification (Glioma, Meningioma, Pituitary, No Tumor)
- **Chest X-Ray Analysis**: Pneumonia detection using GradientBoosting classifier (Normal vs Pneumonic)
- **Diabetes Prediction**: Health parameter analysis using RandomForest classifier
- **General Report Analysis**: AI-powered analysis of uploaded medical reports (PDF, JPEG, PNG)

### 👨‍⚕️ Find a Doctor
- **Location-Based Search**: Find specialized doctors in your city
- **Symptom Integration**: Automatic symptom transfer from analysis to doctor search
- **AI Recommendations**: Gemini AI provides doctor recommendations based on symptoms and location

## 🏗️ Architecture

### Frontend
- **Next.js 15.5.4** with App Router
- **TypeScript** for type safety
- **Tailwind CSS** for responsive design
- **React Components** with modern hooks

### Backend
- **Next.js API Routes** for serverless functions
- **Python ML Models** for local analysis
- **Google Gemini AI** for advanced analysis
- **File Upload Handling** with temporary file management

### Machine Learning Models
- **MRI Classification**: Ensemble model (RandomForest, GradientBoosting, ExtraTrees, SVM, KNN, LogisticRegression)
- **X-Ray Analysis**: GradientBoosting classifier for pneumonia detection
- **Diabetes Prediction**: RandomForest classifier for health parameter analysis

## 📁 Project Structure

```
ai-diagnostic-companion/
├── src/
│   ├── app/                    # Next.js App Router
│   │   ├── api/               # API routes
│   │   │   ├── diagnose/      # Symptom analysis
│   │   │   ├── mri-analysis/ # MRI model
│   │   │   ├── xray-analysis/ # X-Ray model
│   │   │   ├── diabetes-analysis/ # Diabetes model
│   │   │   └── find-doctor/   # Doctor search
│   │   ├── globals.css        # Global styles
│   │   ├── layout.tsx         # Root layout
│   │   └── page.tsx           # Main page
│   ├── components/            # React components
│   │   ├── ReportUpload.tsx   # Report upload interface
│   │   └── FindDoc.tsx        # Doctor search interface
│   └── services/               # Service modules
│       ├── mlAnalysis.ts      # ML analysis service
│       ├── mlService.ts        # ML service wrapper
│       └── reportParser.ts     # Report parsing service
├── models/                     # Machine Learning Models
│   ├── mri/                   # MRI analysis models
│   ├── xray/                  # X-Ray analysis models
│   └── diabetes/              # Diabetes prediction models
├── datasets/                   # Training datasets
├── public/                     # Static assets
└── README.md                   # Project documentation
```

## 🚀 Getting Started

### Prerequisites
- **Node.js** 18.0 or later
- **Python** 3.8 or later
- **npm** or **yarn** package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ParthBrijpuria/ai-diagnostic-companion.git
   cd ai-diagnostic-companion
   ```

2. **Install Node.js dependencies**
```bash
   npm install
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r ml/requirements.txt
   ```
   
4. **Set up environment variables**
   ```bash
   cp env.example .env.local
   ```
   Add your API keys to `.env.local`:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_PRO_API_KEY=your_gemini_pro_api_key_here
   ```

5. **Start the development server**
   ```bash
   npm run dev
   ```
   
6. **Open your browser**
   Navigate to [http://localhost:3000](http://localhost:3000)

## 🔧 Configuration

### Environment Variables
- `GEMINI_API_KEY`: Google Gemini API key for general analysis
- `PYTHON_EXECUTABLE`: Python executable path (optional, defaults to 'python')

### Model Configuration
- Models are automatically loaded from the `models/` directory
- Training datasets are located in the `datasets/` directory
- Model files (`.pkl`) are generated during training

## 📊 Model Performance

### MRI Brain Scan Analysis
- **Overall Accuracy**: Nearly 80%(on test set)
- **Categories**: Glioma, Meningioma, Pituitary, No Tumor
- **Model**: Ensemble classifier with multiple algorithms

### Chest X-Ray Analysis
- **Accuracy**: Nearly 80%(on test set)
- **Categories**: Normal, Pneumonic
- **Model**: GradientBoosting classifier

### Diabetes Prediction
- **Accuracy**: 90%+
- **Categories**: Diabetic, Non-Diabetic
- **Model**: RandomForest classifier

## 🌐 API Endpoints

### Symptom Analysis
```http
POST /api/diagnose
Content-Type: application/json

{
  "symptoms": "stomach pain and vomiting",
  "duration": "4 days",
  "severity": "Critical"
}
```

### MRI Analysis
```http
POST /api/mri-analysis
Content-Type: multipart/form-data

file: [MRI image file]
```

### X-Ray Analysis
```http
POST /api/xray-analysis
Content-Type: multipart/form-data

file: [X-Ray image file]
```

### Diabetes Analysis
```http
POST /api/diabetes-analysis
Content-Type: application/json

{
  "manualValues": {
    "Gender": "male",
    "Pregnancies": "0",
    "Glucose": "100",
    "BloodPressure": "80",
    "SkinThickness": "20",
    "Insulin": "50",
    "BMI": "25",
    "DiabetesPedigreeFunction": "0.5",
    "Age": "30"
  }
}
```

### Find Doctor
```http
POST /api/find-doctor
Content-Type: application/json

{
  "symptoms": "chest pain",
  "city": "Mumbai"
}
```

## 🎯 Use Cases

### For Patients
- **Initial Symptom Assessment**: Get preliminary medical insights
- **Report Analysis**: Upload medical reports for AI analysis
- **Doctor Recommendations**: Find specialized doctors in your area
- **Health Monitoring**: Track health parameters over time

### For Healthcare Providers
- **Diagnostic Support**: AI-assisted diagnosis and recommendations
- **Report Interpretation**: Automated analysis of medical reports
- **Patient Triage**: Severity-based patient prioritization
- **Continuing Education**: Access to AI-powered medical insights

## 🔒 Privacy & Security

- **Local Processing**: ML models run locally without external API calls
- **Data Privacy**: No patient data is stored or transmitted
- **Secure Uploads**: Temporary files are automatically deleted
- **API Security**: Environment variables protect API keys

## 🚨 Medical Disclaimer

**Important**: This AI Diagnostic Companion is for informational purposes only and should not be considered as medical advice. Always consult with qualified healthcare professionals for proper diagnosis and treatment. The application is designed to assist, not replace, professional medical judgment.

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines
- Follow TypeScript best practices
- Write comprehensive tests
- Update documentation for new features
- Ensure mobile responsiveness

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

- **Parth Brijpuria**
- **Moksh Palaresha**
- **Paragi Agarwal**
- **Nehee Parekh**

## 🙏 Acknowledgments

- **Google Gemini AI** for advanced medical analysis
- **Scikit-learn** for machine learning algorithms
- **Next.js Team** for the excellent framework
- **Open Source Community** for various libraries and tools

## 📞 Support

For support, email [support@ai-diagnostic-companion.com](mailto:support@ai-diagnostic-companion.com) or open an issue on GitHub.

## 🔗 Links

- **Live Demo**: [https://ai-diagnostic-companion.vercel.app](https://ai-diagnostic-companion.vercel.app)(however some features may not work in the demo live site)
- **Documentation**: [https://github.com/ParthBrijpuria/ai-diagnostic-companion/wiki](https://github.com/ParthBrijpuria/ai-diagnostic-companion/wiki)
- **Issues**: [https://github.com/ParthBrijpuria/ai-diagnostic-companion/issues](https://github.com/ParthBrijpuria/ai-diagnostic-companion/issues)

---

**Made with ❤️ for better healthcare accessibility**
