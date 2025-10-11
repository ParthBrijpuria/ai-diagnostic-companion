'use client';

import { useState, useRef, useEffect } from 'react';
import type { FC } from 'react';
import ReportUpload from '@/components/ReportUpload';
import FindDoc from '@/components/FindDoc';

// --- Type Definitions for API Response ---
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
  next_step: string;
}

// --- SVG Icon Components ---
const HeartPlusIcon: FC<{ className?: string }> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.29 1.51 4.04 3 5.5l7 7Z"/>
    <path d="M12 8v8"/>
    <path d="M8 12h8"/>
  </svg>
);

const MagnifyingGlassIcon: FC<{ className?: string }> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <circle cx="11" cy="11" r="8"/>
    <path d="m21 21-4.35-4.35"/>
  </svg>
);

const StethoscopeIcon: FC<{ className?: string }> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M6 8a6 6 0 0 1 12 0v8a6 6 0 0 1-12 0V8Z"/>
    <path d="M6 8v8"/>
    <path d="M18 8v8"/>
    <path d="M6 12h12"/>
    <path d="M12 2v4"/>
    <circle cx="12" cy="6" r="1"/>
  </svg>
);

const TestTubeIcon: FC<{ className?: string }> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M14.5 2H9.5L8.5 3H15.5L14.5 2Z"/>
    <path d="M10 3V13.5C10 15.4 8.4 17 6.5 17S3 15.4 3 13.5V3"/>
    <path d="M21 3V13.5C21 15.4 19.4 17 17.5 17S14 15.4 14 13.5V3"/>
    <path d="M7 3H17"/>
  </svg>
);

const ClipboardIcon: FC<{ className?: string }> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/>
    <rect x="8" y="2" width="8" height="4" rx="1" ry="1"/>
  </svg>
);

const AlertTriangleIcon: FC<{ className?: string }> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
    <line x1="12" y1="9" x2="12" y2="13"/>
    <line x1="12" y1="17" x2="12.01" y2="17"/>
  </svg>
);

const LoaderIcon: FC<{ className?: string }> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <line x1="12" y1="2" x2="12" y2="6"/>
    <line x1="12" y1="18" x2="12" y2="22"/>
    <line x1="4.93" y1="4.93" x2="7.76" y2="7.76"/>
    <line x1="16.24" y1="16.24" x2="19.07" y2="19.07"/>
    <line x1="2" y1="12" x2="6" y2="12"/>
    <line x1="18" y1="12" x2="22" y2="12"/>
    <line x1="4.93" y1="19.07" x2="7.76" y2="16.24"/>
    <line x1="16.24" y1="7.76" x2="19.07" y2="4.93"/>
  </svg>
);

const ForwardIcon: FC<{ className?: string }> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <polyline points="9,18 15,12 9,6"/>
  </svg>
);

// const UploadIcon: FC<{ className?: string }> = ({ className }) => (
//   <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
//     <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
//     <polyline points="7,10 12,15 17,10"/>
//     <line x1="12" y1="15" x2="12" y2="3"/>
//   </svg>
// );

const CheckCircleIcon: FC<{ className?: string }> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
    <polyline points="22,4 12,14.01 9,11.01"/>
  </svg>
);

// --- Loading Skeleton Components ---
// const SkeletonCard: FC<{ className?: string }> = ({ className }) => (
//   <div className={`animate-pulse bg-gray-200 rounded-2xl ${className}`}></div>
// );

// const SkeletonText: FC<{ width?: string; className?: string }> = ({ width = "w-full", className }) => (
//   <div className={`h-4 bg-gray-200 rounded ${width} ${className}`}></div>
// );

// --- Main Page Component ---
export default function DiagnosticPage() {
  const [symptoms, setSymptoms] = useState('');
  const [duration, setDuration] = useState('');
  const [severity, setSeverity] = useState('Normal');
  const [result, setResult] = useState<ApiResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  // const [isTyping, setIsTyping] = useState(false);
  const [activeTab, setActiveTab] = useState<'symptoms' | 'reports' | 'finddoc'>('symptoms');
  const [storedSymptoms, setStoredSymptoms] = useState('');
  const [manualValues] = useState({
    bloodPressure: '',
    bloodSugar: '',
    cholesterol: '',
    bmi: ''
  });
  const [testAnalysis, setTestAnalysis] = useState<{ disease_name: string; confidence: number; future_steps: string; top_medicines: Array<{ name: string; description: string }>; lifestyle_changes: string; analysis_type: 'file' | 'manual'; extracted_values?: Record<string, string | number> } | null>(null);
  const [testLoading, setTestLoading] = useState(false);
  const [overallUploadedFile, setOverallUploadedFile] = useState<File | null>(null);
  // const [overallTestLoading, setOverallTestLoading] = useState(false);
  const [overallTestAnalysis, setOverallTestAnalysis] = useState<{ predicted_disease: string; confidence: number; key_indicators: string[]; extracted_metrics: Record<string, string | number>; analysis_summary: string } | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [symptoms]);

  // Scroll to results when they appear
  useEffect(() => {
    if (result && resultsRef.current) {
      resultsRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [result]);

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (symptoms.trim().length === 0) {
      setError('Please describe your symptoms before submitting.');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);
    // setIsTyping(false);

    try {
      const response = await fetch('/api/diagnose', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          symptoms: symptoms.trim(),
          duration: duration.trim() || 'Not specified',
          severity: severity
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Analysis failed');
      }

      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred during analysis');
    } finally {
      setLoading(false);
    }
  };

  // Handle test file upload
  // const handleTestUpload = async (e: React.ChangeEvent<HTMLInputElement>, modelType: string) => {
  //   const file = e.target.files?.[0];
  //   if (!file) return;

  //   setTestLoading(true);
  //   setError('');

  //   try {
  //     const formData = new FormData();
  //     formData.append('file', file);
  //     formData.append('symptoms', symptoms);
  //     formData.append('duration', duration);
  //     formData.append('modelType', modelType);

  //     const response = await fetch('/api/test-analysis', {
  //       method: 'POST',
  //       body: formData,
  //     });

  //     const data = await response.json();

  //     if (!response.ok) {
  //       throw new Error(data.error || 'Test analysis failed');
  //     }

  //     setTestAnalysis(data);
  //   } catch (err) {
  //     setError(err instanceof Error ? err.message : 'An error occurred during test analysis');
  //   } finally {
  //     setTestLoading(false);
  //   }
  // };

  // Handle manual values submission
  // const handleManualSubmit = async () => {
  //   setTestLoading(true);
  //   setError('');

  //   try {
  //     const response = await fetch('/api/test-analysis', {
  //       method: 'POST',
  //       headers: {
  //         'Content-Type': 'application/json',
  //       },
  //       body: JSON.stringify({
  //         manualValues,
  //         symptoms,
  //         duration
  //       }),
  //     });

  //     const data = await response.json();

  //     if (!response.ok) {
  //       throw new Error(data.error || 'Manual analysis failed');
  //     }

  //     setTestAnalysis(data);
  //   } catch (err) {
  //     setError(err instanceof Error ? err.message : 'An error occurred during manual analysis');
  //   } finally {
  //     setTestLoading(false);
  //   }
  // };

  // Get probability color styling
  const getProbabilityColor = (probability: string) => {
    switch (probability.toLowerCase()) {
      case 'high':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'low':
        return 'bg-green-100 text-green-800 border-green-200';
      default:
        return 'bg-blue-100 text-blue-800 border-blue-200';
    }
  };

  // Handle overall test file upload
  // const handleOverallTestUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
  //   const file = e.target.files?.[0];
  //   if (!file) return;

  //   // Validate file type
  //   const allowedTypes = ['application/pdf', 'image/jpeg', 'image/png', 'image/jpg'];
  //   if (!allowedTypes.includes(file.type)) {
  //     setError('Invalid file type. Please upload a PDF or image file.');
  //     return;
  //   }

  //   // Validate file size (10MB max)
  //   const maxSize = 10 * 1024 * 1024;
  //   if (file.size > maxSize) {
  //     setError('File too large. Please upload a file smaller than 10MB.');
  //     return;
  //   }

  //   setOverallUploadedFile(file);
  //   setError('');
  // };

  // Handle overall test analysis
  // const handleOverallAnalysis = async () => {
  //   if (!overallUploadedFile) return;

  //   setOverallTestLoading(true);
  //   setError('');

  //   try {
  //     const formData = new FormData();
  //     formData.append('file', overallUploadedFile);
  //     formData.append('symptoms', symptoms);
  //     formData.append('duration', duration);

  //     const response = await fetch('/api/test-analysis', {
  //       method: 'POST',
  //       body: formData,
  //     });

  //     const data = await response.json();

  //     if (!response.ok) {
  //       throw new Error(data.error || 'Test analysis failed');
  //     }

  //     setOverallTestAnalysis(data);
  //   } catch (err) {
  //     setError(err instanceof Error ? err.message : 'An error occurred during test analysis');
  //   } finally {
  //     setOverallTestLoading(false);
  //   }
  // };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
          {/* Header */}
      <header className="bg-white/90 backdrop-blur-md shadow-lg border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center">
            <div className="flex items-center justify-center mb-4">
              <div className="w-16 h-16 bg-gradient-to-r from-medical-500 to-primary-600 rounded-2xl flex items-center justify-center shadow-lg">
                <HeartPlusIcon className="h-8 w-8 text-white"/>
              </div>
            </div>
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-gradient tracking-tight leading-tight pb-2">
              AI Diagnostic Companion
            </h1>
            <p className="mt-4 text-xl sm:text-2xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
              Your intelligent guide to preliminary health insights
            </p>
            <div className="mt-6 flex items-center justify-center space-x-4 text-sm text-gray-500">
              <div className="flex items-center">
                <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                HIPAA Compliant
              </div>
              <div className="flex items-center">
                <div className="w-2 h-2 bg-blue-500 rounded-full mr-2"></div>
                AI-Powered Analysis
              </div>
              <div className="flex items-center">
                <div className="w-2 h-2 bg-purple-500 rounded-full mr-2"></div>
                Instant Results
              </div>
            </div>
          </div>
        </div>
          </header>

      {/* Tab Navigation */}
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <div className="flex space-x-1 bg-gray-100 p-1 rounded-xl">
          <button
            onClick={() => setActiveTab('symptoms')}
            className={`flex-1 py-3 px-6 rounded-lg font-medium transition-all duration-200 ${
              activeTab === 'symptoms'
                ? 'bg-white text-medical-600 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            <div className="flex items-center justify-center space-x-2">
              <StethoscopeIcon className="h-5 w-5" />
              <span>Symptom Analysis</span>
            </div>
          </button>
          <button
            onClick={() => setActiveTab('reports')}
            className={`flex-1 py-3 px-6 rounded-lg font-medium transition-all duration-200 ${
              activeTab === 'reports'
                ? 'bg-white text-medical-600 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            <div className="flex items-center justify-center space-x-2">
              <ClipboardIcon className="h-5 w-5" />
              <span>Report Upload</span>
            </div>
          </button>
          <button
            onClick={() => setActiveTab('finddoc')}
            className={`flex-1 py-3 px-6 rounded-lg font-medium transition-all duration-200 ${
              activeTab === 'finddoc'
                ? 'bg-white text-medical-600 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            <div className="flex items-center justify-center space-x-2">
              <MagnifyingGlassIcon className="h-5 w-5" />
              <span>Find a Doc</span>
            </div>
          </button>
        </div>
      </div>

      <main className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8 lg:py-12">
        {activeTab === 'symptoms' ? (
          <>
            {/* Input Form Card */}
            <div className="card-medical mb-8 animate-fade-in">
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="text-center mb-8">
                  <h2 className="text-2xl sm:text-3xl font-bold text-gray-800 mb-2">
                    Describe Your Symptoms
                  </h2>
                  <p className="text-gray-600">
                    Provide detailed information about your symptoms for the most accurate analysis
                  </p>
                </div>
                
                <div className="relative">
                  <label htmlFor="symptoms" className="block text-lg font-semibold text-gray-700 mb-3">
                    Symptom Description
              </label>
              <textarea
                    ref={textareaRef}
                id="symptoms"
                value={symptoms}
                    onChange={(e) => {
                      setSymptoms(e.target.value);
                      // setIsTyping(true);
                    }}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        handleSubmit(e as React.FormEvent);
                      }
                    }}
                    placeholder="Describe your symptoms in detail... (e.g., 'I have been experiencing chest pain for 3 days, shortness of breath, and fatigue. The pain is sharp and worsens with deep breathing.')"
                    className="input-medical resize-none"
                    rows={4}
                    maxLength={2000}
                    disabled={loading}
                  />
                  <div className="flex justify-between items-center mt-2">
                    <div className="text-sm text-gray-500">
                      {symptoms.length}/2000 characters
                    </div>
                    <div className="text-sm text-gray-500">
                      Press Enter to analyze
                    </div>
                  </div>
                </div>

                {/* Severity Bar */}
                <div className="relative">
                  <label htmlFor="severity" className="block text-lg font-semibold text-gray-700 mb-3">
                    Severity Level
                  </label>
                  <div className="space-y-3">
                    <div className="relative">
                      <input
                        type="range"
                        min="0"
                        max="4"
                        step="1"
                        value={severity === 'Normal' ? 0 : severity === 'Mild' ? 1 : severity === 'Noticeable' ? 2 : severity === 'Serious' ? 3 : 4}
                        onChange={(e) => {
                          const levels = ['Normal', 'Mild', 'Noticeable', 'Serious', 'Critical'];
                          setSeverity(levels[parseInt(e.target.value)]);
                        }}
                        className="w-full h-3 bg-gradient-to-r from-green-500 via-yellow-500 via-orange-500 to-red-500 rounded-lg appearance-none cursor-pointer"
                        style={{
                          background: 'linear-gradient(to right, #10b981 0%, #f59e0b 25%, #f97316 50%, #ef4444 75%, #dc2626 100%)'
                        }}
                      />
                      <div className="flex justify-between text-xs text-gray-500 mt-1">
                        <span className="text-green-600 font-medium">Normal</span>
                        <span className="text-yellow-600 font-medium">Mild</span>
                        <span className="text-orange-600 font-medium">Noticeable</span>
                        <span className="text-red-500 font-medium">Serious</span>
                        <span className="text-red-700 font-medium">Critical</span>
                      </div>
                    </div>
                    <div className="text-center">
                      <span className={`inline-block px-4 py-2 rounded-full text-sm font-semibold ${
                        severity === 'Normal' ? 'bg-green-100 text-green-800' :
                        severity === 'Mild' ? 'bg-yellow-100 text-yellow-800' :
                        severity === 'Noticeable' ? 'bg-orange-100 text-orange-800' :
                        severity === 'Serious' ? 'bg-red-100 text-red-800' :
                        'bg-red-200 text-red-900'
                      }`}>
                        Selected: {severity}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Duration Input */}
                <div className="relative">
                  <label htmlFor="duration" className="block text-lg font-semibold text-gray-700 mb-3">
                    Duration of Symptoms (optional)
                  </label>
                  <input
                    id="duration"
                    type="text"
                    value={duration}
                    onChange={(e) => setDuration(e.target.value)}
                    placeholder="e.g., &apos;3 days&apos;, &apos;2 weeks&apos;, &apos;1 month&apos;, &apos;6 months&apos;"
                    className="input-medical"
                disabled={loading}
              />
                  <div className="text-sm text-gray-500 mt-2">
                    Specify how long you've been experiencing these symptoms
                  </div>
                </div>

                <div className="flex justify-center">
                <button
                  type="submit"
                    disabled={loading || symptoms.trim().length === 0}
                    className="btn-primary flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {loading ? (
                      <>
                        <LoaderIcon className="h-5 w-5 animate-spin" />
                        <span>Analyzing...</span>
                      </>
                    ) : (
                      <>
                        <StethoscopeIcon className="h-5 w-5" />
                        <span>Analyze Symptoms</span>
                      </>
                    )}
                </button>
              </div>

                <div className="text-center text-sm text-gray-500 bg-blue-50 p-4 rounded-xl border border-blue-200">
                  <p className="font-medium text-blue-800 mb-1">Medical Disclaimer</p>
                  <p className="text-blue-700">
                    This analysis is for informational purposes only and should not replace professional medical advice.
                  </p>
              </div>
            </form>
          </div>

            {/* Loading State */}
            {loading && (
              <div className="card-medical mb-8 animate-fade-in">
                <div className="text-center py-12">
                  <LoaderIcon className="h-16 w-16 text-medical-500 mx-auto mb-6 animate-spin" />
                  <h3 className="text-2xl font-bold text-gray-800 mb-4">
                    Analyzing Your Symptoms
                  </h3>
                  <p className="text-gray-600 text-lg">
                    Our AI is processing your information...
                  </p>
                </div>
                
                {/* Skeleton Loading Cards */}
                <div className="space-y-8">
                  {/* Possible Conditions Skeleton */}
                  <div className="animate-pulse">
                    <div className="flex items-center mb-6">
                      <div className="w-16 h-16 bg-gray-200 rounded-2xl mr-6"></div>
                      <div>
                        <div className="h-8 bg-gray-200 rounded w-64 mb-2"></div>
                        <div className="h-4 bg-gray-200 rounded w-48"></div>
                      </div>
                    </div>
                    <div className="grid gap-6">
                      {[1, 2, 3].map((i) => (
                        <div key={i} className="p-6 bg-gray-100 rounded-2xl">
                          <div className="h-6 bg-gray-200 rounded w-3/4 mb-3"></div>
                          <div className="h-4 bg-gray-200 rounded w-full mb-2"></div>
                          <div className="h-4 bg-gray-200 rounded w-2/3"></div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Recommended Tests Skeleton */}
                  <div className="animate-pulse">
                    <div className="flex items-center mb-6">
                      <div className="w-16 h-16 bg-gray-200 rounded-2xl mr-6"></div>
                      <div>
                        <div className="h-8 bg-gray-200 rounded w-56 mb-2"></div>
                        <div className="h-4 bg-gray-200 rounded w-40"></div>
                      </div>
                    </div>
                    <div className="grid gap-6">
                      {[1, 2].map((i) => (
                        <div key={i} className="p-6 bg-gray-100 rounded-2xl">
                          <div className="h-6 bg-gray-200 rounded w-2/3 mb-3"></div>
                          <div className="h-4 bg-gray-200 rounded w-full"></div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Next Step Skeleton */}
                  <div className="animate-pulse">
                    <div className="flex items-center mb-6">
                      <div className="w-16 h-16 bg-gray-200 rounded-2xl mr-6"></div>
                      <div>
                        <div className="h-8 bg-gray-200 rounded w-48 mb-2"></div>
                        <div className="h-4 bg-gray-200 rounded w-36"></div>
                      </div>
                    </div>
                    <div className="p-8 bg-gray-100 rounded-2xl">
                      <div className="h-6 bg-gray-200 rounded w-full mb-2"></div>
                      <div className="h-6 bg-gray-200 rounded w-3/4"></div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {/* Error Display */}
            {error && (
              <div className="card-medical mb-8 animate-fade-in">
                <div className="flex items-start space-x-4 p-6 bg-red-50 border border-red-200 rounded-xl">
                  <AlertTriangleIcon className="h-6 w-6 text-red-500 flex-shrink-0 mt-1" />
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-red-800 mb-2">
                      Analysis Error
                    </h3>
                    <p className="text-red-700 mb-4">{error}</p>
                    <button
                      onClick={() => setError('')}
                      className="text-red-600 hover:text-red-800 font-medium"
                    >
                      Dismiss
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Results Display */}
            {result && !loading && (
              <div ref={resultsRef} className="space-y-8 animate-fade-in">
                {/* Possible Conditions */}
                <div className="card-medical animate-slide-up">
                  <div className="flex items-center mb-8">
                    <div className="w-16 h-16 bg-gradient-to-r from-red-100 to-pink-100 rounded-2xl flex items-center justify-center mr-6">
                      <AlertTriangleIcon className="h-8 w-8 text-red-600"/>
                    </div>
                    <div>
                      <h3 className="text-2xl sm:text-3xl font-bold text-gray-800">
                    Possible Conditions
                      </h3>
                      <p className="text-gray-600 mt-1">
                        Conditions that may be associated with your symptoms
                      </p>
                    </div>
                  </div>
                  <div className="grid gap-6">
                    {result.possible_diseases.slice(0, 5).map((disease, index) => (
                      <div key={disease.name} className="group p-6 bg-gradient-to-r from-gray-50 to-blue-50 rounded-2xl border border-gray-200 hover:shadow-lg transition-all duration-300 hover:border-medical-200">
                        <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between mb-4">
                          <h4 className="font-bold text-xl text-gray-900 mb-2 lg:mb-0 group-hover:text-medical-700 transition-colors">
                            {disease.name}
                          </h4>
                          <span className={`px-3 py-1 rounded-full text-sm font-medium border ${getProbabilityColor(disease.probability)}`}>
                            {disease.probability} Probability
                          </span>
                        </div>
                        <p className="text-gray-700 leading-relaxed text-lg">{disease.why}</p>
                      </div>
                    ))}
                  </div>
                </div>
                
                {/* Recommended Tests */}
                <div className="card-medical animate-slide-up">
                  <div className="flex items-center mb-8">
                    <div className="w-16 h-16 bg-gradient-to-r from-green-100 to-emerald-100 rounded-2xl flex items-center justify-center mr-6">
                      <TestTubeIcon className="h-8 w-8 text-green-600"/>
                    </div>
                    <div>
                      <h3 className="text-2xl sm:text-3xl font-bold text-gray-800">
                    Recommended Tests
                      </h3>
                      <p className="text-gray-600 mt-1">
                        Diagnostic tests that may help identify the condition
                      </p>
                    </div>
                  </div>
                  <div className="grid gap-6">
                    {result.recommended_tests.map((test, index) => (
                      <div key={test.test} className="group p-6 bg-gradient-to-r from-green-50 to-emerald-50 rounded-2xl border border-green-200 hover:shadow-lg transition-all duration-300 hover:border-green-300">
                        <h4 className="font-bold text-xl text-gray-900 mb-3 group-hover:text-green-700 transition-colors">
                          {test.test}
                        </h4>
                        <p className="text-gray-700 leading-relaxed text-lg">{test.why}</p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Next Step */}
                <div className="card-medical animate-slide-up">
                  <div className="flex items-center mb-8">
                    <div className="w-16 h-16 bg-gradient-to-r from-purple-100 to-indigo-100 rounded-2xl flex items-center justify-center mr-6">
                      <ClipboardIcon className="h-8 w-8 text-purple-600"/>
                    </div>
                    <div>
                      <h3 className="text-2xl sm:text-3xl font-bold text-gray-800">
                        Recommended Next Steps
                      </h3>
                      <p className="text-gray-600 mt-1">
                        What you should do based on this analysis
                      </p>
                    </div>
                  </div>
                  <div className="p-8 bg-gradient-to-r from-purple-50 via-indigo-50 to-blue-50 rounded-2xl border border-purple-200">
                    <div className="flex items-start">
                      <div className="w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center mr-4 flex-shrink-0 mt-1">
                        <ForwardIcon className="h-4 w-4 text-white"/>
                      </div>
                      <p className="text-xl text-gray-800 leading-relaxed font-medium">
                        {result.next_step}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Find a Doc Button */}
                <div className="card-medical animate-slide-up">
                  <div className="text-center">
                    <div className="w-16 h-16 bg-gradient-to-r from-blue-100 to-cyan-100 rounded-2xl flex items-center justify-center mx-auto mb-6">
                      <StethoscopeIcon className="h-8 w-8 text-blue-600"/>
                    </div>
                    <h3 className="text-2xl sm:text-3xl font-bold text-gray-800 mb-4">
                      Need a Doctor?
                    </h3>
                    <p className="text-gray-600 mb-6">
                      Find the best specialists for your symptoms in your city
                    </p>
                    <button
                      onClick={() => {
                        // Store symptoms before switching tabs
                        setStoredSymptoms(symptoms);
                        setActiveTab('finddoc');
                      }}
                      className="btn-primary flex items-center space-x-2 mx-auto"
                    >
                      <MagnifyingGlassIcon className="h-5 w-5" />
                      <span>Find a Doctor</span>
                    </button>
                  </div>
                </div>

                {/* Test Analysis Results */}
                {testAnalysis && (
                  <div className="card-medical animate-slide-up">
                    <div className="flex items-center mb-8">
                      <div className="w-16 h-16 bg-gradient-to-r from-green-100 to-emerald-100 rounded-2xl flex items-center justify-center mr-6">
                        <CheckCircleIcon className="h-8 w-8 text-green-600"/>
                      </div>
                      <div>
                        <h3 className="text-2xl sm:text-3xl font-bold text-gray-800">
                          Test Analysis Results
                        </h3>
                        <p className="text-gray-600 mt-1">
                          AI-powered analysis of your test results
                        </p>
                      </div>
                    </div>

                    {/* Main Diagnosis */}
                    <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-2xl p-8 mb-8 border border-green-200">
                      <div className="text-center">
                        <h4 className="text-2xl font-bold text-gray-900 mb-4">
                          Final Diagnosis
                        </h4>
                        <div className="text-3xl font-bold text-green-700 mb-2">
                          {testAnalysis.disease_name}
                        </div>
                        <div className="text-lg text-gray-600 mb-4">
                          Confidence: {(testAnalysis.confidence * 100).toFixed(1)}%
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-3 mb-4">
                          <div
                            className="bg-gradient-to-r from-green-500 to-green-600 h-3 rounded-full transition-all duration-500"
                            style={{ width: `${testAnalysis.confidence * 100}%` }}
                          />
                        </div>
                      </div>
                    </div>

                    {/* Future Steps */}
                    {testAnalysis.future_steps && (
                      <div className="mb-8">
                        <h4 className="text-xl font-bold text-gray-800 mb-4">
                          Recommended Future Steps
                        </h4>
                        <div className="bg-blue-50 rounded-xl p-6 border border-blue-200">
                          <p className="text-blue-800 leading-relaxed">
                            {testAnalysis.future_steps}
                          </p>
                        </div>
                      </div>
                    )}

                    {/* Top Medicines */}
                    {testAnalysis.top_medicines && testAnalysis.top_medicines.length > 0 && (
                      <div className="mb-8">
                        <h4 className="text-xl font-bold text-gray-800 mb-4">
                          Top Medicines (Priority Order)
                        </h4>
                        <div className="grid gap-4">
                          {testAnalysis.top_medicines.map((medicine: { name: string; description: string }, index: number) => (
                            <div key={index} className="flex items-center p-4 bg-purple-50 rounded-xl border border-purple-200">
                              <div className="w-8 h-8 bg-purple-500 text-white rounded-full flex items-center justify-center text-sm font-bold mr-4">
                                {index + 1}
                              </div>
                              <div>
                                <div className="font-semibold text-purple-900">{medicine.name}</div>
                                <div className="text-sm text-purple-700">{medicine.description}</div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Lifestyle Changes */}
                    {testAnalysis.lifestyle_changes && (
                      <div className="mb-8">
                        <h4 className="text-xl font-bold text-gray-800 mb-4">
                          Lifestyle Changes
                        </h4>
                        <div className="bg-yellow-50 rounded-xl p-6 border border-yellow-200">
                          <p className="text-yellow-800 leading-relaxed">
                            {testAnalysis.lifestyle_changes}
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Test Loading State */}
                {testLoading && (
                  <div className="card-medical mb-8 animate-fade-in">
                    <div className="text-center py-12">
                      <LoaderIcon className="h-16 w-16 text-medical-500 mx-auto mb-6 animate-spin" />
                      <h3 className="text-2xl font-bold text-gray-800 mb-4">
                        Analyzing Test Results
                      </h3>
                      <p className="text-gray-600 text-lg">
                        Our AI is processing your test data...
                      </p>
                    </div>
                  </div>
                )}

                {/* Disclaimer */}
                <div className="bg-gradient-to-r from-amber-50 to-yellow-50 border border-amber-200 rounded-2xl p-6 animate-slide-up">
                  <div className="flex items-start">
                    <AlertTriangleIcon className="w-6 h-6 text-amber-600 mr-3 mt-1 flex-shrink-0"/>
                    <div>
                      <h4 className="font-bold text-lg text-amber-800 mb-2">Important Disclaimer</h4>
                      <p className="text-amber-700 leading-relaxed">
                        This analysis is for informational purposes only and should not be considered as medical advice. 
                        Always consult with a qualified healthcare professional for proper diagnosis and treatment. 
                        In case of emergency, call 112 (Medical Emergency) or 108 (Ambulance) immediately.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Overall Test Analysis Results */}
            {overallTestAnalysis && (
              <div className="card-medical animate-slide-up">
                <div className="flex items-center mb-8">
                  <div className="w-16 h-16 bg-gradient-to-r from-purple-100 to-pink-100 rounded-2xl flex items-center justify-center mr-6">
                    <CheckCircleIcon className="h-8 w-8 text-purple-600"/>
                  </div>
                  <div>
                    <h3 className="text-2xl sm:text-3xl font-bold text-gray-800">
                      Test Report Analysis
                    </h3>
                    <p className="text-gray-600 mt-1">
                      AI-powered analysis of your uploaded test report
                    </p>
                  </div>
                </div>

                {/* Main Diagnosis */}
                <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-2xl p-8 mb-8 border border-purple-200">
                  <div className="text-center">
                    <h4 className="text-2xl font-bold text-gray-900 mb-4">
                      Analysis Result
                    </h4>
                    <div className="text-3xl font-bold text-purple-700 mb-2">
                      {overallTestAnalysis.disease_name}
                    </div>
                    <div className="text-lg text-gray-600 mb-4">
                      Confidence: {(overallTestAnalysis.confidence * 100).toFixed(1)}%
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-3 mb-4">
                      <div
                        className="bg-gradient-to-r from-purple-500 to-pink-500 h-3 rounded-full transition-all duration-500"
                        style={{ width: `${overallTestAnalysis.confidence * 100}%` }}
                      />
                    </div>
                  </div>
                </div>

                {/* Criticality Assessment */}
                <div className="mb-8">
                  <h4 className="text-xl font-bold text-gray-800 mb-4">
                    Criticality Assessment
                  </h4>
                  <div className="bg-red-50 rounded-xl p-6 border border-red-200">
                    <p className="text-red-800 leading-relaxed">
                      {overallTestAnalysis.future_steps || "Please consult a healthcare professional for proper evaluation."}
                    </p>
                  </div>
          </div>
          
                {/* Lifestyle Changes */}
                {overallTestAnalysis.lifestyle_changes && (
                  <div className="mb-8">
                    <h4 className="text-xl font-bold text-gray-800 mb-4">
                      Recommended Lifestyle Changes
                    </h4>
                    <div className="bg-yellow-50 rounded-xl p-6 border border-yellow-200">
                      <p className="text-yellow-800 leading-relaxed">
                        {overallTestAnalysis.lifestyle_changes}
                      </p>
                    </div>
                  </div>
                )}

                {/* Top Medicines */}
                {overallTestAnalysis.top_medicines && overallTestAnalysis.top_medicines.length > 0 && (
                  <div className="mb-8">
                    <h4 className="text-xl font-bold text-gray-800 mb-4">
                      Recommended Medications
                    </h4>
                    <div className="grid gap-4">
                      {overallTestAnalysis.top_medicines.map((medicine: { name: string; description: string }, index: number) => (
                        <div key={index} className="flex items-center p-4 bg-blue-50 rounded-xl border border-blue-200">
                          <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-bold mr-4">
                            {index + 1}
                          </div>
                          <div>
                            <div className="font-semibold text-blue-900">{medicine.name}</div>
                            <div className="text-sm text-blue-700">{medicine.description}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
        </div>
            )}
          </>
        ) : activeTab === 'reports' ? (
          <ReportUpload />
        ) : (
          <FindDoc initialSymptoms={storedSymptoms} />
        )}
      </main>
    </div>
  );
}