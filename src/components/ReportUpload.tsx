/**
 * Medical Report Upload Component - 4 Disease-Specific Segments
 * Handles file upload and displays analysis results for Heart Disease, Diabetes, MRI, and X-Ray
 */

'use client';

import React, { useState, useRef } from 'react';
import { FC } from 'react';

// Define interfaces
interface UploadedFile {
  name: string;
  size: number;
  type: string;
  file: File;
}

interface ReportAnalysis {
  disease_name: string;
  confidence: number;
  future_steps: string;
  lifestyle_changes: string;
  top_medicines: Array<{ name: string; description: string }>;
}

interface MRIAnalysis {
  predicted_class: string;
  confidence: number;
  analysis_type: string;
}

interface XRayAnalysis {
  predicted_class: string;
  confidence: number;
  analysis_type: string;
}

interface DiabetesAnalysis {
  predicted_class: string;
  confidence: number;
  analysis_type: string;
}


interface ManualValues {
  Gender: string;
  Pregnancies: string;
  Glucose: string;
  BloodPressure: string;
  SkinThickness: string;
  Insulin: string;
  BMI: string;
  DiabetesPedigreeFunction: string;
  Age: string;
}

// Custom icons
const UploadIcon: FC<{ className?: string }> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
    <polyline points="17,8 12,3 7,8"/>
    <line x1="12" y1="3" x2="12" y2="15"/>
  </svg>
);

const CheckCircleIcon: FC<{ className?: string }> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
    <polyline points="22,4 12,14.01 9,11.01"/>
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

const ClipboardIcon: FC<{ className?: string }> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <rect x="8" y="2" width="8" height="4" rx="1" ry="1"/>
    <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/>
  </svg>
);

const ChevronDownIcon: FC<{ className?: string }> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <polyline points="6,9 12,15 18,9"/>
  </svg>
);

export default function ReportUpload() {
  const [uploadedFiles, setUploadedFiles] = useState<{ [key: string]: UploadedFile }>({});
  const [isUploading, setIsUploading] = useState<{ [key: string]: boolean }>({});
  const [analysis, setAnalysis] = useState<ReportAnalysis | null>(null);
  const [mriAnalysis, setMriAnalysis] = useState<MRIAnalysis | null>(null);
  const [xrayAnalysis, setXrayAnalysis] = useState<XRayAnalysis | null>(null);
  const [diabetesAnalysis, setDiabetesAnalysis] = useState<DiabetesAnalysis | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDiabetesFormExpanded, setIsDiabetesFormExpanded] = useState(false);
  
  // Refs for scroll animation
  const mriResultsRef = useRef<HTMLDivElement>(null);
  const xrayResultsRef = useRef<HTMLDivElement>(null);
  const diabetesResultsRef = useRef<HTMLDivElement>(null);
  const generalResultsRef = useRef<HTMLDivElement>(null);
  const [manualValues, setManualValues] = useState<ManualValues>({
    Gender: '',
    Pregnancies: '',
    Glucose: '',
    BloodPressure: '',
    SkinThickness: '',
    Insulin: '',
    BMI: '',
    DiabetesPedigreeFunction: '',
    Age: ''
  });

  // Handle file upload for specific disease type
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>, modelType: string) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file type
    const allowedTypes = ['application/pdf', 'image/jpeg', 'image/png', 'image/jpg'];
    if (!allowedTypes.includes(file.type)) {
      setError('Invalid file type. Please upload a PDF or image file.');
      return;
    }

    // Clear all previous results when uploading a new file
    setAnalysis(null);
    setMriAnalysis(null);
    setXrayAnalysis(null);
    setDiabetesAnalysis(null);
    setError(null);

    // Validate file size (10MB max)
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
      setError('File too large. Please upload a file smaller than 10MB.');
      return;
    }

    setUploadedFiles(prev => ({
      ...prev,
      [modelType]: {
        name: file.name,
        size: file.size,
        type: file.type,
        file: file
      }
    }));

    setError(null);
  };

  // Handle analysis for specific disease type
  const handleAnalyze = async (modelType: string) => {
    const file = uploadedFiles[modelType];
    if (!file) {
      setError('Please upload a file first.');
      return;
    }

    setIsUploading(prev => ({ ...prev, [modelType]: true }));
    setError(null);
    setAnalysis(null);
    setMriAnalysis(null);
    setXrayAnalysis(null);
    setDiabetesAnalysis(null);

    try {
      // Use MRI-specific API for MRI analysis
      if (modelType === 'mri') {
        const formData = new FormData();
        formData.append('file', file.file);

        const response = await fetch('/api/mri-analysis', {
          method: 'POST',
          body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.error || 'MRI analysis failed');
        }

        setMriAnalysis(data);
        // Scroll to MRI results
        setTimeout(() => {
          mriResultsRef.current?.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start' 
          });
        }, 100);
      } else if (modelType === 'xray') {
        // Use X-ray-specific API for X-ray analysis
        const formData = new FormData();
        formData.append('file', file.file);

        const response = await fetch('/api/xray-analysis', {
          method: 'POST',
          body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.error || 'X-ray analysis failed');
        }

        setXrayAnalysis(data);
        // Scroll to X-ray results
        setTimeout(() => {
          xrayResultsRef.current?.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start' 
          });
        }, 100);
      } else {
        // Use general test analysis API for other types
      const formData = new FormData();
      formData.append('file', file.file);
      formData.append('modelType', modelType);

      const response = await fetch('/api/test-analysis', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Analysis failed');
      }

      setAnalysis(data);
      // Scroll to general results
      setTimeout(() => {
        generalResultsRef.current?.scrollIntoView({ 
          behavior: 'smooth', 
          block: 'start' 
        });
      }, 100);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred during analysis');
    } finally {
      setIsUploading(prev => ({ ...prev, [modelType]: false }));
    }
  };

  // Handle diabetes analysis
  const handleDiabetesAnalysis = async () => {
    setIsUploading(prev => ({ ...prev, diabetes: true }));
    setError(null);
    setAnalysis(null);
    setMriAnalysis(null);
    setXrayAnalysis(null);
    setDiabetesAnalysis(null);

    try {
      const response = await fetch('/api/diabetes-analysis', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          manualValues
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Diabetes analysis failed');
      }

      setDiabetesAnalysis(data);
      // Scroll to diabetes results
      setTimeout(() => {
        diabetesResultsRef.current?.scrollIntoView({ 
          behavior: 'smooth', 
          block: 'start' 
        });
      }, 100);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred during diabetes analysis');
    } finally {
      setIsUploading(prev => ({ ...prev, diabetes: false }));
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      {/* Header */}
      <div className="card-medical mb-8 animate-fade-in">
        <div className="flex items-center mb-8">
          <div className="w-16 h-16 bg-gradient-to-r from-blue-100 to-cyan-100 rounded-2xl flex items-center justify-center mr-6">
            <ClipboardIcon className="h-8 w-8 text-blue-600"/>
          </div>
          <div>
            <h2 className="text-lg sm:text-xl md:text-2xl lg:text-3xl font-bold text-gray-800">
              Upload Your Test Reports
            </h2>
            <p className="text-xs sm:text-sm md:text-base text-gray-600 mt-1">
              Upload relevant test reports for more accurate diagnosis
            </p>
          </div>
        </div>

        {/* 3 Disease-Specific Upload Segments */}
        <div className="space-y-6">
          {/* Top Row: MRI and X-Ray */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-6">
          
          {/* MRI Segment */}
          <div className="p-4 sm:p-6 md:p-8 bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl border-2 border-purple-200 shadow-lg hover:shadow-xl transition-all duration-300">
            <div className="flex items-center mb-4">
              <div className="w-10 h-10 bg-purple-500 rounded-lg flex items-center justify-center mr-3">
                <span className="text-white font-bold text-lg">🧠</span>
              </div>
              <h4 className="text-lg sm:text-xl md:text-2xl font-bold text-purple-900">MRI Analysis</h4>
            </div>
            <div className="space-y-3">
              <div className="border-2 border-dashed border-purple-300 rounded-lg p-6 text-center hover:border-purple-500 hover:bg-purple-25 transition-all duration-300 cursor-pointer relative">
                <input
                  type="file"
                  accept=".pdf,.jpg,.jpeg,.png"
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                  onChange={(e) => handleFileUpload(e, 'mri')}
                />
                <div className="space-y-2">
                  <UploadIcon className="h-8 w-8 text-purple-400 mx-auto" />
                  <p className="text-sm sm:text-base text-purple-600">
                    {uploadedFiles.mri ? uploadedFiles.mri.name : 'Upload MRI scan'}
                  </p>
                </div>
              </div>
              {uploadedFiles.mri && (
                <button
                  onClick={() => handleAnalyze('mri')}
                  disabled={isUploading.mri}
                  className="w-full bg-purple-500 hover:bg-purple-600 text-white font-semibold py-2 px-4 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                >
                  {isUploading.mri ? (
                    <>
                      <LoaderIcon className="h-4 w-4 animate-spin" />
                      <span>Analyzing...</span>
                    </>
                  ) : (
                    <>
                      <CheckCircleIcon className="h-4 w-4" />
                      <span>Analyze MRI</span>
                    </>
                  )}
                </button>
              )}
            </div>
          </div>

          {/* X-Ray Segment */}
          <div className="p-4 sm:p-6 md:p-8 bg-gradient-to-br from-green-50 to-green-100 rounded-xl border-2 border-green-200 shadow-lg hover:shadow-xl transition-all duration-300">
            <div className="flex items-center mb-4">
              <div className="w-10 h-10 bg-green-500 rounded-lg flex items-center justify-center mr-3">
                <span className="text-white font-bold text-lg">🫁</span>
              </div>
              <h4 className="text-lg sm:text-xl md:text-2xl font-bold text-green-900">X-Ray Analysis</h4>
            </div>
            <div className="space-y-3">
              <div className="border-2 border-dashed border-green-300 rounded-lg p-6 text-center hover:border-green-500 hover:bg-green-25 transition-all duration-300 cursor-pointer relative">
                <input
                  type="file"
                  accept=".pdf,.jpg,.jpeg,.png"
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                  onChange={(e) => handleFileUpload(e, 'xray')}
                />
                <div className="space-y-2">
                  <UploadIcon className="h-8 w-8 text-green-400 mx-auto" />
                  <p className="text-sm sm:text-base text-green-600">
                    {uploadedFiles.xray ? uploadedFiles.xray.name : 'Upload X-ray scan'}
                  </p>
                </div>
              </div>
              {uploadedFiles.xray && (
                <button
                  onClick={() => handleAnalyze('xray')}
                  disabled={isUploading.xray}
                  className="w-full bg-green-500 hover:bg-green-600 text-white font-semibold py-2 px-4 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                >
                  {isUploading.xray ? (
                    <>
                      <LoaderIcon className="h-4 w-4 animate-spin" />
                      <span>Analyzing...</span>
                    </>
                  ) : (
                    <>
                      <CheckCircleIcon className="h-4 w-4" />
                      <span>Analyze X-Ray</span>
                    </>
                  )}
                </button>
              )}
            </div>
          </div>
          </div>
          
          {/* Bottom Row: Diabetes */}
          <div className="flex justify-center">
            <div className="w-full max-w-md md:max-w-lg">
              <div className="p-4 sm:p-6 md:p-8 bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl border-2 border-blue-200 shadow-lg hover:shadow-xl transition-all duration-300">
            <div className="flex items-center mb-4">
              <div className="w-10 h-10 bg-blue-500 rounded-lg flex items-center justify-center mr-3">
                <span className="text-white font-bold text-lg">🍬</span>
              </div>
              <h4 className="text-lg sm:text-xl md:text-2xl font-bold text-blue-900">Diabetes Analysis</h4>
            </div>
            
            {/* Dropdown Toggle */}
            <div 
              className="border-2 border-dashed border-blue-300 rounded-lg p-6 text-center hover:border-blue-500 hover:bg-blue-25 transition-all duration-300 cursor-pointer"
              onClick={() => setIsDiabetesFormExpanded(!isDiabetesFormExpanded)}
            >
              <div className="flex items-center justify-center space-x-2">
                <ChevronDownIcon className={`h-6 w-6 text-blue-400 transition-transform ${isDiabetesFormExpanded ? 'rotate-180' : ''}`} />
                <p className="text-sm sm:text-base text-blue-600">
                  {isDiabetesFormExpanded ? 'Hide Parameters' : 'Enter Diabetes Parameters'}
                </p>
              </div>
              </div>
              
              {/* Diabetes Parameters Form */}
            {isDiabetesFormExpanded && (
              <div className="space-y-2 sm:space-y-3 mt-3 sm:mt-4">
                <div className="grid gap-2">
                  <div>
                    <label className="block text-xs font-medium text-blue-800 mb-1">Gender</label>
                    <div className="flex space-x-4">
                      <label className="flex items-center">
                        <input
                          type="radio"
                          name="gender"
                          value="male"
                          className="mr-2"
                          checked={manualValues.Gender === 'male'}
                          onChange={(e) => setManualValues({...manualValues, Gender: e.target.value, Pregnancies: e.target.value === 'male' ? '0' : manualValues.Pregnancies})}
                        />
                        <span className="text-sm text-blue-800">Male</span>
                      </label>
                      <label className="flex items-center">
                        <input
                          type="radio"
                          name="gender"
                          value="female"
                          className="mr-2"
                          checked={manualValues.Gender === 'female'}
                          onChange={(e) => setManualValues({...manualValues, Gender: e.target.value})}
                        />
                        <span className="text-sm text-blue-800">Female</span>
                      </label>
                    </div>
                  </div>
                  
                  {manualValues.Gender === 'female' && (
                    <div>
                      <label className="block text-xs font-medium text-blue-800 mb-1">Pregnancies</label>
                      <input
                        type="number"
                        placeholder="e.g., 2"
                        className="w-full p-2 border border-blue-300 rounded text-xs sm:text-sm text-gray-900"
                        value={manualValues.Pregnancies}
                        onChange={(e) => setManualValues({...manualValues, Pregnancies: e.target.value})}
                      />
                    </div>
                  )}
                  
                  <div>
                    <label className="block text-xs font-medium text-blue-800 mb-1">Glucose (mg/dL)</label>
                    <input
                      type="number"
                      placeholder="e.g., 100"
                      className="w-full p-2 border border-blue-300 rounded text-sm text-gray-900"
                      value={manualValues.Glucose}
                      onChange={(e) => setManualValues({...manualValues, Glucose: e.target.value})}
                    />
                  </div>
                  
                  <div>
                    <label className="block text-xs font-medium text-blue-800 mb-1">Blood Pressure (mmHg)</label>
                    <input
                      type="number"
                      placeholder="e.g., 80"
                      className="w-full p-2 border border-blue-300 rounded text-sm text-gray-900"
                      value={manualValues.BloodPressure}
                      onChange={(e) => setManualValues({...manualValues, BloodPressure: e.target.value})}
                    />
                  </div>
                  
                  <div>
                    <label className="block text-xs font-medium text-blue-800 mb-1">Skin Thickness (mm)</label>
                    <input
                      type="number"
                      placeholder="e.g., 20"
                      className="w-full p-2 border border-blue-300 rounded text-sm text-gray-900"
                      value={manualValues.SkinThickness}
                      onChange={(e) => setManualValues({...manualValues, SkinThickness: e.target.value})}
                    />
                  </div>
                  
                  <div>
                    <label className="block text-xs font-medium text-blue-800 mb-1">Insulin (mu U/ml)</label>
                    <input
                      type="number"
                      placeholder="e.g., 100"
                      className="w-full p-2 border border-blue-300 rounded text-sm text-gray-900"
                      value={manualValues.Insulin}
                      onChange={(e) => setManualValues({...manualValues, Insulin: e.target.value})}
                    />
                  </div>
                  
                  <div>
                    <label className="block text-xs font-medium text-blue-800 mb-1">BMI</label>
                    <input
                      type="number"
                      step="0.1"
                      placeholder="e.g., 25.5"
                      className="w-full p-2 border border-blue-300 rounded text-sm text-gray-900"
                      value={manualValues.BMI}
                      onChange={(e) => setManualValues({...manualValues, BMI: e.target.value})}
                    />
                  </div>
                  
                  <div>
                    <label className="block text-xs font-medium text-blue-800 mb-1">Diabetes Pedigree Function</label>
                    <input
                      type="number"
                      step="0.001"
                      placeholder="e.g., 0.627"
                      className="w-full p-2 border border-blue-300 rounded text-sm text-gray-900"
                      value={manualValues.DiabetesPedigreeFunction}
                      onChange={(e) => setManualValues({...manualValues, DiabetesPedigreeFunction: e.target.value})}
                    />
                  </div>
                  
                  <div>
                    <label className="block text-xs font-medium text-blue-800 mb-1">Age (years)</label>
                    <input
                      type="number"
                      placeholder="e.g., 50"
                      className="w-full p-2 border border-blue-300 rounded text-sm text-gray-900"
                      value={manualValues.Age}
                      onChange={(e) => setManualValues({...manualValues, Age: e.target.value})}
                    />
                </div>
              </div>
              
              <button
                  onClick={handleDiabetesAnalysis}
                  disabled={isUploading.diabetes}
                className="w-full mt-3 bg-blue-500 hover:bg-blue-600 text-white font-semibold py-2 px-3 sm:px-4 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2 text-xs sm:text-sm"
              >
                {isUploading.diabetes ? (
                  <>
                    <LoaderIcon className="h-4 w-4 animate-spin" />
                    <span>Analyzing...</span>
                  </>
                ) : (
                  <>
                    <CheckCircleIcon className="h-4 w-4" />
                      <span>Analyze Diabetes Parameters</span>
                  </>
                )}
              </button>
            </div>
            )}
              </div>
            </div>
          </div>
        </div>

      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-6 mb-8">
          <div className="flex items-start space-x-3">
            <div className="h-6 w-6 text-red-500 flex-shrink-0 mt-0.5">⚠️</div>
            <div>
              <h3 className="text-lg font-semibold text-red-800 mb-2">
                Analysis Error
              </h3>
              <p className="text-red-700">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* MRI Analysis Results */}
      {mriAnalysis && (
        <div ref={mriResultsRef} className="card-medical animate-slide-up">
          <div className="flex items-center mb-8">
            <div className="w-16 h-16 bg-gradient-to-r from-purple-100 to-violet-100 rounded-2xl flex items-center justify-center mr-6">
              <CheckCircleIcon className="h-8 w-8 text-purple-600"/>
            </div>
            <div>
              <h3 className="text-2xl sm:text-3xl font-bold text-gray-800">
                MRI Analysis Results
              </h3>
              <p className="text-gray-600 mt-1">
                Brain tumor classification analysis
              </p>
            </div>
          </div>

          {/* Main Diagnosis */}
          <div className="bg-gradient-to-r from-purple-50 to-violet-50 rounded-2xl p-8 mb-8 border border-purple-200">
            <div className="text-center">
              <h4 className="text-2xl font-bold text-gray-900 mb-4">
                Brain Tumor Classification
              </h4>
              <div className="text-4xl font-bold text-purple-700 mb-4">
                {mriAnalysis.predicted_class}
              </div>
            </div>
          </div>

          {/* Disclaimer */}
          <div className="mt-8 p-4 bg-yellow-50 border border-yellow-200 rounded-xl">
            <p className="text-yellow-800 text-sm">
              <strong>Medical Disclaimer:</strong> This AI analysis is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for proper medical evaluation.
            </p>
          </div>
        </div>
      )}

      {/* X-Ray Analysis Results */}
      {xrayAnalysis && (
        <div ref={xrayResultsRef} className="card-medical animate-slide-up">
          <div className="flex items-center mb-8">
            <div className="w-16 h-16 bg-gradient-to-r from-green-100 to-emerald-100 rounded-2xl flex items-center justify-center mr-6">
              <CheckCircleIcon className="h-8 w-8 text-green-600"/>
            </div>
            <div>
              <h3 className="text-2xl sm:text-3xl font-bold text-gray-800">
                X-Ray Analysis Results
              </h3>
              <p className="text-gray-600 mt-1">
                Chest X-ray pneumonia classification analysis
              </p>
            </div>
          </div>

          {/* Main Diagnosis */}
          <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-2xl p-8 mb-8 border border-green-200">
            <div className="text-center">
              <h4 className="text-2xl font-bold text-gray-900 mb-4">
                Chest X-Ray Classification
              </h4>
              <div className="text-4xl font-bold text-green-700 mb-4">
                {xrayAnalysis.predicted_class}
              </div>
            </div>
          </div>

          {/* Disclaimer */}
          <div className="mt-8 p-4 bg-yellow-50 border border-yellow-200 rounded-xl">
            <p className="text-yellow-800 text-sm">
              <strong>Medical Disclaimer:</strong> This AI analysis is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for proper medical evaluation.
            </p>
          </div>
        </div>
      )}

      {/* Diabetes Analysis Results */}
      {diabetesAnalysis && (
        <div ref={diabetesResultsRef} className="card-medical animate-slide-up">
          <div className="flex items-center mb-8">
            <div className="w-16 h-16 bg-gradient-to-r from-blue-100 to-cyan-100 rounded-2xl flex items-center justify-center mr-6">
              <CheckCircleIcon className="h-8 w-8 text-blue-600"/>
            </div>
            <div>
              <h3 className="text-2xl sm:text-3xl font-bold text-gray-800">
                Diabetes Analysis Results
              </h3>
              <p className="text-gray-600 mt-1">
                Diabetes prediction analysis
              </p>
            </div>
          </div>

          {/* Main Diagnosis */}
          <div className="bg-gradient-to-r from-blue-50 to-cyan-50 rounded-2xl p-8 mb-8 border border-blue-200">
            <div className="text-center">
              <h4 className="text-2xl font-bold text-gray-900 mb-4">
                Diabetes Prediction
              </h4>
              <div className="text-4xl font-bold text-blue-700 mb-4">
                {diabetesAnalysis.predicted_class}
              </div>
            </div>
          </div>

          {/* Disclaimer */}
          <div className="mt-8 p-4 bg-yellow-50 border border-yellow-200 rounded-xl">
            <p className="text-yellow-800 text-sm">
              <strong>Medical Disclaimer:</strong> This AI analysis is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for proper medical evaluation.
            </p>
          </div>
        </div>
      )}


      {/* General Analysis Results */}
      {analysis && (
        <div ref={generalResultsRef} className="card-medical animate-slide-up">
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
                {analysis.disease_name}
              </div>
              <div className="text-lg text-gray-600 mb-4">
                Confidence: {(analysis.confidence * 100).toFixed(1)}%
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3 mb-4">
                <div
                  className="bg-gradient-to-r from-green-500 to-green-600 h-3 rounded-full transition-all duration-500"
                  style={{ width: `${analysis.confidence * 100}%` }}
                />
              </div>
            </div>
          </div>

          {/* Future Steps */}
          {analysis.future_steps && (
            <div className="mb-8">
              <h4 className="text-xl font-bold text-gray-800 mb-4">
                Recommended Future Steps
              </h4>
              <div className="bg-blue-50 rounded-xl p-6 border border-blue-200">
                <p className="text-blue-800 leading-relaxed">
                  {analysis.future_steps}
                </p>
              </div>
            </div>
          )}

          {/* Top Medicines */}
          {analysis.top_medicines && analysis.top_medicines.length > 0 && (
            <div className="mb-8">
              <h4 className="text-xl font-bold text-gray-800 mb-4">
                Top Medicines (Priority Order)
              </h4>
              <div className="grid gap-4">
                {analysis.top_medicines.map((medicine: { name: string; description: string }, index: number) => (
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
          {analysis.lifestyle_changes && (
            <div className="mb-8">
              <h4 className="text-xl font-bold text-gray-800 mb-4">
                Lifestyle Changes
              </h4>
              <div className="bg-yellow-50 rounded-xl p-6 border border-yellow-200">
                <p className="text-yellow-800 leading-relaxed">
                  {analysis.lifestyle_changes}
                </p>
              </div>
            </div>
          )}

          {/* Disclaimer */}
          <div className="mt-8 p-4 bg-yellow-50 border border-yellow-200 rounded-xl">
            <p className="text-yellow-800 text-sm">
              <strong>Medical Disclaimer:</strong> This AI analysis is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for proper medical evaluation.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}