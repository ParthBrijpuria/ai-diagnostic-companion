'use client';

import React, { useState, useRef, useEffect } from 'react';
import type { FC } from 'react';

// Custom icons
const MagnifyingGlassIcon: FC<{ className?: string }> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <circle cx="11" cy="11" r="8"/>
    <path d="m21 21-4.35-4.35"/>
  </svg>
);

const LocationIcon: FC<{ className?: string }> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/>
    <circle cx="12" cy="10" r="3"/>
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

interface FindDocResponse {
  doctors: Doctor[];
  analysis_type: 'find_doctor';
}

interface FindDocProps {
  initialSymptoms?: string;
}

const FindDoc: React.FC<FindDocProps> = ({ initialSymptoms = '' }) => {
  const [symptoms, setSymptoms] = useState(initialSymptoms);
  const [city, setCity] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<FindDocResponse | null>(null);
  const [isTyping, setIsTyping] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  // Update symptoms when initialSymptoms prop changes
  useEffect(() => {
    if (initialSymptoms && initialSymptoms !== symptoms) {
      setSymptoms(initialSymptoms);
    }
  }, [initialSymptoms]);

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
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ 
          behavior: 'smooth', 
          block: 'start' 
        });
      }, 100);
    }
  }, [result]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!symptoms.trim()) {
      setError('Please describe your symptoms.');
      return;
    }

    if (!city.trim()) {
      setError('Please enter your city.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('/api/find-doctor', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symptoms: symptoms.trim(),
          city: city.trim()
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to find doctors');
      }

      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred while finding doctors');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      {/* Header */}
      <div className="card-medical mb-8 animate-fade-in">
        <div className="text-center">
          <div className="w-16 h-16 bg-gradient-to-r from-blue-100 to-cyan-100 rounded-2xl flex items-center justify-center mx-auto mb-6">
            <MagnifyingGlassIcon className="h-8 w-8 text-blue-600"/>
          </div>
          <h1 className="text-2xl sm:text-3xl font-bold text-gray-800 mb-4">
            Find a Doctor
          </h1>
          <p className="text-gray-600 text-sm sm:text-base max-w-2xl mx-auto">
            Describe your symptoms and find the best doctors in your city
          </p>
        </div>
      </div>

      {/* Input Form */}
      <div className="card-medical mb-8 animate-fade-in">
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="text-center mb-8">
            <h2 className="text-xl sm:text-2xl font-bold text-gray-800 mb-2">
              Describe Your Symptoms
            </h2>
            <p className="text-gray-600 text-sm sm:text-base">
              Provide details about your symptoms to find the right specialist
            </p>
          </div>

          {/* Symptoms Input */}
          <div className="relative">
            <label htmlFor="symptoms" className="block text-base font-semibold text-gray-700 mb-3">
              Symptoms Description
            </label>
            <textarea
              ref={textareaRef}
              id="symptoms"
              data-symptoms-input
              value={symptoms}
              onChange={(e) => {
                setSymptoms(e.target.value);
                setIsTyping(true);
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit(e);
                }
              }}
              placeholder="Describe your symptoms in detail... (e.g., chest pain, shortness of breath, fever, etc.)"
              className="w-full p-4 border border-gray-300 rounded-xl text-gray-900 placeholder-gray-500 focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none transition-all duration-200"
              rows={4}
              style={{ minHeight: '120px' }}
            />
            {isTyping && (
              <div className="absolute bottom-3 right-3 text-xs text-gray-400">
                Press Enter to search
              </div>
            )}
          </div>

          {/* City Input */}
          <div className="relative">
            <label htmlFor="city" className="block text-base font-semibold text-gray-700 mb-3">
              City/Location
            </label>
            <div className="relative">
              <LocationIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
              <input
                id="city"
                type="text"
                value={city}
                onChange={(e) => {
                  setCity(e.target.value);
                  // Prevent symptoms from being cleared
                  if (symptoms.trim() === '') {
                    // Only clear if symptoms is empty, otherwise preserve it
                    return;
                  }
                }}
                placeholder="Enter your city (e.g., Mumbai, Delhi, Bangalore)"
                className="w-full pl-12 pr-4 py-4 border border-gray-300 rounded-xl text-gray-900 placeholder-gray-500 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
              />
            </div>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={loading || !symptoms.trim() || !city.trim()}
            className="w-full bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white font-semibold py-4 px-8 rounded-xl transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-3 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
          >
            {loading ? (
              <>
                <LoaderIcon className="h-5 w-5 animate-spin" />
                <span>Finding Doctors...</span>
              </>
            ) : (
              <>
                <MagnifyingGlassIcon className="h-5 w-5" />
                <span>Find Doctors</span>
              </>
            )}
          </button>
        </form>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="card-medical mb-8 animate-fade-in">
          <div className="text-center py-12">
            <LoaderIcon className="h-16 w-16 text-blue-500 mx-auto mb-6 animate-spin" />
            <h3 className="text-xl font-bold text-gray-800 mb-4">
              Finding Best Doctors
            </h3>
            <p className="text-gray-600 text-sm sm:text-base">
              Searching for specialists in {city}...
            </p>
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-6 mb-8">
          <div className="flex items-start space-x-3">
            <div className="w-6 h-6 bg-red-500 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
              <span className="text-white text-sm font-bold">!</span>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-red-800 mb-1">Error</h3>
              <p className="text-red-700">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Results */}
      {result && (
        <div ref={resultsRef} className="card-medical animate-slide-up">
          <div className="flex items-center mb-8">
            <div className="w-16 h-16 bg-gradient-to-r from-green-100 to-emerald-100 rounded-2xl flex items-center justify-center mr-6">
              <CheckCircleIcon className="h-8 w-8 text-green-600"/>
            </div>
            <div>
              <h3 className="text-xl sm:text-2xl font-bold text-gray-800">
                Recommended Doctors
              </h3>
              <p className="text-gray-600 mt-1 text-sm sm:text-base">
                Best specialists for your symptoms in {city}
              </p>
            </div>
          </div>

          {/* Doctors List */}
          <div className="space-y-6">
            {result.doctors.map((doctor, index) => (
              <div key={index} className="bg-gradient-to-r from-blue-50 to-cyan-50 rounded-2xl p-6 border border-blue-200 hover:shadow-lg transition-all duration-200">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-3">
                      <div className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center">
                        <span className="text-white font-bold text-lg">
                          {doctor.name.charAt(0)}
                        </span>
                      </div>
                      <div>
                        <h4 className="text-xl font-bold text-gray-900">{doctor.name}</h4>
                        <p className="text-blue-600 font-semibold">{doctor.specialty}</p>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                      <div className="space-y-2">
                        <p className="text-sm text-gray-600">
                          <span className="font-semibold">Hospital:</span> {doctor.hospital}
                        </p>
                        <p className="text-sm text-gray-600">
                          <span className="font-semibold">Experience:</span> {doctor.experience}
                        </p>
                        <p className="text-sm text-gray-600">
                          <span className="font-semibold">Location:</span> {doctor.location}
                        </p>
                      </div>
                      <div className="space-y-2">
                        <p className="text-sm text-gray-600">
                          <span className="font-semibold">Rating:</span> ⭐ {doctor.rating}
                        </p>
                        <p className="text-sm text-gray-600">
                          <span className="font-semibold">Phone:</span> {doctor.phone}
                        </p>
                        <p className="text-sm text-gray-600">
                          <span className="font-semibold">Consultation Fee:</span> {doctor.consultation_fee}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Disclaimer */}
          <div className="mt-8 p-4 bg-yellow-50 border border-yellow-200 rounded-xl">
            <p className="text-yellow-800 text-sm">
              <strong>Disclaimer:</strong> This information is provided for reference purposes only. Please verify doctor details and availability before booking appointments. Always consult with healthcare professionals for medical advice.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default FindDoc;
