#!/usr/bin/env python3
"""
Enhanced X-ray prediction using the gradient boosting model
"""

import sys
import cv2
import numpy as np
import joblib
import json

def extract_features_quick(img_array):
    """Quick feature extraction matching the quick model"""
    img = img_array.reshape(64, 64)
    
    # Basic features
    basic_features = [
        np.mean(img), np.std(img), np.min(img), np.max(img), np.median(img)
    ]
    
    # Texture features
    try:
        img_uint8 = (img * 255).astype(np.uint8)
        sobel_x = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        texture_features = [
            np.mean(sobel_magnitude), np.std(sobel_magnitude),
            np.mean(sobel_x), np.std(sobel_x),
            np.mean(sobel_y), np.std(sobel_y)
        ]
        
        laplacian = cv2.Laplacian(img_uint8, cv2.CV_64F)
        texture_features.extend([np.mean(laplacian), np.std(laplacian)])
    except:
        texture_features = [0] * 8
    
    # Pixel features
    pixel_features = img_array[:100]
    
    feature_vector = basic_features + texture_features + pixel_features.tolist()
    return np.array(feature_vector)

def predict_xray_image(image_path):
    """Predict X-ray image class using the gradient boosting model"""
    try:
        # Load model
        model_data = joblib.load('models/xray/xray_quick_model.pkl')
        model = model_data['model']
        scaler = model_data['scaler']
        feature_selector = model_data['feature_selector']
        classes = model_data['classes']
        
        # Load and preprocess image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"error": "Could not load image"}
        
        img_resized = cv2.resize(img, (64, 64))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_flat = img_normalized.flatten()
        
        # Extract features
        features = extract_features_quick(img_flat)
        
        # Scale and select features
        features_scaled = scaler.transform(features.reshape(1, -1))
        features_selected = feature_selector.transform(features_scaled)
        
        # Make prediction
        prediction = model.predict(features_selected)[0]
        probabilities = model.predict_proba(features_selected)[0]
        
        # Get confidence
        confidence = max(probabilities)
        
        # Get class name
        class_idx = np.argmax(probabilities)
        predicted_class = classes[class_idx]
        
        result = {
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "probabilities": {
                classes[i]: float(prob) for i, prob in enumerate(probabilities)
            }
        }
        
        return result
        
    except Exception as e:
        return {"error": str(e)}

def main():
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: python xray_enhanced_predict.py <image_path>"}))
        sys.exit(1)
    
    image_path = sys.argv[1]
    result = predict_xray_image(image_path)
    print(json.dumps(result))

if __name__ == "__main__":
    main()
