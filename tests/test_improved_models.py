#!/usr/bin/env python3
"""
Test the improved MRI and X-ray models
"""

import os
import sys
import cv2
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_images_from_directory(directory, max_images_per_class=None):
    """Load images from directory structure"""
    images = []
    labels = []
    class_names = []
    
    if not os.path.exists(directory):
        logger.error(f"Directory not found: {directory}")
        return images, labels, class_names
    
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if not os.path.isdir(class_path):
            continue
            
        class_names.append(class_name)
        logger.info(f"Loading images from {class_name}...")
        
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if max_images_per_class:
            image_files = image_files[:max_images_per_class]
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, (64, 64))
                    img_normalized = img_resized.astype(np.float32) / 255.0
                    images.append(img_normalized.flatten())
                    labels.append(class_name)
            except Exception as e:
                logger.warning(f"Error loading {img_path}: {e}")
                continue
    
    logger.info(f"Loaded {len(images)} images total")
    return images, labels, class_names

def extract_mri_features_v2(images):
    """Extract features matching MRI model V2 expectations (218 features)"""
    features = []
    
    for img_array in images:
        img = img_array.reshape(64, 64)
        
        # Basic statistical features (5)
        basic_features = [
            np.mean(img),
            np.std(img),
            np.min(img),
            np.max(img),
            np.median(img)
        ]
        
        # Texture features (8)
        try:
            img_uint8 = (img * 255).astype(np.uint8)
            
            # Sobel gradients
            sobel_x = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            texture_features = [
                np.mean(sobel_magnitude),
                np.std(sobel_magnitude),
                np.mean(sobel_x),
                np.std(sobel_x),
                np.mean(sobel_y),
                np.std(sobel_y)
            ]
            
            # Laplacian
            laplacian = cv2.Laplacian(img_uint8, cv2.CV_64F)
            texture_features.extend([
                np.mean(laplacian),
                np.std(laplacian)
            ])
        except:
            texture_features = [0] * 8
        
        # LBP features (18) - simplified approximation
        lbp_features = [0] * 18
        
        # Original pixel values (200)
        pixel_features = img_array[:200]
        
        # Combine all features
        feature_vector = basic_features + texture_features + lbp_features + pixel_features.tolist()
        features.append(feature_vector)
    
    return np.array(features)

def extract_xray_features_v2(images):
    """Extract features matching X-ray model V2 expectations (113 features)"""
    features = []
    
    for img_array in images:
        img = img_array.reshape(64, 64)
        
        # Basic statistical features (5)
        basic_features = [
            np.mean(img),
            np.std(img),
            np.min(img),
            np.max(img),
            np.median(img)
        ]
        
        # Texture features (8)
        try:
            img_uint8 = (img * 255).astype(np.uint8)
            
            # Sobel gradients
            sobel_x = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            texture_features = [
                np.mean(sobel_magnitude),
                np.std(sobel_magnitude),
                np.mean(sobel_x),
                np.std(sobel_x),
                np.mean(sobel_y),
                np.std(sobel_y)
            ]
            
            # Laplacian
            laplacian = cv2.Laplacian(img_uint8, cv2.CV_64F)
            texture_features.extend([
                np.mean(laplacian),
                np.std(laplacian)
            ])
        except:
            texture_features = [0] * 8
        
        # Original pixel values (100)
        pixel_features = img_array[:100]
        
        # Combine all features
        feature_vector = basic_features + texture_features + pixel_features.tolist()
        features.append(feature_vector)
    
    return np.array(features)

def test_mri_model_v2():
    """Test MRI model V2 accuracy"""
    logger.info("🧠 Testing MRI Model V2 Accuracy...")
    logger.info("=" * 50)
    
    # Load model
    model_path = 'models/mri/mri_improved_v2_model.pkl'
    if not os.path.exists(model_path):
        logger.error(f"MRI model V2 not found: {model_path}")
        return None
    
    model_data = joblib.load(model_path)
    model = model_data['model']
    scaler = model_data['scaler']
    
    # Load test data
    test_dir = 'datasets/mri scan/Testing'
    if not os.path.exists(test_dir):
        logger.error(f"Test directory not found: {test_dir}")
        return None
    
    logger.info("Loading test images...")
    test_images, test_labels, class_names = load_images_from_directory(test_dir)
    
    if len(test_images) == 0:
        logger.error("No test images found")
        return None
    
    # Extract features
    logger.info("Extracting features...")
    test_features = extract_mri_features_v2(test_images)
    
    # Scale features
    test_features_scaled = scaler.transform(test_features)
    
    # Make predictions
    predictions = model.predict(test_features_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)
    
    logger.info(f"✅ MRI Model V2 Test Results:")
    logger.info(f"   Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"   Test Samples: {len(test_images)}")
    logger.info(f"   Classes: {class_names}")
    
    # Classification report
    print("\n📊 Classification Report:")
    print(classification_report(test_labels, predictions, target_names=class_names))
    
    return accuracy

def test_xray_model_v2():
    """Test X-ray model V2 accuracy"""
    logger.info("🫁 Testing X-ray Model V2 Accuracy...")
    logger.info("=" * 50)
    
    # Load model
    model_path = 'models/xray/xray_improved_v2_model.pkl'
    if not os.path.exists(model_path):
        logger.error(f"X-ray model V2 not found: {model_path}")
        return None
    
    model_data = joblib.load(model_path)
    model = model_data['model']
    scaler = model_data['scaler']
    
    # Load test data
    test_dir = 'datasets/chest_xray/test'
    if not os.path.exists(test_dir):
        logger.error(f"Test directory not found: {test_dir}")
        return None
    
    logger.info("Loading test images...")
    test_images, test_labels, class_names = load_images_from_directory(test_dir)
    
    if len(test_images) == 0:
        logger.error("No test images found")
        return None
    
    # Extract features
    logger.info("Extracting features...")
    test_features = extract_xray_features_v2(test_images)
    
    # Scale features
    test_features_scaled = scaler.transform(test_features)
    
    # Make predictions
    predictions = model.predict(test_features_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)
    
    logger.info(f"✅ X-ray Model V2 Test Results:")
    logger.info(f"   Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"   Test Samples: {len(test_images)}")
    logger.info(f"   Classes: {class_names}")
    
    # Classification report
    print("\n📊 Classification Report:")
    print(classification_report(test_labels, predictions, target_names=class_names))
    
    return accuracy

def main():
    """Main function to test improved models"""
    logger.info("🚀 Testing Improved Models...")
    logger.info("=" * 60)
    
    results = {}
    
    # Test MRI Model V2
    try:
        mri_accuracy = test_mri_model_v2()
        results['MRI V2'] = mri_accuracy
    except Exception as e:
        logger.error(f"MRI model V2 test failed: {e}")
        results['MRI V2'] = None
    
    print("\n" + "=" * 60)
    
    # Test X-ray Model V2
    try:
        xray_accuracy = test_xray_model_v2()
        results['X-ray V2'] = xray_accuracy
    except Exception as e:
        logger.error(f"X-ray model V2 test failed: {e}")
        results['X-ray V2'] = None
    
    # Summary
    print("\n" + "=" * 60)
    print("📈 IMPROVED MODELS RESULTS SUMMARY")
    print("=" * 60)
    
    for model_name, accuracy in results.items():
        if accuracy is not None:
            print(f"✅ {model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
        else:
            print(f"❌ {model_name}: Test Failed")
    
    print("\n🎯 Comparison with Original Models:")
    print("   Original MRI: 33.49% test accuracy")
    print("   Original X-ray: 61.70% test accuracy")
    print("   Improved models should show better performance")

if __name__ == "__main__":
    main()
