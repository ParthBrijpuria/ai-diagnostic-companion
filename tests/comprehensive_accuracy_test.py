#!/usr/bin/env python3
"""
Comprehensive accuracy test for both MRI and X-ray models on full test datasets
"""

import os
import cv2
import numpy as np
import logging
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_all_test_images(directory):
    """Load ALL test images from directory"""
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
        logger.info(f"Loading ALL images from {class_name}...")
        
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
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

def extract_mri_features_enhanced(images):
    """Extract enhanced features for MRI model"""
    features = []
    
    for img_array in images:
        img = img_array.reshape(64, 64)
        
        # Basic features
        basic_features = [
            np.mean(img), np.std(img), np.min(img), np.max(img), np.median(img)
        ]
        
        # Advanced statistical features
        advanced_features = [
            np.var(img), np.ptp(img), np.percentile(img, 25), np.percentile(img, 75), np.percentile(img, 90)
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
        
        # Histogram features
        hist = cv2.calcHist([img_uint8], [0], None, [16], [0, 256])
        hist_features = hist.flatten()
        
        # Edge features
        edges = cv2.Canny(img_uint8, 50, 150)
        edge_density = np.sum(edges > 0) / (64 * 64)
        
        # Pixel features
        pixel_features = img_array[:200]
        
        feature_vector = basic_features + advanced_features + texture_features + hist_features.tolist() + [edge_density] + pixel_features.tolist()
        features.append(feature_vector)
    
    return np.array(features)

def extract_xray_features_enhanced(images):
    """Extract enhanced features for X-ray model"""
    features = []
    
    for img_array in images:
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
        features.append(feature_vector)
    
    return np.array(features)

def test_mri_comprehensive():
    """Test MRI model on complete test dataset"""
    logger.info("🧠 Testing MRI Model on Complete Test Dataset...")
    logger.info("=" * 60)
    
    # Load model
    model_path = 'models/mri/mri_quick_v2_model.pkl'
    if not os.path.exists(model_path):
        logger.error(f"MRI model not found: {model_path}")
        return None
    
    model_data = joblib.load(model_path)
    model = model_data['model']
    scaler = model_data['scaler']
    feature_selector = model_data['feature_selector']
    
    # Load ALL test data
    test_dir = 'datasets/mri scan/Testing'
    logger.info("Loading ALL test images...")
    test_images, test_labels, class_names = load_all_test_images(test_dir)
    
    if len(test_images) == 0:
        logger.error("No test images found")
        return None
    
    # Extract features
    logger.info("Extracting features...")
    test_features = extract_mri_features_enhanced(test_images)
    
    # Scale features
    test_features_scaled = scaler.transform(test_features)
    
    # Select features
    test_features_selected = feature_selector.transform(test_features_scaled)
    
    # Make predictions
    predictions = model.predict(test_features_selected)
    
    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)
    
    logger.info(f"✅ MRI Model Complete Test Results:")
    logger.info(f"   Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"   Total Test Samples: {len(test_images)}")
    logger.info(f"   Classes: {class_names}")
    
    # Classification report
    print("\n📊 MRI Classification Report:")
    print(classification_report(test_labels, predictions, target_names=class_names))
    
    # Confusion matrix
    print("\n🔍 MRI Confusion Matrix:")
    cm = confusion_matrix(test_labels, predictions, labels=class_names)
    print("     ", end="")
    for class_name in class_names:
        print(f"{class_name:>12}", end="")
    print()
    for i, class_name in enumerate(class_names):
        print(f"{class_name:>5}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i][j]:>12}", end="")
        print()
    
    return accuracy

def test_xray_comprehensive():
    """Test X-ray model on complete test dataset"""
    logger.info("🫁 Testing X-ray Model on Complete Test Dataset...")
    logger.info("=" * 60)
    
    # Load model
    model_path = 'models/xray/xray_quick_model.pkl'
    if not os.path.exists(model_path):
        logger.error(f"X-ray model not found: {model_path}")
        return None
    
    model_data = joblib.load(model_path)
    model = model_data['model']
    scaler = model_data['scaler']
    feature_selector = model_data['feature_selector']
    
    # Load ALL test data
    test_dir = 'datasets/chest_xray/test'
    logger.info("Loading ALL test images...")
    test_images, test_labels, class_names = load_all_test_images(test_dir)
    
    if len(test_images) == 0:
        logger.error("No test images found")
        return None
    
    # Extract features
    logger.info("Extracting features...")
    test_features = extract_xray_features_enhanced(test_images)
    
    # Scale features
    test_features_scaled = scaler.transform(test_features)
    
    # Select features
    test_features_selected = feature_selector.transform(test_features_scaled)
    
    # Make predictions
    predictions = model.predict(test_features_selected)
    
    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)
    
    logger.info(f"✅ X-ray Model Complete Test Results:")
    logger.info(f"   Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"   Total Test Samples: {len(test_images)}")
    logger.info(f"   Classes: {class_names}")
    
    # Classification report
    print("\n📊 X-ray Classification Report:")
    print(classification_report(test_labels, predictions, target_names=class_names))
    
    # Confusion matrix
    print("\n🔍 X-ray Confusion Matrix:")
    cm = confusion_matrix(test_labels, predictions, labels=class_names)
    print("     ", end="")
    for class_name in class_names:
        print(f"{class_name:>12}", end="")
    print()
    for i, class_name in enumerate(class_names):
        print(f"{class_name:>5}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i][j]:>12}", end="")
        print()
    
    return accuracy

def main():
    """Main function to test both models comprehensively"""
    logger.info("🚀 Comprehensive Accuracy Test on Complete Test Datasets...")
    logger.info("=" * 80)
    
    results = {}
    
    # Test MRI Model
    try:
        mri_accuracy = test_mri_comprehensive()
        results['MRI'] = mri_accuracy
    except Exception as e:
        logger.error(f"MRI model test failed: {e}")
        results['MRI'] = None
    
    print("\n" + "=" * 80)
    
    # Test X-ray Model
    try:
        xray_accuracy = test_xray_comprehensive()
        results['X-ray'] = xray_accuracy
    except Exception as e:
        logger.error(f"X-ray model test failed: {e}")
        results['X-ray'] = None
    
    # Summary
    print("\n" + "=" * 80)
    print("📈 COMPREHENSIVE TEST RESULTS SUMMARY")
    print("=" * 80)
    
    for model_name, accuracy in results.items():
        if accuracy is not None:
            print(f"✅ {model_name} Model: {accuracy:.4f} ({accuracy*100:.2f}%)")
        else:
            print(f"❌ {model_name} Model: Test Failed")
    
    print("\n🎯 Model Performance:")
    print("   MRI: Enhanced Ensemble Model")
    print("   X-ray: Gradient Boosting Model")
    print("   Both models tested on complete test datasets")
    
    print("\n📊 Accuracy Comparison:")
    print("   Original MRI: 33.49% test accuracy")
    print("   Original X-ray: 61.70% test accuracy")
    
    if results['MRI'] is not None:
        mri_improvement = ((results['MRI'] - 0.3349) / 0.3349) * 100
        print(f"   MRI Improvement: {mri_improvement:+.1f}%")
    
    if results['X-ray'] is not None:
        xray_improvement = ((results['X-ray'] - 0.6170) / 0.6170) * 100
        print(f"   X-ray Improvement: {xray_improvement:+.1f}%")

if __name__ == "__main__":
    main()
