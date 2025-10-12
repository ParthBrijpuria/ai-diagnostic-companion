#!/usr/bin/env python3
"""
Quick MRI accuracy test with different approaches
"""

import os
import cv2
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_images_quick(directory, max_images_per_class=300):
    """Quick load images for testing"""
    images = []
    labels = []
    
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if not os.path.isdir(class_path):
            continue
            
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
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
            except:
                continue
    
    return np.array(images), np.array(labels)

def extract_features_quick(images):
    """Quick feature extraction"""
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

def train_mri_quick():
    """Quick MRI model training"""
    logger.info("🧠 Quick MRI Training...")
    
    # Load data
    train_images, train_labels = load_images_quick('datasets/mri scan/Training', 300)
    test_images, test_labels = load_images_quick('datasets/mri scan/Testing', 200)
    
    # Extract features
    train_features = extract_features_quick(train_images)
    test_features = extract_features_quick(test_images)
    
    # Scale features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    # Feature selection
    feature_selector = SelectKBest(f_classif, k=200)
    train_features_selected = feature_selector.fit_transform(train_features_scaled, train_labels)
    test_features_selected = feature_selector.transform(test_features_scaled)
    
    # Train ensemble
    rf = RandomForestClassifier(n_estimators=500, max_depth=30, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=15, random_state=42)
    et = ExtraTreesClassifier(n_estimators=500, max_depth=30, random_state=42)
    svm = SVC(kernel='rbf', C=100, gamma='scale', probability=True, random_state=42)
    ada = AdaBoostClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
    
    ensemble = VotingClassifier([
        ('rf', rf), ('gb', gb), ('et', et), ('svm', svm), ('ada', ada)
    ], voting='soft')
    
    ensemble.fit(train_features_selected, train_labels)
    
    # Test
    predictions = ensemble.predict(test_features_selected)
    accuracy = accuracy_score(test_labels, predictions)
    
    logger.info(f"✅ MRI Quick Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Save model
    model_data = {
        'model': ensemble,
        'scaler': scaler,
        'feature_selector': feature_selector,
        'classes': ['glioma', 'meningioma', 'notumor', 'pituitary'],
        'training_accuracy': accuracy
    }
    joblib.dump(model_data, 'models/mri/mri_quick_v2_model.pkl')
    
    return accuracy

def main():
    """Main function"""
    logger.info("🚀 Quick MRI Accuracy Test...")
    logger.info("=" * 50)
    
    # Train MRI model
    mri_accuracy = train_mri_quick()
    
    # Summary
    print("\n" + "=" * 50)
    print("📈 QUICK MRI TEST RESULTS")
    print("=" * 50)
    print(f"✅ MRI Enhanced Model: {mri_accuracy:.4f} ({mri_accuracy*100:.2f}%)")
    
    print("\n🎯 Comparison with Previous Models:")
    print("   Original MRI: 33.49% test accuracy")
    print("   Previous Ensemble: 66.00% test accuracy")
    
    improvement_original = ((mri_accuracy - 0.3349) / 0.3349) * 100
    improvement_previous = ((mri_accuracy - 0.6600) / 0.6600) * 100
    
    print(f"\n📊 Improvement:")
    print(f"   vs Original: {improvement_original:+.1f}% improvement")
    print(f"   vs Previous: {improvement_previous:+.1f}% improvement")
    
    if mri_accuracy > 0.66:
        print(f"\n🎉 SUCCESS: Achieved higher accuracy than previous model!")
    else:
        print(f"\n📈 Still working on improving accuracy...")

if __name__ == "__main__":
    main()
