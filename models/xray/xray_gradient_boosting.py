#!/usr/bin/env python3
"""
X-ray Pneumonia Classification Model using Gradient Boosting
Optimized for maximum accuracy
"""

import os
import cv2
import numpy as np
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class XRayGradientBoostingClassifier:
    def __init__(self, model_path='xray_gradient_boosting_model.pkl', dataset_path='../../datasets/chest_xray'):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.image_size = (64, 64)
        self.training_accuracy = 0.0

    def _load_and_preprocess_images(self, data_dir, max_images_per_class=None):
        """Load and preprocess images from the dataset"""
        logger.info(f"Loading images from {data_dir}")
        
        images = []
        labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if max_images_per_class:
                image_files = image_files[:max_images_per_class]
            
            logger.info(f"Processing {len(image_files)} images from {class_name}")
            
            for filename in image_files:
                try:
                    img_path = os.path.join(class_dir, filename)
                    img = cv2.imread(img_path)
                    
                    if img is None:
                        continue
                    
                    # Convert to grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Resize
                    img = cv2.resize(img, self.image_size)
                    
                    # Normalize
                    img = img.astype(np.float32) / 255.0
                    
                    # Flatten
                    img_flat = img.flatten()
                    
                    images.append(img_flat)
                    labels.append(class_name)
                    
                except Exception as e:
                    logger.warning(f"Error processing {filename}: {e}")
                    continue
        
        logger.info(f"Loaded {len(images)} images total")
        return np.array(images), np.array(labels)

    def extract_enhanced_features(self, images):
        """Extract enhanced features from images"""
        logger.info("Extracting enhanced features...")
        
        enhanced_features = []
        
        for img_flat in images:
            # Reshape back to image
            img = img_flat.reshape(self.image_size)
            
            # Basic statistical features
            features = []
            
            # Mean, std, min, max, median
            features.extend([
                np.mean(img),
                np.std(img),
                np.min(img),
                np.max(img),
                np.median(img)
            ])
            
            # Texture features using Sobel and Laplacian
            try:
                # Convert to uint8 for OpenCV operations
                img_uint8 = (img * 255).astype(np.uint8)
                
                # Sobel gradients
                sobel_x = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
                sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
                
                features.extend([
                    np.mean(sobel_magnitude),
                    np.std(sobel_magnitude),
                    np.mean(sobel_x),
                    np.std(sobel_x),
                    np.mean(sobel_y),
                    np.std(sobel_y)
                ])
                
                # Laplacian
                laplacian = cv2.Laplacian(img_uint8, cv2.CV_64F)
                features.extend([
                    np.mean(laplacian),
                    np.std(laplacian)
                ])
                
            except Exception as e:
                logger.warning(f"Error in texture feature extraction: {e}")
                # Add zeros if texture extraction fails
                features.extend([0] * 8)
            
            # Add original pixel values (first 100 pixels for efficiency)
            features.extend(img_flat[:100])
            
            enhanced_features.append(features)
        
        return np.array(enhanced_features)

    def train_gradient_boosting_model(self, max_images_per_class=1500):
        """Train Gradient Boosting model with hyperparameter tuning"""
        logger.info("Training Gradient Boosting X-ray classification model...")
        
        # Load training data
        train_images, train_labels = self._load_and_preprocess_images(
            os.path.join(self.dataset_path, 'train'), 
            max_images_per_class
        )
        
        # Extract features
        train_features = self.extract_enhanced_features(train_images)
        
        # Scale features
        train_features_scaled = self.scaler.fit_transform(train_features)
        
        # Feature selection
        self.feature_selector = SelectKBest(f_classif, k=80)
        train_features_selected = self.feature_selector.fit_transform(train_features_scaled, train_labels)
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            train_features_selected, train_labels, test_size=0.2, random_state=42, stratify=train_labels
        )
        
        # Hyperparameter tuning for Gradient Boosting
        param_grid = {
            'n_estimators': [200, 300, 400],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        # Base Gradient Boosting model
        base_model = GradientBoostingClassifier(random_state=42)
        
        # Grid search for best parameters
        logger.info("Performing hyperparameter tuning...")
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=3, 
            scoring='accuracy', 
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        self.model = grid_search.best_estimator_
        self.training_accuracy = grid_search.best_score_
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Final validation
        val_score = self.model.score(X_val, y_val)
        logger.info(f"Validation accuracy: {val_score:.4f}")
        
        # Final training on full dataset
        self.model.fit(train_features_selected, train_labels)
        
        # Save model
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'classes': self.classes,
            'image_size': self.image_size,
            'training_accuracy': self.training_accuracy,
            'best_params': grid_search.best_params_
        }
        
        joblib.dump(model_data, self.model_path)
        logger.info(f"Model saved to {self.model_path}")
        logger.info(f"Training accuracy: {self.training_accuracy:.4f} ({self.training_accuracy*100:.2f}%)")
        
        return self.training_accuracy

    def test_model(self, max_images_per_class=500):
        """Test model on test dataset"""
        logger.info("Testing model...")
        
        # Load test data
        test_images, test_labels = self._load_and_preprocess_images(
            os.path.join(self.dataset_path, 'test'), 
            max_images_per_class
        )
        
        # Extract features
        test_features = self.extract_enhanced_features(test_images)
        
        # Scale features
        test_features_scaled = self.scaler.transform(test_features)
        
        # Select features
        test_features_selected = self.feature_selector.transform(test_features_scaled)
        
        # Make predictions
        predictions = self.model.predict(test_features_selected)
        
        # Calculate accuracy
        accuracy = accuracy_score(test_labels, predictions)
        
        logger.info(f"Test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(test_labels, predictions, target_names=self.classes))
        
        return accuracy

def main():
    """Main function"""
    classifier = XRayGradientBoostingClassifier()
    
    # Train model
    training_accuracy = classifier.train_gradient_boosting_model(max_images_per_class=1500)
    
    # Test model
    test_accuracy = classifier.test_model(max_images_per_class=500)
    
    print(f"\n🎯 Final Results:")
    print(f"Training Accuracy: {training_accuracy:.4f} ({training_accuracy*100:.2f}%)")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

if __name__ == "__main__":
    main()
