#!/usr/bin/env python3
"""
Improved MRI Brain Tumor Classification Model V2
Fixes feature extraction to match original model and improves accuracy
"""

import os
import cv2
import numpy as np
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedMRIClassifierV2:
    def __init__(self, model_path='mri_improved_v2_model.pkl', dataset_path='datasets/mri scan'):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
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
        """Extract enhanced features from images - matching original model exactly"""
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
                
                # Additional texture features
                # Local Binary Pattern approximation
                lbp_features = self._extract_lbp_features(img_uint8)
                features.extend(lbp_features)
                
            except Exception as e:
                logger.warning(f"Error in texture feature extraction: {e}")
                # Add zeros if texture extraction fails
                features.extend([0] * 18)  # 18 LBP features
            
            # Add original pixel values (first 200 pixels for better representation)
            features.extend(img_flat[:200])
            
            enhanced_features.append(features)
        
        return np.array(enhanced_features)

    def _extract_lbp_features(self, img):
        """Extract Local Binary Pattern features"""
        try:
            # Simple LBP approximation
            lbp_img = np.zeros_like(img)
            
            for i in range(1, img.shape[0] - 1):
                for j in range(1, img.shape[1] - 1):
                    center = img[i, j]
                    binary_string = ""
                    
                    # Check 8 neighbors
                    neighbors = [
                        img[i-1, j-1], img[i-1, j], img[i-1, j+1],
                        img[i, j+1], img[i+1, j+1], img[i+1, j],
                        img[i+1, j-1], img[i, j-1]
                    ]
                    
                    for neighbor in neighbors:
                        binary_string += "1" if neighbor >= center else "0"
                    
                    # Convert binary to decimal
                    lbp_img[i, j] = int(binary_string, 2)
            
            # Calculate LBP histogram features
            hist = cv2.calcHist([lbp_img], [0], None, [18], [0, 256])
            return hist.flatten().tolist()
            
        except Exception as e:
            logger.warning(f"Error in LBP extraction: {e}")
            return [0] * 18

    def train_improved_model(self, max_images_per_class=1000):
        """Train improved model with better algorithms"""
        logger.info("Training improved MRI classification model...")
        
        # Load training data
        train_images, train_labels = self._load_and_preprocess_images(
            os.path.join(self.dataset_path, 'Training'), 
            max_images_per_class
        )
        
        # Extract features
        train_features = self.extract_enhanced_features(train_images)
        
        # Scale features
        train_features_scaled = self.scaler.fit_transform(train_features)
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            train_features_scaled, train_labels, test_size=0.2, random_state=42, stratify=train_labels
        )
        
        # Train multiple models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=10, random_state=42),
            'ExtraTrees': ExtraTreesClassifier(n_estimators=200, max_depth=20, random_state=42),
            'SVM': SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
        }
        
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            logger.info(f"{name} validation accuracy: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model = model
        
        # Create ensemble model
        logger.info("Creating ensemble model...")
        ensemble = VotingClassifier([
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=10, random_state=42)),
            ('et', ExtraTreesClassifier(n_estimators=200, max_depth=20, random_state=42)),
            ('svm', SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42))
        ], voting='soft')
        
        ensemble.fit(X_train, y_train)
        ensemble_score = ensemble.score(X_val, y_val)
        logger.info(f"Ensemble validation accuracy: {ensemble_score:.4f}")
        
        # Use ensemble if it's better
        if ensemble_score > best_score:
            self.model = ensemble
            self.training_accuracy = ensemble_score
            logger.info("Using ensemble model")
        else:
            self.model = best_model
            self.training_accuracy = best_score
            logger.info(f"Using {best_model.__class__.__name__} model")
        
        # Final training on full dataset
        self.model.fit(train_features_scaled, train_labels)
        
        # Save model
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'classes': self.classes,
            'image_size': self.image_size,
            'training_accuracy': self.training_accuracy
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
            os.path.join(self.dataset_path, 'Testing'), 
            max_images_per_class
        )
        
        # Extract features
        test_features = self.extract_enhanced_features(test_images)
        
        # Scale features
        test_features_scaled = self.scaler.transform(test_features)
        
        # Make predictions
        predictions = self.model.predict(test_features_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(test_labels, predictions)
        
        logger.info(f"Test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(test_labels, predictions, target_names=self.classes))
        
        return accuracy

def main():
    """Main function"""
    classifier = ImprovedMRIClassifierV2()
    
    # Train model
    training_accuracy = classifier.train_improved_model(max_images_per_class=1000)
    
    # Test model
    test_accuracy = classifier.test_model(max_images_per_class=500)
    
    print(f"\n🎯 Final Results:")
    print(f"Training Accuracy: {training_accuracy:.4f} ({training_accuracy*100:.2f}%)")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

if __name__ == "__main__":
    main()
