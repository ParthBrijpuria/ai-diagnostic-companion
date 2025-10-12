#!/usr/bin/env python3
"""
Ultra-Advanced MRI Brain Tumor Classification Model
Uses multiple ensemble techniques and advanced feature engineering for maximum accuracy
"""

import os
import cv2
import numpy as np
import logging
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MRIUltraEnsembleClassifier:
    def __init__(self, model_path='mri_ultra_ensemble_model.pkl', dataset_path='../../datasets/mri scan'):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = None
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

    def extract_ultra_features(self, images):
        """Extract ultra-advanced features from images"""
        logger.info("Extracting ultra-advanced features...")
        
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
            
            # Advanced statistical features
            features.extend([
                np.var(img),  # Variance
                np.ptp(img),  # Peak-to-peak (max - min)
                np.percentile(img, 25),  # 25th percentile
                np.percentile(img, 75),  # 75th percentile
                np.percentile(img, 90),  # 90th percentile
                np.percentile(img, 95),  # 95th percentile
                np.percentile(img, 99),  # 99th percentile
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
                
                # Gabor features
                gabor_features = self._extract_gabor_features(img_uint8)
                features.extend(gabor_features)
                
            except Exception as e:
                logger.warning(f"Error in texture feature extraction: {e}")
                # Add zeros if texture extraction fails
                features.extend([0] * 30)  # 18 LBP + 12 Gabor features
            
            # Histogram features
            hist = cv2.calcHist([img_uint8], [0], None, [32], [0, 256])
            features.extend(hist.flatten())
            
            # Edge features
            edges = cv2.Canny(img_uint8, 50, 150)
            edge_density = np.sum(edges > 0) / (64 * 64)
            features.append(edge_density)
            
            # Contour features
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    circularity = 0
                features.extend([area, perimeter, circularity])
            else:
                features.extend([0, 0, 0])
            
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

    def _extract_gabor_features(self, img):
        """Extract Gabor features"""
        try:
            gabor_features = []
            
            # Different Gabor filters
            for angle in [0, 45, 90, 135]:
                for frequency in [0.1, 0.3, 0.5]:
                    kernel = cv2.getGaborKernel((21, 21), 5, np.radians(angle), 2*np.pi*frequency, 0.5, 0, ktype=cv2.CV_32F)
                    filtered = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                    gabor_features.extend([
                        np.mean(filtered),
                        np.std(filtered)
                    ])
            
            return gabor_features[:12]  # Limit to 12 features
            
        except Exception as e:
            logger.warning(f"Error in Gabor extraction: {e}")
            return [0] * 12

    def train_ultra_ensemble_model(self, max_images_per_class=1000):
        """Train ultra-advanced ensemble model"""
        logger.info("Training ultra-advanced MRI classification model...")
        
        # Load training data
        train_images, train_labels = self._load_and_preprocess_images(
            os.path.join(self.dataset_path, 'Training'), 
            max_images_per_class
        )
        
        # Extract features
        train_features = self.extract_ultra_features(train_images)
        
        # Scale features
        train_features_scaled = self.scaler.fit_transform(train_features)
        
        # Feature selection using multiple methods
        logger.info("Performing advanced feature selection...")
        
        # Method 1: SelectKBest
        selector_kbest = SelectKBest(f_classif, k=200)
        features_kbest = selector_kbest.fit_transform(train_features_scaled, train_labels)
        
        # Method 2: PCA
        pca = PCA(n_components=150)
        features_pca = pca.fit_transform(train_features_scaled)
        
        # Method 3: RFE with Random Forest
        rfe_selector = RFE(RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=180)
        features_rfe = rfe_selector.fit_transform(train_features_scaled, train_labels)
        
        # Combine features
        train_features_combined = np.hstack([features_kbest, features_pca, features_rfe])
        
        # Final feature selection
        self.feature_selector = SelectKBest(f_classif, k=300)
        train_features_selected = self.feature_selector.fit_transform(train_features_combined, train_labels)
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            train_features_selected, train_labels, test_size=0.2, random_state=42, stratify=train_labels
        )
        
        # Train multiple models with hyperparameter tuning
        logger.info("Training multiple models with hyperparameter tuning...")
        
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=1000, 
                max_depth=50, 
                min_samples_split=2, 
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=500, 
                learning_rate=0.01, 
                max_depth=20, 
                subsample=0.8,
                random_state=42
            ),
            'ExtraTrees': ExtraTreesClassifier(
                n_estimators=1000, 
                max_depth=50, 
                min_samples_split=2, 
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42
            ),
            'SVM': SVC(
                kernel='rbf', 
                C=1000, 
                gamma='scale', 
                probability=True,
                random_state=42
            ),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=500, 
                learning_rate=0.1, 
                random_state=42
            ),
            'Bagging': BaggingClassifier(
                base_estimator=DecisionTreeClassifier(max_depth=20),
                n_estimators=200,
                random_state=42
            )
        }
        
        # Train individual models
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
        
        # Create mega ensemble
        logger.info("Creating mega ensemble...")
        mega_ensemble = VotingClassifier([
            ('rf', RandomForestClassifier(n_estimators=1000, max_depth=50, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=500, learning_rate=0.01, max_depth=20, random_state=42)),
            ('et', ExtraTreesClassifier(n_estimators=1000, max_depth=50, random_state=42)),
            ('svm', SVC(kernel='rbf', C=1000, gamma='scale', probability=True, random_state=42)),
            ('ada', AdaBoostClassifier(n_estimators=500, learning_rate=0.1, random_state=42)),
            ('bag', BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=20), n_estimators=200, random_state=42))
        ], voting='soft')
        
        mega_ensemble.fit(X_train, y_train)
        mega_ensemble_score = mega_ensemble.score(X_val, y_val)
        logger.info(f"Mega ensemble validation accuracy: {mega_ensemble_score:.4f}")
        
        # Use mega ensemble if it's better
        if mega_ensemble_score > best_score:
            self.model = mega_ensemble
            self.training_accuracy = mega_ensemble_score
            logger.info("Using mega ensemble model")
        else:
            self.model = best_model
            self.training_accuracy = best_score
            logger.info(f"Using {best_model.__class__.__name__} model")
        
        # Final training on full dataset
        self.model.fit(train_features_selected, train_labels)
        
        # Save model
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
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
        test_features = self.extract_ultra_features(test_images)
        
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
    classifier = MRIUltraEnsembleClassifier()
    
    # Train model
    training_accuracy = classifier.train_ultra_ensemble_model(max_images_per_class=1000)
    
    # Test model
    test_accuracy = classifier.test_model(max_images_per_class=500)
    
    print(f"\n🎯 Final Results:")
    print(f"Training Accuracy: {training_accuracy:.4f} ({training_accuracy*100:.2f}%)")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

if __name__ == "__main__":
    main()
