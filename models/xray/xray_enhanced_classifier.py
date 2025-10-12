#!/usr/bin/env python3
"""
Enhanced X-Ray Pneumonia Classification Model
Uses advanced ensemble methods and deep learning features for maximum accuracy
"""

import os
import cv2
import numpy as np
import logging
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedXRayClassifier:
    def __init__(self, model_path='xray_enhanced_model.pkl', dataset_path='datasets/chest_xray'):
        self.model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.image_size = (128, 128)  # Increased size for better features
        self.training_accuracy = 0.0
        self.test_accuracy = 0.0

    def _load_and_preprocess_images(self, data_dir, max_images_per_class=None):
        """Load and preprocess images with enhanced features"""
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
            
            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                try:
                    # Load image
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    # Convert to grayscale
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Resize to larger size for better features
                    resized = cv2.resize(gray, self.image_size)
                    
                    # Normalize
                    normalized = resized.astype(np.float32) / 255.0
                    
                    images.append(normalized)
                    labels.append(class_name)
                    
                except Exception as e:
                    logger.warning(f"Error processing {img_path}: {e}")
                    continue
        
        logger.info(f"Loaded {len(images)} images total")
        return np.array(images), np.array(labels)

    def _extract_enhanced_features(self, images):
        """Extract comprehensive features including CNN-like features"""
        logger.info("Extracting enhanced features...")
        
        features = []
        
        for img in images:
            feature_vector = []
            
            # 1. Raw pixel features (flattened)
            feature_vector.extend(img.flatten())
            
            # 2. Statistical features
            feature_vector.extend([
                np.mean(img),
                np.std(img),
                np.min(img),
                np.max(img),
                np.median(img),
                np.var(img),
                np.percentile(img, 25),
                np.percentile(img, 75)
            ])
            
            # 3. Texture features using Local Binary Patterns
            try:
                from skimage.feature import local_binary_pattern
                lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
                feature_vector.extend([
                    np.mean(lbp),
                    np.std(lbp),
                    np.histogram(lbp, bins=10)[0].tolist()
                ])
            except ImportError:
                # Fallback if skimage not available
                feature_vector.extend([0, 0] + [0] * 10)
            
            # 4. Edge features
            edges = cv2.Canny((img * 255).astype(np.uint8), 50, 150)
            feature_vector.extend([
                np.sum(edges > 0),
                np.mean(edges),
                np.std(edges)
            ])
            
            # 5. Gradient features
            grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            feature_vector.extend([
                np.mean(gradient_magnitude),
                np.std(gradient_magnitude),
                np.max(gradient_magnitude)
            ])
            
            # 6. Histogram features
            hist = cv2.calcHist([img], [0], None, [32], [0, 1])
            feature_vector.extend(hist.flatten())
            
            # 7. Lung-specific features
            # Detect lung contours
            blurred = cv2.GaussianBlur((img * 255).astype(np.uint8), (5, 5), 0)
            thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Largest contour area (likely lungs)
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                feature_vector.extend([area, perimeter])
            else:
                feature_vector.extend([0, 0])
            
            # 8. Frequency domain features
            f_transform = np.fft.fft2(img)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            feature_vector.extend([
                np.mean(magnitude_spectrum),
                np.std(magnitude_spectrum),
                np.max(magnitude_spectrum)
            ])
            
            features.append(feature_vector)
        
        return np.array(features)

    def train_enhanced_model(self, max_images_per_class=1500):
        """Train enhanced ensemble model"""
        logger.info("Training enhanced X-ray classification model...")
        
        # Load training data
        train_images, train_labels = self._load_and_preprocess_images(
            os.path.join(self.dataset_path, 'train'), max_images_per_class
        )
        
        if len(train_images) == 0:
            raise ValueError("No training images found")
        
        # Extract features
        train_features = self._extract_enhanced_features(train_images)
        
        # Scale features
        train_features_scaled = self.scaler.fit_transform(train_features)
        
        # Apply PCA for dimensionality reduction
        train_features_pca = self.pca.fit_transform(train_features_scaled)
        
        logger.info(f"Feature dimensions: {train_features_pca.shape}")
        
        # Create ensemble of best performing models
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=25,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=10,
            random_state=42
        )
        
        et = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=25,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        svm = SVC(
            kernel='rbf',
            C=100,
            gamma='scale',
            probability=True,
            random_state=42
        )
        
        # Create voting classifier
        self.model = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('gb', gb),
                ('et', et),
                ('svm', svm)
            ],
            voting='soft'
        )
        
        # Train the ensemble
        self.model.fit(train_features_pca, train_labels)
        
        # Calculate training accuracy
        train_pred = self.model.predict(train_features_pca)
        self.training_accuracy = accuracy_score(train_labels, train_pred)
        
        logger.info(f"Enhanced model training completed")
        logger.info(f"Training Accuracy: {self.training_accuracy:.4f} ({self.training_accuracy*100:.2f}%)")
        
        return self.training_accuracy

    def evaluate(self, test_dir):
        """Evaluate model on test set"""
        logger.info("Evaluating enhanced model on test set...")
        
        # Load test data
        test_images, test_labels = self._load_and_preprocess_images(test_dir)
        
        if len(test_images) == 0:
            raise ValueError("No test images found")
        
        # Extract features
        test_features = self._extract_enhanced_features(test_images)
        
        # Scale and transform features
        test_features_scaled = self.scaler.transform(test_features)
        test_features_pca = self.pca.transform(test_features_scaled)
        
        # Make predictions
        predictions = self.model.predict(test_features_pca)
        
        # Calculate accuracy
        accuracy = accuracy_score(test_labels, predictions)
        self.test_accuracy = accuracy
        
        # Calculate class-wise accuracy
        class_accuracy = {}
        for class_name in self.classes:
            class_mask = test_labels == class_name
            if np.sum(class_mask) > 0:
                class_true = test_labels[class_mask]
                class_pred = predictions[class_mask]
                class_acc = accuracy_score(class_true, class_pred)
                class_accuracy[class_name] = class_acc
        
        logger.info(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info("Class-wise Test Results:")
        for cls, acc in class_accuracy.items():
            logger.info(f"  {cls:<12}: {acc:.4f} ({acc*100:.2f}%)")
        
        # Print confusion matrix
        cm = confusion_matrix(test_labels, predictions, labels=self.classes)
        logger.info("Confusion Matrix:")
        logger.info(f"  {'':<12} {' '.join(f'{cls:<12}' for cls in self.classes)}")
        for i, cls in enumerate(self.classes):
            logger.info(f"  {cls:<12} {' '.join(f'{cm[i,j]:<12}' for j in range(len(self.classes)))}")
        
        return accuracy, class_accuracy

    def predict(self, image_path):
        """Predict single image"""
        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, self.image_size)
            normalized = resized.astype(np.float32) / 255.0
            
            # Extract features
            features = self._extract_enhanced_features([normalized])
            
            # Scale and transform
            features_scaled = self.scaler.transform(features)
            features_pca = self.pca.transform(features_scaled)
            
            # Predict
            prediction = self.model.predict(features_pca)[0]
            probabilities = self.model.predict_proba(features_pca)[0]
            confidence = np.max(probabilities)
            
            return {
                'predicted_class': prediction,
                'confidence': confidence,
                'probabilities': dict(zip(self.classes, probabilities))
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'predicted_class': 'unknown',
                'confidence': 0.0,
                'error': str(e)
            }

    def save_model(self, path=None):
        """Save the trained model"""
        if path is None:
            path = self.model_path
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'classes': self.classes,
            'image_size': self.image_size,
            'training_accuracy': self.training_accuracy,
            'test_accuracy': self.test_accuracy
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Enhanced model saved to {path}")

    def load_model(self, path=None):
        """Load a trained model"""
        if path is None:
            path = self.model_path
        
        if os.path.exists(path):
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.pca = model_data['pca']
            self.classes = model_data['classes']
            self.image_size = model_data['image_size']
            self.training_accuracy = model_data.get('training_accuracy', 0.0)
            self.test_accuracy = model_data.get('test_accuracy', 0.0)
            logger.info(f"Enhanced model loaded from {path}")
        else:
            logger.error(f"Model file not found: {path}")

def main():
    """Main function to train and evaluate the enhanced X-ray model"""
    classifier = EnhancedXRayClassifier()
    
    # Train the model
    logger.info("Training enhanced X-ray classification model...")
    training_accuracy = classifier.train_enhanced_model(max_images_per_class=1500)
    logger.info(f"Training Accuracy: {training_accuracy:.4f} ({training_accuracy*100:.2f}%)")
    
    # Evaluate on the test set
    logger.info("Evaluating model on test set...")
    test_accuracy, class_wise_accuracy = classifier.evaluate('datasets/chest_xray/test')
    logger.info(f"Overall Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Save the model
    classifier.save_model()
    logger.info(f"Enhanced model saved as: {classifier.model_path}")

if __name__ == '__main__':
    main()
