#!/usr/bin/env python3
"""
Chest X-Ray Pneumonia Classification Model
Classifies chest X-ray images as Normal or Pneumonia
Uses multiple algorithms to find the best performing one
"""

import os
import cv2
import numpy as np
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class XRayPneumoniaClassifier:
    def __init__(self, model_path='xray_pneumonia_model.pkl', dataset_path='datasets/chest_xray'):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.image_size = (64, 64)  # Resize images to this size
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
                image_path = os.path.join(class_dir, filename)
                try:
                    # Load image
                    img = cv2.imread(image_path)
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

    def train_multiple_models(self, max_images_per_class=500):
        """Train multiple models and select the best one"""
        logger.info("Training multiple models to find the best performer...")
        
        # Load training data
        train_images, train_labels = self._load_and_preprocess_images(
            os.path.join(self.dataset_path, 'train'), 
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
        
        # Define models to test
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200, 
                random_state=42, 
                max_depth=15, 
                min_samples_split=5, 
                min_samples_leaf=2,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'ExtraTrees': ExtraTreesClassifier(
                n_estimators=200,
                random_state=42,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=-1
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42,
                probability=True
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        }
        
        best_model = None
        best_score = 0
        best_name = ""
        
        # Train and evaluate each model
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                
                train_acc = accuracy_score(y_train, train_pred)
                val_acc = accuracy_score(y_val, val_pred)
                
                logger.info(f"{name} - Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")
                
                # Select best model based on validation accuracy
                if val_acc > best_score:
                    best_score = val_acc
                    best_model = model
                    best_name = name
                    
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
        
        if best_model is None:
            raise ValueError("No model could be trained successfully")
        
        # Train final model on full training set
        logger.info(f"Training final {best_name} model on full training set...")
        best_model.fit(train_features_scaled, train_labels)
        
        # Calculate final training accuracy
        train_pred_final = best_model.predict(train_features_scaled)
        self.training_accuracy = accuracy_score(train_labels, train_pred_final)
        
        self.model = best_model
        
        logger.info(f"Best model: {best_name} with validation accuracy: {best_score:.3f}")
        logger.info(f"Final training accuracy: {self.training_accuracy:.3f}")
        
        return self.training_accuracy

    def evaluate(self, test_data_dir):
        """Evaluate the model on test data"""
        logger.info("Evaluating model on test set...")
        
        # Load test data
        test_images, test_labels = self._load_and_preprocess_images(test_data_dir)
        
        # Extract features
        test_features = self.extract_enhanced_features(test_images)
        
        # Scale features
        test_features_scaled = self.scaler.transform(test_features)
        
        # Make predictions
        predictions = self.model.predict(test_features_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(test_labels, predictions)
        
        # Calculate class-wise accuracy
        class_accuracy = {}
        for class_name in self.classes:
            class_mask = test_labels == class_name
            if np.sum(class_mask) > 0:
                class_pred = predictions[class_mask]
                class_true = test_labels[class_mask]
                class_acc = accuracy_score(class_true, class_pred)
                class_accuracy[class_name] = class_acc
        
        # Print detailed results
        logger.info(f"Test Accuracy: {accuracy:.3f}")
        logger.info("Class-wise Test Results:")
        for cls, acc in class_accuracy.items():
            logger.info(f"  {cls}: {acc:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, predictions, labels=self.classes)
        logger.info("Confusion Matrix:")
        logger.info(f"  {self.classes[0]}  {self.classes[1]}")
        for i, class_name in enumerate(self.classes):
            logger.info(f"{class_name} {cm[i]}")
        
        return accuracy, class_accuracy

    def predict(self, image_path):
        """Predict pneumonia from image path"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize
        img = cv2.resize(img, self.image_size)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        # Extract features
        img_flat = img.flatten()
        features = self.extract_enhanced_features([img_flat])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction with probability
        prediction_proba = self.model.predict_proba(features_scaled)[0]
        predicted_class_idx = np.argmax(prediction_proba)
        confidence = prediction_proba[predicted_class_idx]
        
        predicted_class = self.classes[predicted_class_idx]
        
        # Format the class name properly
        if predicted_class == 'NORMAL':
            formatted_class = 'Normal'
        else:
            formatted_class = 'Pneumonia'
        
        return {
            'predicted_class': formatted_class,
            'confidence': float(confidence)
        }

    def save_model(self):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'classes': self.classes,
            'image_size': self.image_size,
            'training_accuracy': self.training_accuracy
        }
        
        joblib.dump(model_data, self.model_path)
        logger.info(f"Model saved to {self.model_path}")

    def load_model(self, model_path=None):
        """Load a trained model"""
        path = model_path or self.model_path
        if os.path.exists(path):
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.classes = model_data['classes']
            self.image_size = model_data['image_size']
            self.training_accuracy = model_data['training_accuracy']
            logger.info(f"Model loaded from {path}")
        else:
            logger.error(f"Model file not found: {path}")

def main():
    """Main function to train and evaluate the X-ray model"""
    classifier = XRayPneumoniaClassifier()
    
    # Train the model
    logger.info("Training X-ray pneumonia classification model...")
    training_accuracy = classifier.train_multiple_models(max_images_per_class=800)  # Use more data for better accuracy
    logger.info(f"Training Accuracy: {training_accuracy:.2%}")

    # Evaluate on the test set
    logger.info("Evaluating model on test set...")
    test_accuracy, class_wise_accuracy = classifier.evaluate(os.path.join(classifier.dataset_path, 'test'))
    logger.info(f"Overall Test Accuracy: {test_accuracy:.2%}")
    logger.info("Class-wise Test Results:")
    for cls, acc in class_wise_accuracy.items():
        logger.info(f"  {cls:<10}: {acc:.2%}")

    # Save the model
    classifier.save_model()
    logger.info(f"Model saved as: {classifier.model_path}")

if __name__ == '__main__':
    main()
