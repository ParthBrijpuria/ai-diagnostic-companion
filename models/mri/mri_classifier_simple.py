#!/usr/bin/env python3
"""
Advanced MRI Brain Tumor Classification Model
Tests multiple ML algorithms and selects the best performing one
"""

import os
import cv2
import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from collections import Counter
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MRITumorClassifier:
    def __init__(self, dataset_path='datasets/mri scan'):
        self.dataset_path = dataset_path
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.image_size = (64, 64)  # Resize images to this size
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.best_accuracy = 0
        
    def load_and_preprocess_images(self, data_dir, max_images_per_class=300):
        """Load and preprocess MRI images from the dataset"""
        logger.info(f"Loading images from {data_dir}")
        
        images = []
        labels = []
        
        for class_name in self.classes:
            class_path = os.path.join(data_dir, 'Training', class_name)
            if not os.path.exists(class_path):
                logger.warning(f"Class directory not found: {class_path}")
                continue
                
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            image_files = image_files[:max_images_per_class]  # Limit for faster training
            
            logger.info(f"Loading {len(image_files)} images from {class_name}")
            
            for filename in image_files:
                try:
                    img_path = os.path.join(class_path, filename)
                    img = cv2.imread(img_path)
                    
                    if img is None:
                        continue
                        
                    # Convert to grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Resize
                    img = cv2.resize(img, self.image_size)
                    
                    # Normalize
                    img = img.astype(np.float32) / 255.0
                    
                    # Flatten for traditional ML algorithms
                    img_flat = img.flatten()
                    
                    images.append(img_flat)
                    labels.append(class_name)
                    
                except Exception as e:
                    logger.warning(f"Error loading image {img_path}: {e}")
                    continue
        
        logger.info(f"Loaded {len(images)} images total")
        return np.array(images), np.array(labels)
    
    def extract_features(self, images):
        """Extract additional features from images"""
        logger.info("Extracting additional features...")
        
        features = []
        for img_flat in images:
            img = img_flat.reshape(self.image_size)
            
            # Basic statistical features
            mean_val = np.mean(img)
            std_val = np.std(img)
            min_val = np.min(img)
            max_val = np.max(img)
            
            # Texture features using simple filters
            sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Combine pixel features with statistical features
            feature_vector = np.concatenate([
                img_flat,  # Original pixel values
                [mean_val, std_val, min_val, max_val],  # Statistical features
                [np.mean(gradient_magnitude), np.std(gradient_magnitude)]  # Texture features
            ])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def test_algorithms(self, X_train, X_test, y_train, y_test):
        """Test multiple ML algorithms and return results"""
        logger.info("Testing multiple ML algorithms...")
        
        algorithms = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            ),
            'ExtraTrees': ExtraTreesClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'SVM_RBF': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42,
                probability=True
            ),
            'SVM_Poly': SVC(
                kernel='poly',
                C=1.0,
                degree=3,
                random_state=42,
                probability=True
            ),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(512, 256, 128),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        }
        
        results = {}
        
        for name, model in algorithms.items():
            logger.info(f"Training {name}...")
            start_time = time.time()
            
            try:
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                training_time = time.time() - start_time
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'training_time': training_time,
                    'predictions': y_pred
                }
                
                logger.info(f"{name}: Accuracy={accuracy:.4f}, CV={cv_mean:.4f}±{cv_std:.4f}, Time={training_time:.2f}s")
                
                # Update best model
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_model = model
                    self.best_model_name = name
                    
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                results[name] = {
                    'model': None,
                    'accuracy': 0,
                    'cv_mean': 0,
                    'cv_std': 0,
                    'training_time': 0,
                    'error': str(e)
                }
        
        return results
    
    def optimize_best_model(self, X_train, y_train):
        """Optimize the best performing model using grid search"""
        logger.info(f"Optimizing {self.best_model_name}...")
        
        if self.best_model_name == 'RandomForest':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        elif self.best_model_name == 'GradientBoosting':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1]
            }
        elif self.best_model_name == 'SVM_RBF':
            param_grid = {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto']
            }
        else:
            logger.info("No optimization defined for this model type")
            return self.best_model
        
        # Use a subset for faster grid search
        X_subset = X_train[:500] if len(X_train) > 500 else X_train
        y_subset = y_train[:500] if len(y_train) > 500 else y_train
        
        grid_search = GridSearchCV(
            self.best_model,
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_subset, y_subset)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Train on full dataset with best parameters
        optimized_model = grid_search.best_estimator_
        optimized_model.fit(X_train, y_train)
        
        return optimized_model
    
    def train(self, max_images_per_class=200):
        """Train the MRI classification model"""
        logger.info("Starting MRI model training...")
        
        # Load and preprocess images
        X, y = self.load_and_preprocess_images(self.dataset_path, max_images_per_class)
        
        if len(X) == 0:
            raise ValueError("No images found in the dataset")
        
        # Extract features
        X_features = self.extract_features(X)
        
        # Convert labels to numeric
        label_mapping = {label: idx for idx, label in enumerate(self.classes)}
        y_numeric = np.array([label_mapping[label] for label in y])
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y_numeric, test_size=0.2, random_state=42, stratify=y_numeric
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Training set: {X_train_scaled.shape}")
        logger.info(f"Test set: {X_test_scaled.shape}")
        
        # Test different algorithms
        results = self.test_algorithms(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Optimize the best model
        self.best_model = self.optimize_best_model(X_train_scaled, y_train)
        
        # Final evaluation
        y_pred_final = self.best_model.predict(X_test_scaled)
        final_accuracy = accuracy_score(y_test, y_pred_final)
        
        logger.info(f"Final optimized {self.best_model_name} accuracy: {final_accuracy:.4f}")
        
        # Print detailed results
        print("\n" + "="*60)
        print("ALGORITHM COMPARISON RESULTS")
        print("="*60)
        for name, result in results.items():
            if 'error' not in result:
                print(f"{name:15} | Accuracy: {result['accuracy']:.4f} | CV: {result['cv_mean']:.4f}±{result['cv_std']:.4f} | Time: {result['training_time']:.2f}s")
            else:
                print(f"{name:15} | Error: {result['error']}")
        
        print(f"\n🏆 BEST MODEL: {self.best_model_name} with accuracy: {final_accuracy:.4f}")
        
        # Classification report
        report = classification_report(y_test, y_pred_final, target_names=self.classes)
        print(f"\nClassification Report:\n{report}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_final)
        print(f"\nConfusion Matrix:\n{cm}")
        
        return results, final_accuracy
    
    def save_model(self, model_path='mri_tumor_classifier.pkl'):
        """Save the best model"""
        if self.best_model is not None:
            model_data = {
                'model': self.best_model,
                'scaler': self.scaler,
                'classes': self.classes,
                'image_size': self.image_size,
                'model_name': self.best_model_name,
                'accuracy': self.best_accuracy
            }
            joblib.dump(model_data, model_path)
            logger.info(f"Model saved to {model_path}")
        else:
            logger.error("No model to save")
    
    def load_model(self, model_path='mri_tumor_classifier.pkl'):
        """Load a saved model"""
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            self.best_model = model_data['model']
            self.scaler = model_data['scaler']
            self.classes = model_data['classes']
            self.image_size = model_data['image_size']
            self.best_model_name = model_data['model_name']
            self.best_accuracy = model_data['accuracy']
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.error(f"Model file not found: {model_path}")
    
    def predict(self, image_path):
        """Predict tumor type from image path"""
        if self.best_model is None:
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
        features = self.extract_features([img_flat])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction_proba = self.best_model.predict_proba(features_scaled)[0]
        predicted_class_idx = np.argmax(prediction_proba)
        confidence = prediction_proba[predicted_class_idx]
        
        predicted_class = self.classes[predicted_class_idx]
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': {
                self.classes[i]: prob for i, prob in enumerate(prediction_proba)
            },
            'model_used': self.best_model_name
        }

def test_on_test_dataset(classifier):
    """Test the trained model on the test dataset"""
    logger.info("Testing on test dataset...")
    
    test_path = 'datasets/mri scan/Testing'
    total_correct = 0
    total_images = 0
    class_results = {}
    
    for class_name in classifier.classes:
        class_dir = os.path.join(test_path, class_name)
        if not os.path.exists(class_dir):
            continue
        
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_correct = 0
        class_total = len(image_files)
        
        logger.info(f"Testing {class_total} images from {class_name}")
        
        for filename in image_files[:50]:  # Test first 50 images per class
            image_path = os.path.join(class_dir, filename)
            try:
                prediction = classifier.predict(image_path)
                if prediction['predicted_class'] == class_name:
                    class_correct += 1
                    total_correct += 1
                total_images += 1
            except Exception as e:
                logger.warning(f"Error processing {filename}: {e}")
        
        class_accuracy = class_correct / min(class_total, 50) if min(class_total, 50) > 0 else 0
        class_results[class_name] = {
            'correct': class_correct,
            'total': min(class_total, 50),
            'accuracy': class_accuracy
        }
        
        logger.info(f"{class_name}: {class_correct}/{min(class_total, 50)} correct ({class_accuracy:.2%})")
    
    overall_accuracy = total_correct / total_images if total_images > 0 else 0
    logger.info(f"Overall test accuracy: {overall_accuracy:.2%} ({total_correct}/{total_images})")
    
    return class_results, overall_accuracy

def main():
    """Main function to train and test the model"""
    print("MRI Brain Tumor Classification - Advanced Model")
    print("=" * 50)
    
    # Initialize classifier
    classifier = MRITumorClassifier()
    
    # Train the model (using subset for faster training)
    print("Training model with multiple algorithms...")
    results, final_accuracy = classifier.train(max_images_per_class=150)
    
    # Save the best model
    classifier.save_model()
    
    # Test on test dataset
    print("\n" + "="*50)
    print("TESTING ON TEST DATASET")
    print("="*50)
    
    class_results, test_accuracy = test_on_test_dataset(classifier)
    
    # Final summary
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    print(f"Best Model: {classifier.best_model_name}")
    print(f"Training Accuracy: {final_accuracy:.2%}")
    print(f"Test Accuracy: {test_accuracy:.2%}")
    print("\nClass-wise Test Results:")
    for class_name, result in class_results.items():
        print(f"  {class_name}: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")

if __name__ == '__main__':
    main()
