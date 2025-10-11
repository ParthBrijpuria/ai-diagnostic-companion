#!/usr/bin/env python3
"""
MRI Brain Tumor Classification using RandomForest
Optimized RandomForest model for MRI tumor classification
"""

import os
import cv2
import numpy as np
import joblib
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MRIRandomForestClassifier:
    def __init__(self, dataset_path='datasets/mri scan'):
        self.dataset_path = dataset_path
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.image_size = (64, 64)
        self.scaler = StandardScaler()
        self.model = None
        self.training_accuracy = 0
        
    def load_and_preprocess_images(self, data_dir, max_images_per_class=200):
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
            image_files = image_files[:max_images_per_class]
            
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
                    
                    # Flatten
                    img_flat = img.flatten()
                    
                    images.append(img_flat)
                    labels.append(class_name)
                    
                except Exception as e:
                    logger.warning(f"Error loading image {img_path}: {e}")
                    continue
        
        logger.info(f"Loaded {len(images)} images total")
        return np.array(images), np.array(labels)
    
    def extract_enhanced_features(self, images):
        """Extract enhanced features from images"""
        logger.info("Extracting enhanced features...")
        
        features = []
        for img_flat in images:
            img = img_flat.reshape(self.image_size)
            
            # Basic statistical features
            mean_val = np.mean(img)
            std_val = np.std(img)
            min_val = np.min(img)
            max_val = np.max(img)
            median_val = np.median(img)
            
            # Texture features
            img_uint8 = (img * 255).astype(np.uint8)  # Convert to uint8 for OpenCV
            sobel_x = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Laplacian for edge detection
            laplacian = cv2.Laplacian(img_uint8, cv2.CV_64F)
            
            # Combine all features
            feature_vector = np.concatenate([
                img_flat,  # Original pixel values (4096)
                [mean_val, std_val, min_val, max_val, median_val],  # Statistical features (5)
                [np.mean(gradient_magnitude), np.std(gradient_magnitude), np.mean(laplacian), np.std(laplacian)]  # Texture features (4)
            ])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def optimize_random_forest(self, X_train, y_train):
        """Optimize RandomForest parameters using GridSearch"""
        logger.info("Optimizing RandomForest parameters...")
        
        # Use subset for faster grid search
        X_subset = X_train[:1000] if len(X_train) > 1000 else X_train
        y_subset = y_train[:1000] if len(y_train) > 1000 else y_train
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(
            rf,
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_subset, y_subset)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train(self, max_images_per_class=200):
        """Train the RandomForest model"""
        logger.info("Starting RandomForest training...")
        
        # Load and preprocess images
        X, y = self.load_and_preprocess_images(self.dataset_path, max_images_per_class)
        
        if len(X) == 0:
            raise ValueError("No images found in the dataset")
        
        # Extract enhanced features
        X_features = self.extract_enhanced_features(X)
        
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
        
        # Optimize RandomForest
        self.model = self.optimize_random_forest(X_train_scaled, y_train)
        
        # Train on full dataset
        logger.info("Training optimized model on full dataset...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        self.training_accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Training accuracy: {self.training_accuracy:.4f}")
        
        # Print results
        print("\n" + "="*60)
        print("RANDOM FOREST TRAINING RESULTS")
        print("="*60)
        print(f"Training Accuracy: {self.training_accuracy:.4f} ({self.training_accuracy*100:.2f}%)")
        
        # Classification report
        report = classification_report(y_test, y_pred, target_names=self.classes)
        print(f"\nClassification Report:\n{report}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:\n{cm}")
        
        return self.training_accuracy
    
    def save_model(self, model_path='mri_random_forest.pkl'):
        """Save the trained model"""
        if self.model is not None:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'classes': self.classes,
                'image_size': self.image_size,
                'training_accuracy': self.training_accuracy
            }
            joblib.dump(model_data, model_path)
            logger.info(f"Model saved to {model_path}")
        else:
            logger.error("No model to save")
    
    def load_model(self, model_path='mri_random_forest.pkl'):
        """Load a saved model"""
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.classes = model_data['classes']
            self.image_size = model_data['image_size']
            self.training_accuracy = model_data['training_accuracy']
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.error(f"Model file not found: {model_path}")
    
    def predict(self, image_path):
        """Predict tumor type from image path - returns category name and probability"""
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
        if predicted_class == 'notumor':
            formatted_class = 'No Tumor'
        else:
            formatted_class = predicted_class.capitalize()
        
        return {
            'predicted_class': formatted_class,
            'confidence': float(confidence)
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
        
        # Test all images for comprehensive evaluation
        for filename in image_files:
            image_path = os.path.join(class_dir, filename)
            try:
                prediction = classifier.predict(image_path)
                if prediction == class_name:
                    class_correct += 1
                    total_correct += 1
                total_images += 1
            except Exception as e:
                logger.warning(f"Error processing {filename}: {e}")
        
        class_accuracy = class_correct / class_total if class_total > 0 else 0
        class_results[class_name] = {
            'correct': class_correct,
            'total': class_total,
            'accuracy': class_accuracy
        }
        
        logger.info(f"{class_name}: {class_correct}/{class_total} correct ({class_accuracy:.2%})")
    
    overall_accuracy = total_correct / total_images if total_images > 0 else 0
    logger.info(f"Overall test accuracy: {overall_accuracy:.2%} ({total_correct}/{total_images})")
    
    return class_results, overall_accuracy

def main():
    """Main function to train and test the RandomForest model"""
    print("MRI Brain Tumor Classification - RandomForest Model")
    print("=" * 60)
    
    # Initialize classifier
    classifier = MRIRandomForestClassifier()
    
    # Train the model
    print("Training RandomForest model...")
    training_accuracy = classifier.train(max_images_per_class=200)
    
    # Save the model
    classifier.save_model()
    
    # Test on test dataset
    print("\n" + "="*60)
    print("TESTING ON TEST DATASET")
    print("="*60)
    
    class_results, test_accuracy = test_on_test_dataset(classifier)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"Model: RandomForest")
    print(f"Training Accuracy: {training_accuracy:.2%}")
    print(f"Test Accuracy: {test_accuracy:.2%}")
    print(f"Total Test Images: {sum(r['total'] for r in class_results.values())}")
    print(f"Total Correct Predictions: {sum(r['correct'] for r in class_results.values())}")
    
    print("\nClass-wise Test Results:")
    for class_name, result in class_results.items():
        print(f"  {class_name:12}: {result['accuracy']:.2%} ({result['correct']:3d}/{result['total']:3d})")
    
    # Performance assessment
    if test_accuracy >= 0.80:
        print(f"\n🎉 EXCELLENT: Model achieved {test_accuracy:.2%} accuracy!")
    elif test_accuracy >= 0.70:
        print(f"\n✅ GOOD: Model achieved {test_accuracy:.2%} accuracy!")
    elif test_accuracy >= 0.60:
        print(f"\n⚠️  FAIR: Model achieved {test_accuracy:.2%} accuracy - could be improved")
    else:
        print(f"\n❌ POOR: Model achieved {test_accuracy:.2%} accuracy - needs improvement")
    
    print(f"\nModel saved as: mri_random_forest.pkl")

if __name__ == '__main__':
    main()
