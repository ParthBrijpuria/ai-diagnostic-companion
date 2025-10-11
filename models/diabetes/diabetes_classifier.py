#!/usr/bin/env python3
"""
Diabetes Classification Model
Classifies diabetes based on patient parameters
Uses multiple algorithms to find the best performing one
"""

import os
import pandas as pd
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

class DiabetesClassifier:
    def __init__(self, model_path='diabetes_model.pkl', dataset_path='datasets/diabetes.csv'):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.feature_columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        self.training_accuracy = 0.0

    def load_and_preprocess_data(self):
        """Load and preprocess the diabetes dataset"""
        logger.info(f"Loading diabetes dataset from {self.dataset_path}")
        
        try:
            # Load the dataset
            df = pd.read_csv(self.dataset_path)
            logger.info(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
            
            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.any():
                logger.warning(f"Missing values found: {missing_values[missing_values > 0]}")
                # Fill missing values with median
                df = df.fillna(df.median())
            
            # Separate features and target
            X = df[self.feature_columns]
            y = df['Outcome']  # Assuming 'Outcome' is the target column
            
            logger.info(f"Features: {X.shape}, Target: {y.shape}")
            logger.info(f"Class distribution: {y.value_counts().to_dict()}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def train_multiple_models(self):
        """Train multiple models and select the best one"""
        logger.info("Training multiple models to find the best performer...")
        
        # Load and preprocess data
        X, y = self.load_and_preprocess_data()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
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
        best_model.fit(X_scaled, y)
        
        # Calculate final training accuracy
        train_pred_final = best_model.predict(X_scaled)
        self.training_accuracy = accuracy_score(y, train_pred_final)
        
        self.model = best_model
        
        logger.info(f"Best model: {best_name} with validation accuracy: {best_score:.3f}")
        logger.info(f"Final training accuracy: {self.training_accuracy:.3f}")
        
        return self.training_accuracy

    def evaluate(self, test_size=0.2):
        """Evaluate the model on test data"""
        logger.info("Evaluating model on test set...")
        
        # Load and preprocess data
        X, y = self.load_and_preprocess_data()
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Make predictions
        predictions = self.model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)
        
        # Print detailed results
        logger.info(f"Test Accuracy: {accuracy:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        logger.info("Confusion Matrix:")
        logger.info(f"  Non-Diabetic  Diabetic")
        logger.info(f"Non-Diabetic {cm[0]}")
        logger.info(f"Diabetic     {cm[1]}")
        
        # Classification report
        report = classification_report(y_test, predictions, target_names=['Non-Diabetic', 'Diabetic'])
        logger.info("Classification Report:")
        logger.info(report)
        
        return accuracy

    def predict(self, input_data):
        """Predict diabetes from input parameters"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Convert input data to numpy array
        if isinstance(input_data, dict):
            # Convert dictionary to array in the correct order
            input_array = np.array([
                input_data.get('Pregnancies', 0),
                input_data.get('Glucose', 0),
                input_data.get('BloodPressure', 0),
                input_data.get('SkinThickness', 0),
                input_data.get('Insulin', 0),
                input_data.get('BMI', 0),
                input_data.get('DiabetesPedigreeFunction', 0),
                input_data.get('Age', 0)
            ]).reshape(1, -1)
        else:
            input_array = np.array(input_data).reshape(1, -1)
        
        # Scale features
        input_scaled = self.scaler.transform(input_array)
        
        # Make prediction with probability
        prediction_proba = self.model.predict_proba(input_scaled)[0]
        predicted_class_idx = np.argmax(prediction_proba)
        confidence = prediction_proba[predicted_class_idx]
        
        # Convert to human-readable format
        if predicted_class_idx == 0:
            predicted_class = 'Non-Diabetic'
        else:
            predicted_class = 'Diabetic'
        
        return {
            'predicted_class': predicted_class,
            'confidence': float(confidence)
        }

    def save_model(self):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
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
            self.feature_columns = model_data['feature_columns']
            self.training_accuracy = model_data['training_accuracy']
            logger.info(f"Model loaded from {path}")
        else:
            logger.error(f"Model file not found: {path}")

def main():
    """Main function to train and evaluate the diabetes model"""
    classifier = DiabetesClassifier()
    
    # Train the model
    logger.info("Training diabetes classification model...")
    training_accuracy = classifier.train_multiple_models()
    logger.info(f"Training Accuracy: {training_accuracy:.2%}")

    # Evaluate on the test set
    logger.info("Evaluating model on test set...")
    test_accuracy = classifier.evaluate()
    logger.info(f"Overall Test Accuracy: {test_accuracy:.2%}")

    # Save the model
    classifier.save_model()
    logger.info(f"Model saved as: {classifier.model_path}")

if __name__ == '__main__':
    main()
