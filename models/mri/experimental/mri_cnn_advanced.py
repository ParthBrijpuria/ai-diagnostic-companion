#!/usr/bin/env python3
"""
Advanced CNN-based MRI Brain Tumor Classification Model
Uses deep learning for maximum accuracy
"""

import os
import cv2
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MRIAdvancedCNNClassifier:
    def __init__(self, model_path='mri_cnn_advanced_model.pkl', dataset_path='../../datasets/mri scan'):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.image_size = (224, 224)  # Larger size for better CNN performance
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
                    
                    # Convert to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Resize
                    img = cv2.resize(img, self.image_size)
                    
                    # Normalize
                    img = img.astype(np.float32) / 255.0
                    
                    images.append(img)
                    labels.append(class_name)
                    
                except Exception as e:
                    logger.warning(f"Error processing {filename}: {e}")
                    continue
        
        logger.info(f"Loaded {len(images)} images total")
        return np.array(images), np.array(labels)

    def create_advanced_cnn_model(self):
        """Create advanced CNN model with transfer learning"""
        logger.info("Creating advanced CNN model with transfer learning...")
        
        # Use ResNet50 as base model
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Create model
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            BatchNormalization(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            BatchNormalization(),
            Dense(256, activation='relu'),
            Dropout(0.3),
            BatchNormalization(),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(4, activation='softmax')  # 4 classes
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def create_custom_cnn_model(self):
        """Create custom CNN model"""
        logger.info("Creating custom CNN model...")
        
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Fourth Convolutional Block
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Fifth Convolutional Block
            Conv2D(512, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(512, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Flatten and Dense layers
            Flatten(),
            Dense(1024, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(4, activation='softmax')  # 4 classes
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train_advanced_cnn_model(self, max_images_per_class=800):
        """Train advanced CNN model"""
        logger.info("Training advanced CNN MRI classification model...")
        
        # Load training data
        train_images, train_labels = self._load_and_preprocess_images(
            os.path.join(self.dataset_path, 'Training'), 
            max_images_per_class
        )
        
        # Load test data
        test_images, test_labels = self._load_and_preprocess_images(
            os.path.join(self.dataset_path, 'Testing'), 
            max_images_per_class // 2
        )
        
        # Encode labels
        train_labels_encoded = self.label_encoder.fit_transform(train_labels)
        test_labels_encoded = self.label_encoder.transform(test_labels)
        
        train_labels_categorical = to_categorical(train_labels_encoded, num_classes=4)
        test_labels_categorical = to_categorical(test_labels_encoded, num_classes=4)
        
        # Split training data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            train_images, train_labels_categorical, test_size=0.2, random_state=42
        )
        
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # Try transfer learning first
        try:
            logger.info("Attempting transfer learning with ResNet50...")
            self.model = self.create_advanced_cnn_model()
            
            # Callbacks
            early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=0.0001)
            model_checkpoint = ModelCheckpoint('best_mri_model.h5', monitor='val_accuracy', save_best_only=True)
            
            # Train model
            history = self.model.fit(
                datagen.flow(X_train, y_train, batch_size=32),
                validation_data=(X_val, y_val),
                epochs=50,
                callbacks=[early_stopping, reduce_lr, model_checkpoint],
                verbose=1
            )
            
        except Exception as e:
            logger.warning(f"Transfer learning failed: {e}")
            logger.info("Falling back to custom CNN...")
            
            # Fallback to custom CNN
            self.model = self.create_custom_cnn_model()
            
            # Callbacks
            early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=0.0001)
            model_checkpoint = ModelCheckpoint('best_mri_model.h5', monitor='val_accuracy', save_best_only=True)
            
            # Train model
            history = self.model.fit(
                datagen.flow(X_train, y_train, batch_size=32),
                validation_data=(X_val, y_val),
                epochs=50,
                callbacks=[early_stopping, reduce_lr, model_checkpoint],
                verbose=1
            )
        
        # Calculate training accuracy
        train_loss, train_acc = self.model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = self.model.evaluate(X_val, y_val, verbose=0)
        
        self.training_accuracy = val_acc
        
        logger.info(f"Training accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        logger.info(f"Validation accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        
        # Test on test set
        test_loss, test_acc = self.model.evaluate(test_images, test_labels_categorical, verbose=0)
        logger.info(f"Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        # Save model
        self.model.save(f"{self.model_path.replace('.pkl', '.h5')}")
        
        # Save metadata
        model_data = {
            'label_encoder': self.label_encoder,
            'classes': self.classes,
            'image_size': self.image_size,
            'training_accuracy': self.training_accuracy,
            'test_accuracy': test_acc
        }
        
        joblib.dump(model_data, self.model_path)
        logger.info(f"Model saved to {self.model_path}")
        
        return self.training_accuracy, test_acc

    def test_cnn_model(self, max_images_per_class=500):
        """Test CNN model on test dataset"""
        logger.info("Testing CNN model...")
        
        # Load test data
        test_images, test_labels = self._load_and_preprocess_images(
            os.path.join(self.dataset_path, 'Testing'), 
            max_images_per_class
        )
        
        # Encode labels
        test_labels_encoded = self.label_encoder.transform(test_labels)
        test_labels_categorical = to_categorical(test_labels_encoded, num_classes=4)
        
        # Make predictions
        predictions = self.model.predict(test_images)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Calculate accuracy
        accuracy = accuracy_score(test_labels_encoded, predicted_classes)
        
        logger.info(f"Test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(test_labels_encoded, predicted_classes, target_names=self.classes))
        
        return accuracy

def main():
    """Main function"""
    classifier = MRIAdvancedCNNClassifier()
    
    # Train model
    training_accuracy, test_accuracy = classifier.train_advanced_cnn_model(max_images_per_class=800)
    
    # Test model
    final_test_accuracy = classifier.test_cnn_model(max_images_per_class=500)
    
    print(f"\n🎯 Final Results:")
    print(f"Training Accuracy: {training_accuracy:.4f} ({training_accuracy*100:.2f}%)")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Final Test Accuracy: {final_test_accuracy:.4f} ({final_test_accuracy*100:.2f}%)")

if __name__ == "__main__":
    main()
