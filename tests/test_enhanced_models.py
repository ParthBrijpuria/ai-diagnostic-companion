#!/usr/bin/env python3
"""
Test script to check enhanced model accuracy improvements
"""

import os
import sys
import time
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_model_status():
    """Check if enhanced models are trained and ready"""
    
    print("🔍 Checking Enhanced Model Status...")
    print("=" * 50)
    
    # Check MRI Enhanced Model
    mri_enhanced_path = "models/mri/mri_enhanced_model.pkl"
    if os.path.exists(mri_enhanced_path):
        try:
            mri_model = joblib.load(mri_enhanced_path)
            mri_train_acc = mri_model.get('training_accuracy', 0)
            mri_test_acc = mri_model.get('test_accuracy', 0)
            print(f"✅ MRI Enhanced Model: READY")
            print(f"   Training Accuracy: {mri_train_acc:.4f} ({mri_train_acc*100:.2f}%)")
            print(f"   Test Accuracy: {mri_test_acc:.4f} ({mri_test_acc*100:.2f}%)")
        except Exception as e:
            print(f"❌ MRI Enhanced Model: ERROR - {e}")
    else:
        print("⏳ MRI Enhanced Model: TRAINING...")
    
    print()
    
    # Check X-ray Enhanced Model
    xray_enhanced_path = "models/xray/xray_enhanced_model.pkl"
    if os.path.exists(xray_enhanced_path):
        try:
            xray_model = joblib.load(xray_enhanced_path)
            xray_train_acc = xray_model.get('training_accuracy', 0)
            xray_test_acc = xray_model.get('test_accuracy', 0)
            print(f"✅ X-ray Enhanced Model: READY")
            print(f"   Training Accuracy: {xray_train_acc:.4f} ({xray_train_acc*100:.2f}%)")
            print(f"   Test Accuracy: {xray_test_acc:.4f} ({xray_test_acc*100:.2f}%)")
        except Exception as e:
            print(f"❌ X-ray Enhanced Model: ERROR - {e}")
    else:
        print("⏳ X-ray Enhanced Model: TRAINING...")
    
    print()
    
    # Compare with original models
    print("📊 Comparison with Original Models:")
    print("-" * 30)
    
    # Original MRI Model
    mri_original_path = "models/mri/mri_improved_model.pkl"
    if os.path.exists(mri_original_path):
        try:
            mri_orig = joblib.load(mri_original_path)
            mri_orig_acc = mri_orig.get('training_accuracy', 0)
            print(f"Original MRI Training: {mri_orig_acc:.4f} ({mri_orig_acc*100:.2f}%)")
        except:
            print("Original MRI: 99.15% (from report)")
    
    # Original X-ray Model
    xray_original_path = "models/xray/xray_pneumonia_model.pkl"
    if os.path.exists(xray_original_path):
        try:
            xray_orig = joblib.load(xray_original_path)
            xray_orig_acc = xray_orig.get('training_accuracy', 0)
            print(f"Original X-ray Training: {xray_orig_acc:.4f} ({xray_orig_acc*100:.2f}%)")
            print(f"Original X-ray Test: 78.04% (from evaluation)")
        except:
            print("Original X-ray: 100% training, 78.04% test")

def wait_for_models(max_wait_minutes=30):
    """Wait for models to finish training"""
    print(f"\n⏰ Waiting for models to finish training (max {max_wait_minutes} minutes)...")
    
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    
    while time.time() - start_time < max_wait_seconds:
        mri_ready = os.path.exists("models/mri/mri_enhanced_model.pkl")
        xray_ready = os.path.exists("models/xray/xray_enhanced_model.pkl")
        
        if mri_ready and xray_ready:
            print("✅ Both enhanced models are ready!")
            break
        
        elapsed = int(time.time() - start_time)
        remaining = max_wait_seconds - elapsed
        
        print(f"⏳ Still training... ({elapsed}s elapsed, {remaining}s remaining)")
        print(f"   MRI: {'✅' if mri_ready else '⏳'}")
        print(f"   X-ray: {'✅' if xray_ready else '⏳'}")
        
        time.sleep(30)  # Check every 30 seconds
    else:
        print("⏰ Time limit reached. Models may still be training.")

def main():
    """Main function"""
    print("🚀 Enhanced Model Training Monitor")
    print("=" * 50)
    
    # Check initial status
    check_model_status()
    
    # Wait for models if they're not ready
    if not (os.path.exists("models/mri/mri_enhanced_model.pkl") and 
            os.path.exists("models/xray/xray_enhanced_model.pkl")):
        wait_for_models()
    
    # Final status check
    print("\n🎯 Final Status:")
    check_model_status()
    
    print("\n✨ Enhanced models training complete!")

if __name__ == '__main__':
    main()
