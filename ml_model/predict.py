import joblib
import pandas as pd
import numpy as np
import os

def load_heart_model(model_path='heart_model.joblib'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")
    return joblib.load(model_path)

def predict_risk(model, features_df):
    """
    Returns (prediction, probability)
    """
    pred = model.predict(features_df)[0]
    prob = model.predict_proba(features_df)[0][1]
    return pred, prob

if __name__ == "__main__":
    # Example logic
    try:
        model = load_heart_model()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error: {e}")
