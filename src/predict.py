import joblib
import numpy as np

def load_model_and_scaler():
    model = joblib.load('model/random_forest_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    return model, scaler

def predict(input_features: list):
    model, scaler = load_model_and_scaler()
    features_scaled = scaler.transform([input_features])
    prediction = model.predict(features_scaled)
    return prediction[0]
