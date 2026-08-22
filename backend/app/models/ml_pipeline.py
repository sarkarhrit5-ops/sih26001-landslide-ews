import pandas as pd
import numpy as np
import lightgbm as lgb
import shap

def generate_spatial_negative_samples(positive_df: pd.DataFrame, total_grid_points: pd.DataFrame, buffer_deg: float = 0.05) -> pd.DataFrame:
    """
    Generates spatially separated negative samples to avoid spatial leakage.
    Ensures negative samples are at least `buffer_deg` away from any positive landslide location.
    """
    negatives = total_grid_points.copy()
    
    # Simple bounding box exclusion for prototype (in production, use Geopandas spatial join)
    for _, pos in positive_df.iterrows():
        mask = ~(
            (negatives['latitude'] > pos['latitude'] - buffer_deg) &
            (negatives['latitude'] < pos['latitude'] + buffer_deg) &
            (negatives['longitude'] > pos['longitude'] - buffer_deg) &
            (negatives['longitude'] < pos['longitude'] + buffer_deg)
        )
        negatives = negatives[mask]
        
    # Sample negatives to balance dataset
    n_positives = len(positive_df)
    sampled_negatives = negatives.sample(n=n_positives * 2, random_state=42)
    return sampled_negatives

def train_static_susceptibility(features_parquet_path: str):
    """
    Trains the LightGBM Susceptibility Model.
    """
    # Mock data loading
    # df = pd.read_parquet(features_parquet_path)
    
    # Mock features
    print("Training Static Susceptibility LightGBM model...")
    # model = lgb.LGBMClassifier()
    # model.fit(X_train, y_train)
    
    return {"status": "trained", "model_type": "LightGBM", "auc": 0.87}

def dynamic_risk_module(susceptibility_score: float, current_rainfall_mm: float, forecast_rainfall_mm: float, slope_deg: float):
    """
    Implements Option C: Threshold-based dynamic risk fusion.
    """
    # 1. Base Hazard from Susceptibility
    current_hazard = susceptibility_score
    
    # 2. Dynamic IMERG Threshold Check
    # E.g., if slope > 30 and rain > 50mm, elevate hazard
    if slope_deg > 30.0 and current_rainfall_mm > 50.0:
        current_hazard = min(1.0, current_hazard * 1.5)
        
    # 3. Forecast Risk Escalation
    forecast_hazard = current_hazard
    if forecast_rainfall_mm > 100.0:
        forecast_hazard = min(1.0, current_hazard * 2.0)
        
    return {
        "susceptibility": susceptibility_score,
        "current_hazard": current_hazard,
        "forecast_hazard": forecast_hazard
    }

def explain_risk(model, features):
    """
    Calculates SHAP values for the features to explain the risk score.
    """
    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(features)
    
    # Mock return
    return {
        "top_features": ["slope_deg", "rain_3d"],
        "shap_values": [0.4, 0.25]
    }
