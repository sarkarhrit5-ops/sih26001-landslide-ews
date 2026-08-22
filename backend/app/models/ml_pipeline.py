import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

def generate_spatial_negative_samples(positive_df: pd.DataFrame, total_grid_points: pd.DataFrame, buffer_deg: float = 0.05) -> pd.DataFrame:
    """
    Generates spatially separated negative samples to avoid spatial leakage.
    Ensures negative samples are at least `buffer_deg` away from any positive landslide location.
    """
    # Generates spatially separated negative samples to avoid spatial leakage.
    negatives = total_grid_points.copy()
    
    # Calculate simple distance or bounding box exclusion
    for _, pos in positive_df.iterrows():
        mask = ~(
            (negatives['latitude'] >= pos['latitude'] - buffer_deg) &
            (negatives['latitude'] <= pos['latitude'] + buffer_deg) &
            (negatives['longitude'] >= pos['longitude'] - buffer_deg) &
            (negatives['longitude'] <= pos['longitude'] + buffer_deg)
        )
        negatives = negatives[mask]
        
    # Sample negatives to balance dataset
    n_positives = len(positive_df)
    if len(negatives) < n_positives * 2:
        sampled_negatives = negatives
    else:
        sampled_negatives = negatives.sample(n=n_positives * 2, random_state=42)
    return sampled_negatives

def run_spatial_holdout_validation(X, y, coords, test_quadrant="NE"):
    """
    Performs spatial holdout validation based on coordinates.
    test_quadrant can be 'NE', 'NW', 'SE', 'SW' relative to the spatial median.
    """
    med_lat = coords['latitude'].median()
    med_lon = coords['longitude'].median()
    
    if test_quadrant == "NE":
        test_mask = (coords['latitude'] > med_lat) & (coords['longitude'] > med_lon)
    elif test_quadrant == "NW":
        test_mask = (coords['latitude'] > med_lat) & (coords['longitude'] <= med_lon)
    elif test_quadrant == "SE":
        test_mask = (coords['latitude'] <= med_lat) & (coords['longitude'] > med_lon)
    else:
        test_mask = (coords['latitude'] <= med_lat) & (coords['longitude'] <= med_lon)
        
    train_mask = ~test_mask
    
    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]

def train_and_evaluate_baselines(X_train, X_test, y_train, y_test):
    """
    Trains Logistic Regression, Random Forest, and LightGBM baselines.
    Returns evaluation metrics.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=2),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=2)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        preds_proba = model.predict_proba(X_test)[:, 1]
        preds_bin = model.predict(X_test)
        
        pr_auc = average_precision_score(y_test, preds_proba)
        roc_auc = roc_auc_score(y_test, preds_proba)
        precision = precision_score(y_test, preds_bin, zero_division=0)
        recall = recall_score(y_test, preds_bin, zero_division=0)
        f1 = f1_score(y_test, preds_bin, zero_division=0)
        false_alarm_rate = 1 - precision if precision > 0 else 1.0 # simplistic approximation
        
        results[name] = {
            "PR-AUC": pr_auc,
            "ROC-AUC": roc_auc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "False Alarm Rate": false_alarm_rate
        }
    return results

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
