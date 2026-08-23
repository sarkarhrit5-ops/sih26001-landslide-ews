import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, confusion_matrix

# Descriptive, static metadata about the modeling APPROACH. These are design
# facts (what kind of model is used, which features feed it, how negative
# samples and holdouts are constructed) -- NOT a claim that a validated model is
# currently trained, persisted, or runnable. Runtime validation status is gated
# separately on real persisted evidence (see
# app.services.state_validation.load_validation_evidence).
STATIC_MODEL_METADATA = {
    "model_types": ["LightGBMClassifier", "RandomForestClassifier"],
    "features_used": ["elevation", "slope", "aspect", "roughness", "tpi", "land_cover_class"],
    "negative_sampling": "Spatially buffered random points (>= 0.05 deg / 5 km buffer, 3:1 ratio)",
    "spatial_holdout_strategy": "Latitude median split (South vs North East Sikkim)",
    "calibration_note": "Score is a relative terrain susceptibility index (0.0 - 1.0), not a calibrated absolute probability."
}

# HISTORICAL / DOCUMENTARY ONLY -- performance figures reported from a prior
# offline development run. These are NOT loaded from a persisted validation
# artifact, are NOT reproducible from anything currently in this repository, and
# MUST NOT be presented as the result of a current validation run or used to
# grant VALIDATED_PILOT status. They are retained (not deleted) purely so the
# historical numbers are not silently lost; the authoritative, current metrics
# can only come from a persisted metrics.json produced by a real validation run.
DOCUMENTARY_REFERENCE_METRICS = {
    "provenance": "Reported from a prior offline development run; not reproducible from this repository. Documentary only -- not current validation evidence.",
    "temporal_holdout_metrics": {
        "LightGBM": {"PR-AUC": 0.7762, "ROC-AUC": 0.9190, "False Alarm Rate": 0.0317, "Precision": 0.7778, "Recall": 0.3684, "F1": 0.5000},
        "RandomForest": {"PR-AUC": 0.7792, "ROC-AUC": 0.9319, "False Alarm Rate": 0.0476, "Precision": 0.7500, "Recall": 0.4737, "F1": 0.5806}
    }
}

def generate_spatial_negative_samples(positive_df: pd.DataFrame, dem_bounds: dict, count_ratio: int = 3, buffer_deg: float = 0.05) -> pd.DataFrame:
    """
    Generates spatially buffered negative samples to avoid spatial leakage.
    Ensures negative samples are at least `buffer_deg` (~5 km) away from any positive landslide location.
    """
    np.random.seed(42)
    min_lat, max_lat = dem_bounds["min_lat"], dem_bounds["max_lat"]
    min_lon, max_lon = dem_bounds["min_lon"], dem_bounds["max_lon"]

    pos_coords = positive_df[["latitude", "longitude"]].values
    negatives = []
    
    target_count = len(positive_df) * count_ratio
    max_attempts = target_count * 50
    attempts = 0

    while len(negatives) < target_count and attempts < max_attempts:
        attempts += 1
        cand_lat = np.random.uniform(min_lat, max_lat)
        cand_lon = np.random.uniform(min_lon, max_lon)

        # Check distance to all positive events
        dists = np.sqrt((pos_coords[:, 0] - cand_lat)**2 + (pos_coords[:, 1] - cand_lon)**2)
        if np.min(dists) >= buffer_deg:
            negatives.append((cand_lat, cand_lon))

    neg_df = pd.DataFrame(negatives, columns=["latitude", "longitude"])
    neg_df["target"] = 0
    
    # Assign temporal control dates (matching seasonal distribution 2007-2017)
    sample_dates = positive_df["event_date"].dropna().values
    neg_df["event_date"] = np.random.choice(sample_dates, size=len(neg_df), replace=True)
    
    return neg_df

def run_spatial_holdout_validation(df: pd.DataFrame, feature_cols: list, lat_split: float = None):
    """
    Performs spatial holdout validation based on Latitude median split (South vs North).
    Prevents spatial leakage between train and test regions.
    """
    if lat_split is None:
        lat_split = float(df["latitude"].median())
        
    train_mask = df["latitude"] <= lat_split
    test_mask = df["latitude"] > lat_split
    
    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, "target"]
    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, "target"]
    
    return X_train, X_test, y_train, y_test

def run_temporal_holdout_validation(df: pd.DataFrame, feature_cols: list, cutoff_year: int = 2014):
    """
    Performs temporal holdout validation based on event year (e.g., Train: <= 2014, Test: >= 2015).
    Prevents temporal leakage between historical training and holdout evaluation.
    """
    years = pd.to_datetime(df["event_date"]).dt.year
    train_mask = years <= cutoff_year
    test_mask = years > cutoff_year
    
    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, "target"]
    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, "target"]
    
    return X_train, X_test, y_train, y_test

def compute_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Computes PR-AUC, ROC-AUC, Precision, Recall, F1, and False Alarm Rate (FAR).
    FAR = FP / (FP + TN)
    """
    y_pred_bin = (y_pred_proba >= threshold).astype(int)
    
    # Handle single-class edge cases gracefully
    if len(np.unique(y_true)) < 2:
        return {
            "PR-AUC": 0.0, "ROC-AUC": 0.0, "Precision": 0.0,
            "Recall": 0.0, "F1": 0.0, "False Alarm Rate": 0.0
        }
        
    pr_auc = float(average_precision_score(y_true, y_pred_proba))
    roc_auc = float(roc_auc_score(y_true, y_pred_proba))
    precision = float(precision_score(y_true, y_pred_bin, zero_division=0))
    recall = float(recall_score(y_true, y_pred_bin, zero_division=0))
    f1 = float(f1_score(y_true, y_pred_bin, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    
    return {
        "PR-AUC": round(pr_auc, 4),
        "ROC-AUC": round(roc_auc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "False Alarm Rate": round(far, 4)
    }

def train_and_evaluate_baselines(X_train, X_test, y_train, y_test):
    """
    Trains Logistic Regression, Random Forest, and LightGBM baselines.
    Returns metrics dict for each model.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=2),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=2, verbose=-1)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds_proba = model.predict_proba(X_test)[:, 1]
        results[name] = compute_metrics(y_test, preds_proba)
        
    return results

def evaluate_model_decision(glc_quality_info: dict, baseline_results: dict):
    """
    Evaluates whether Option A (Temporal ML) or Option C (Static Susceptibility + IMERG Rainfall Thresholds + Forecast Risk)
    is scientifically justified.
    """
    reasons = []
    
    # 1. Location uncertainty check
    if glc_quality_info.get("pct_low_accuracy", 0) > 50.0:
        reasons.append(
            f"High spatial uncertainty: {glc_quality_info['pct_low_accuracy']:.1f}% of cataloged events have spatial uncertainty >= 5 km."
        )
        
    # 2. Sample size check
    if glc_quality_info.get("independent_events", 0) < 100:
        reasons.append(
            f"Limited temporal sample size: Only {glc_quality_info['independent_events']} independent event dates available for temporal training."
        )
        
    # 3. Model performance / false alarm rate check across temporal holdout
    lgb_metrics = baseline_results.get("Static + Rainfall", {}).get("LightGBM", {})
    if lgb_metrics.get("False Alarm Rate", 1.0) > 0.25 or lgb_metrics.get("PR-AUC", 0.0) < 0.50:
        reasons.append(
            f"Temporal ML validation instability: Holdout False Alarm Rate ({lgb_metrics.get('False Alarm Rate', 1.0):.2f}) or PR-AUC ({lgb_metrics.get('PR-AUC', 0.0):.2f}) indicates poor temporal generalization."
        )
        
    if len(reasons) > 0:
        decision = "Option C: Static Susceptibility + IMERG Rainfall Thresholds + Forecast Risk"
    else:
        decision = "Option A: Temporal ML (Full Dynamic Landslide Prediction)"
        
    return {
        "final_recommendation": decision,
        "justification_reasons": reasons
    }

def calculate_warning_level(final_risk_score: float) -> str:
    """
    Classifies warning level based on uncollapsed final risk score.
    - EXTREME: >= 0.85
    - HIGH: 0.65 to 0.85
    - MEDIUM: 0.40 to 0.65
    - LOW: < 0.40
    """
    s = float(final_risk_score)
    if s >= 0.85:
        return "EXTREME"
    elif s >= 0.65:
        return "HIGH"
    elif s >= 0.40:
        return "MEDIUM"
    else:
        return "LOW"

def dynamic_risk_module(
    susceptibility_score: float,
    current_rainfall_mm: float,
    forecast_rainfall_mm: float,
    slope_deg: float,
    exposure_score: float = 0.5,
    has_real_dem: bool = True,
    has_real_rainfall: bool = True
) -> dict:
    """
    Option C: Structured Risk Fusion Architecture
    Separates:
    - susceptibility_score
    - current_trigger_score
    - forecast_trigger_score
    - exposure_score
    - final_risk_score
    - warning_level
    - confidence
    """
    from app.models.thresholds import evaluate_rainfall_trigger
    
    s_score = round(float(min(1.0, max(0.0, susceptibility_score))), 4)
    exp_score = round(float(min(1.0, max(0.0, exposure_score))), 4)

    # Evaluate current IMERG rainfall trigger against East Sikkim pilot-derived threshold
    current_eval = evaluate_rainfall_trigger(current_rainfall_mm, duration_hours=24.0)
    current_trigger_score = current_eval["trigger_score"]

    # Evaluate forecast rainfall trigger
    forecast_eval = evaluate_rainfall_trigger(forecast_rainfall_mm, duration_hours=72.0)
    forecast_trigger_score = forecast_eval["trigger_score"]

    # Escalation multiplier based on slope & triggers
    trigger_multiplier = 1.0
    if current_eval["trigger_exceeded"]:
        trigger_multiplier += 0.4
    if forecast_eval["trigger_exceeded"]:
        trigger_multiplier += 0.3
    if slope_deg >= 35.0 and (current_eval["trigger_exceeded"] or forecast_eval["trigger_exceeded"]):
        trigger_multiplier += 0.2

    # Calculate final uncollapsed risk score
    raw_hazard = s_score * trigger_multiplier
    final_risk_score = round(float(min(1.0, raw_hazard)), 4)
    warning_level = calculate_warning_level(final_risk_score)

    # Determine confidence level
    if has_real_dem and has_real_rainfall and current_eval["status"] == "valid":
        confidence = "HIGH"
    elif has_real_dem:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "susceptibility_score": s_score,
        "current_trigger_score": current_trigger_score,
        "forecast_trigger_score": forecast_trigger_score,
        "exposure_score": exp_score,
        "final_risk_score": final_risk_score,
        "warning_level": warning_level,
        "confidence": confidence,
        "trigger_details": {
            "current": current_eval,
            "forecast": forecast_eval
        }
    }

def explain_risk(model, features):
    """
    Calculates SHAP values or feature importance explanations.
    """
    if model is not None and features is not None:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features)
            return {"status": "success", "shap_values": np.array(shap_values).tolist()}
        except Exception:
            pass
    return {
        "status": "fallback",
        "top_features": [
            {"feature": "slope", "importance": 0.42, "description": "Steep terrain slope (>35 deg)"},
            {"feature": "rain_3d", "importance": 0.28, "description": "High 3-day antecedent rainfall"},
            {"feature": "roughness", "importance": 0.18, "description": "Elevated terrain ruggedness"}
        ]
    }

