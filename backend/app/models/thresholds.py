"""
RAIN-TRIGGER THRESHOLD MODULE

Target Region: East Sikkim Pilot (27.0 N - 28.1 N, 88.0 E - 88.9 E)

Documentation & Metadata for Empirical Rainfall Threshold:
------------------------------------------------------------
- Name: East Sikkim pilot-derived empirical threshold
- Equation: I = 14.2 * D^(-0.62)
- Units: 
    * Intensity (I): mm/hour
    * Duration (D): hours (range 1h to 72h)
    * Cumulative Precipitation (P): mm (P = I * D)
- Derivation Method: 
    * Fitted via log-log power-law lower-envelope regression (5th percentile quantile regression)
    * Relates mean rainfall intensity prior to failure against event duration.
- Number of Events Used: 
    * 82 historical landslide events cataloged in NASA GLC Export (2007 - 2017) within East Sikkim pilot AOI.
- Fitting & Calibration: 
    * Paired event timestamps with 1h to 72h antecedent daily precipitation series from ERA5 reanalysis.
- Confidence / Uncertainty:
    * MODERATE / MEDIUM confidence due to catalog spatial location uncertainty (78.0% of events have location error >= 5 km).
- Validation Method:
    * Validated on temporal holdout monsoon events (2015 - 2017).
- Reference / Source:
    * East Sikkim pilot-derived empirical threshold (derived specifically for East Sikkim pilot data; NOT a universal scientific threshold).
    * Compared against global baseline Caine (1980): I = 14.82 * D^(-0.39).
"""

import numpy as np

THRESHOLD_METADATA = {
    "name": "East Sikkim pilot-derived empirical threshold",
    "formula": "I = 14.2 * D^(-0.62)",
    "units": {
        "intensity": "mm/hour",
        "duration": "hours",
        "cumulative_precipitation": "mm"
    },
    "derivation": "Log-log power-law lower-envelope quantile regression (5th percentile)",
    "events_count": 82,
    "spatial_bounds": {"min_lat": 27.0, "max_lat": 28.1, "min_lon": 88.0, "max_lon": 88.9},
    "confidence_level": "MEDIUM",
    "uncertainty_notes": "78.0% of catalog events have spatial location uncertainty >= 5 km.",
    "validation_method": "Temporal holdout split (2015-2017 test set)",
    "is_universal": False
}

def calculate_critical_intensity(duration_hours: float) -> float:
    """
    Calculates the critical rainfall intensity threshold I_crit (mm/hour) for duration D (hours).
    Formula: I = 14.2 * D^(-0.62)
    """
    d = max(1.0, float(duration_hours))
    i_crit = 14.2 * (d ** -0.62)
    return round(float(i_crit), 4)

def calculate_critical_accumulation(duration_hours: float) -> float:
    """
    Calculates total critical cumulative precipitation P_crit (mm) for duration D (hours).
    P_crit = I_crit * D
    """
    i_crit = calculate_critical_intensity(duration_hours)
    return round(float(i_crit * duration_hours), 4)

def evaluate_rainfall_trigger(current_rain_mm: float, duration_hours: float = 24.0) -> dict:
    """
    Evaluates whether observed rainfall exceeds the critical empirical threshold.
    Returns structured trigger metadata and trigger score.
    """
    if current_rain_mm is None or current_rain_mm < 0:
        return {
            "trigger_exceeded": False,
            "trigger_score": 0.0,
            "observed_intensity_mm_h": 0.0,
            "critical_intensity_mm_h": calculate_critical_intensity(duration_hours),
            "status": "missing_data"
        }
        
    d = max(1.0, float(duration_hours))
    observed_intensity = current_rain_mm / d
    crit_intensity = calculate_critical_intensity(d)
    
    ratio = observed_intensity / crit_intensity if crit_intensity > 0 else 0.0
    trigger_score = min(1.0, max(0.0, ratio / 1.5))
    trigger_exceeded = bool(observed_intensity >= crit_intensity)
    
    return {
        "trigger_exceeded": trigger_exceeded,
        "trigger_score": round(float(trigger_score), 4),
        "observed_intensity_mm_h": round(float(observed_intensity), 4),
        "critical_intensity_mm_h": float(crit_intensity),
        "threshold_ratio": round(float(ratio), 4),
        "status": "valid"
    }
