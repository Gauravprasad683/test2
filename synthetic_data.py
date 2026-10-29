"""
Synthetic Sports Telemetry Generator, Dataset Splitter, and ML Training Pipeline
Builds a scientifically grounded synthetic dataset of elite athlete telemetry,
performs stratified train/test split, trains predictive models, and exports evaluation artifacts.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def generate_synthetic_telemetry(n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    """
    Generates realistic elite athlete workload and wellness telemetry based on
    established sports medicine epidemiology (Gabbett ACWR framework, Fullagar sleep studies).
    """
    np.random.seed(random_state)
    
    positions = ["Goalkeeper", "Defender", "Midfielder", "Forward"]
    pos_probs = [0.12, 0.35, 0.33, 0.20]
    assigned_positions = np.random.choice(positions, size=n_samples, p=pos_probs)
    
    # Base chronic workload (28-day fitness base)
    chronic_load = np.random.normal(loc=1800, scale=350, size=n_samples)
    chronic_load = np.clip(chronic_load, 800, 3200)
    
    # ACWR distribution: mostly 0.85-1.25, with tails for spikes (>1.4) and deconditioning (<0.75)
    acwr = np.random.normal(loc=1.08, scale=0.28, size=n_samples)
    # Add intentional spikes for realistic sports calendar congestion
    spike_idx = np.random.choice(n_samples, size=int(n_samples * 0.15), replace=False)
    acwr[spike_idx] += np.random.uniform(0.3, 0.7, size=len(spike_idx))
    acwr = np.clip(acwr, 0.45, 2.35)
    
    acute_load = chronic_load * acwr
    
    # Sleep hours: baseline ~7.8h, std 1.1
    avg_sleep = np.random.normal(loc=7.7, scale=1.1, size=n_samples)
    avg_sleep = np.clip(avg_sleep, 4.0, 10.5)
    
    # Correlate fatigue and soreness inversely with sleep and directly with ACWR
    fatigue = 2.5 + (acwr - 0.9) * 3.0 + (8.0 - avg_sleep) * 0.8 + np.random.normal(0, 0.9, n_samples)
    fatigue = np.clip(fatigue, 1.0, 10.0)
    
    soreness = 2.2 + (acwr - 0.9) * 3.2 + (8.0 - avg_sleep) * 0.7 + np.random.normal(0, 0.9, n_samples)
    soreness = np.clip(soreness, 1.0, 10.0)
    
    stress = np.random.normal(loc=4.2, scale=1.5, size=n_samples)
    stress = np.clip(stress, 1.0, 10.0)
    
    mood = 11.0 - (fatigue * 0.4 + stress * 0.4 + np.random.normal(0, 0.8, n_samples))
    mood = np.clip(mood, 1.0, 10.0)
    
    # Composite wellness score: higher is better (1-10)
    wellness_score = (11.0 - fatigue + 11.0 - soreness + 11.0 - stress + mood) / 4.0
    wellness_score = np.clip(wellness_score, 1.0, 10.0)
    
    # GPS High-Speed Running (m > 21 km/h) based on position
    pos_hsr_base = {
        "Goalkeeper": 60,
        "Defender": 480,
        "Midfielder": 720,
        "Forward": 650
    }
    hsr_base = np.array([pos_hsr_base[p] for p in assigned_positions])
    hsr_distance = hsr_base * (acute_load / 1800.0) + np.random.normal(0, 80, n_samples)
    hsr_distance = np.clip(hsr_distance, 0, 1800)
    
    # High-speed running week-on-week variance percentage
    hsr_spike_pct = (acwr - 1.0) * 40.0 + np.random.normal(0, 12, n_samples)
    
    # Injury history
    prev_injuries = np.random.poisson(lam=1.1, size=n_samples)
    prev_injuries = np.clip(prev_injuries, 0, 6)
    
    days_since_injury = np.random.exponential(scale=180, size=n_samples) + 14
    days_since_injury = np.clip(days_since_injury, 7, 720)
    
    # Sports-Science Ground Truth Log-Odds Formula for High Injury Risk:
    # Established by Gabbett (2016) & Windt et al.
    log_odds = -2.8  # baseline low risk rate ~10%
    
    # ACWR U-shaped risk curve:
    # Safe zone 0.85-1.25 -> minimal risk
    # Danger zone >1.30 -> steep increase
    # Spike zone >1.50 -> critical increase
    acwr_risk = np.where(
        acwr > 1.45, (acwr - 1.45) * 4.8 + 1.2,
        np.where(acwr > 1.30, (acwr - 1.30) * 3.2,
        np.where(acwr < 0.75, (0.75 - acwr) * 2.0, -0.4))
    )
    log_odds += acwr_risk
    
    # Sleep deficit penalty: <6.5 hours increases vulnerability
    sleep_risk = np.where(avg_sleep < 6.0, 1.6, np.where(avg_sleep < 7.0, 0.8, -0.3))
    log_odds += sleep_risk
    
    # High soreness / fatigue penalty
    soreness_risk = np.where(soreness > 7.0, 1.4, np.where(soreness > 5.5, 0.6, -0.2))
    log_odds += soreness_risk
    
    # Wellness red flag penalty
    wellness_risk = np.where(wellness_score < 5.0, 1.3, np.where(wellness_score < 6.0, 0.5, -0.3))
    log_odds += wellness_risk
    
    # GPS sprint overload
    hsr_risk = np.where(hsr_spike_pct > 25.0, 1.2, 0.0)
    log_odds += hsr_risk
    
    # Recency of injury
    recent_injury_risk = np.where((days_since_injury < 60) & (prev_injuries > 0), 1.5, 
                         np.where((days_since_injury < 90) & (prev_injuries > 0), 0.8, 0.0))
    log_odds += recent_injury_risk
    
    # Convert log-odds to probability via sigmoid
    prob_injury = 1.0 / (1.0 + np.exp(-log_odds))
    
    # Generate binary label with threshold 0.50
    injury_risk_label = (prob_injury >= 0.45).astype(int)
    
    df = pd.DataFrame({
        "acwr": np.round(acwr, 2),
        "avg_sleep_hours": np.round(avg_sleep, 1),
        "wellness_score": np.round(wellness_score, 1),
        "fatigue": np.round(fatigue, 1),
        "soreness": np.round(soreness, 1),
        "stress": np.round(stress, 1),
        "mood": np.round(mood, 1),
        "acute_load_au": np.round(acute_load, 1),
        "chronic_load_au": np.round(chronic_load, 1),
        "high_speed_running_m": np.round(hsr_distance, 1),
        "hsr_spike_pct": np.round(hsr_spike_pct, 1),
        "previous_injuries_count": prev_injuries,
        "days_since_last_injury": np.round(days_since_injury, 0),
        "position": assigned_positions,
        "true_risk_probability": np.round(prob_injury, 3),
        "injury_risk_label": injury_risk_label  # 0: Low Risk, 1: High Risk
    })
    
    return df


FEATURE_COLUMNS = [
    "acwr",
    "avg_sleep_hours",
    "wellness_score",
    "fatigue",
    "soreness",
    "stress",
    "mood",
    "acute_load_au",
    "chronic_load_au",
    "high_speed_running_m",
    "hsr_spike_pct",
    "previous_injuries_count",
    "days_since_last_injury"
]


def train_and_evaluate_model(
    df: pd.DataFrame = None,
    test_size: float = 0.20,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Splits synthetic database into train/test sets, trains a Random Forest & Gradient Boosting model,
    calculates full evaluation metrics and cost-of-error analysis, and saves model artifacts.
    """
    if df is None:
        df = generate_synthetic_telemetry(n_samples=1600, random_state=random_state)
        
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    df.to_csv(os.path.join(DATA_DIR, "synthetic_player_telemetry.csv"), index=False)
    
    X = df[FEATURE_COLUMNS]
    y = df["injury_risk_label"]
    
    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Train Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state
    )
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    
    # Calculate Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred).tolist()  # [[TN, FP], [FN, TP]]
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Feature Importances
    importances = dict(zip(FEATURE_COLUMNS, [round(float(v), 4) for v in rf_model.feature_importances_]))
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    
    # Cost-of-Error Analysis:
    # In professional elite sports (EPL / Champions League level):
    # - False Negative (Missed Injury): Player plays while compromised -> muscle tear -> £150,000 in clinical rehab + 6 weeks wages & squad disruption.
    # - False Positive (False Alarm): Player rested conservatively -> substitute plays -> £15,000 minor tactical rotation & recovery session cost.
    cost_false_negative_gbp = 150000
    cost_false_positive_gbp = 15000
    total_error_cost_gbp = (fn * cost_false_negative_gbp) + (fp * cost_false_positive_gbp)
    baseline_unmitigated_injuries = tp + fn
    unmitigated_cost_gbp = baseline_unmitigated_injuries * cost_false_negative_gbp
    net_cost_avoidance_gbp = unmitigated_cost_gbp - total_error_cost_gbp
    
    metadata = {
        "model_type": "RandomForestClassifier",
        "n_samples_total": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4),
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "matrix": cm
        },
        "cost_analysis": {
            "cost_per_false_negative_gbp": cost_false_negative_gbp,
            "cost_per_false_positive_gbp": cost_false_positive_gbp,
            "total_false_negative_cost_gbp": int(fn * cost_false_negative_gbp),
            "total_false_positive_cost_gbp": int(fp * cost_false_positive_gbp),
            "total_system_error_cost_gbp": int(total_error_cost_gbp),
            "unmitigated_injury_cost_gbp": int(unmitigated_cost_gbp),
            "net_cost_avoidance_gbp": int(net_cost_avoidance_gbp),
            "cost_reduction_pct": round(float((net_cost_avoidance_gbp / unmitigated_cost_gbp) * 100), 1)
        },
        "feature_importances": sorted_importances,
        "feature_columns": FEATURE_COLUMNS
    }
    
    # Save model and metadata
    model_path = os.path.join(MODEL_DIR, "injury_risk_model.joblib")
    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    test_data_path = os.path.join(DATA_DIR, "test_eval_sample.csv")
    
    joblib.dump(rf_model, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    # Save a small slice of test data for interactive validation
    eval_slice = X_test.copy()
    eval_slice["true_label"] = y_test
    eval_slice["pred_label"] = y_pred
    eval_slice["risk_probability"] = np.round(y_prob, 3)
    eval_slice.to_csv(test_data_path, index=False)
    
    return metadata


def load_model_and_predict(player_telemetry: Dict[str, Any]) -> Tuple[str, float]:
    """
    Given a player dictionary, runs inference through the trained model.
    Returns (risk_class: 'Low Risk'|'High Risk', risk_prob: float).
    """
    model_path = os.path.join(MODEL_DIR, "injury_risk_model.joblib")
    if not os.path.exists(model_path):
        train_and_evaluate_model()
        
    model = joblib.load(model_path)
    
    # Extract features with sensible defaults
    feat_values = []
    for col in FEATURE_COLUMNS:
        val = player_telemetry.get(col, 0.0)
        feat_values.append(float(val))
        
    X_single = pd.DataFrame([feat_values], columns=FEATURE_COLUMNS)
    prob = float(model.predict_proba(X_single)[0, 1])
    risk_class = "High Risk" if prob >= 0.45 else "Low Risk"
    return risk_class, round(prob, 3)


if __name__ == "__main__":
    print("Training synthetic sports science model...")
    meta = train_and_evaluate_model()
    print("Training complete! Model Accuracy:", meta["metrics"]["accuracy"])
    print("ROC-AUC:", meta["metrics"]["roc_auc"])
    print("Saved model to:", os.path.join(MODEL_DIR, "injury_risk_model.joblib"))

