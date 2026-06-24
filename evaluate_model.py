# PIE – Pediatric Insight Engine
# Run:  python evaluate_model.py
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix,
    classification_report, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore")

print("\n" + "="*60)
print("   PIE – Model Loss & Accuracy Evaluation")
print("="*60)

print("\n Loading datasets...")
from speech.childes_pipeline import build_linguistic_dataset
from speech.acoustic_features import build_acoustic_dataset
from motor.kdst_processor     import build_motor_dataset
from fusion.feature_engineering import FeatureEngineer

ling_df     = build_linguistic_dataset(n_children=30, sessions_per_child=5, seed=42)
acoustic_df = build_acoustic_dataset(n_children=30, sessions_per_child=5, seed=42)
motor_df    = build_motor_dataset(n_children=30, sessions_per_child=5, seed=42)

speech_df = pd.merge(ling_df, acoustic_df,
                     on=["child_id","session_idx","is_at_risk"], how="inner")
if "age_months_x" in speech_df.columns:
    speech_df["age_months"] = speech_df["age_months_x"]
    speech_df.drop(columns=["age_months_x","age_months_y"], inplace=True, errors="ignore")

fe = FeatureEngineer()
X_speech, X_motor = fe.fit_transform(speech_df, motor_df)
y_speech = speech_df["is_at_risk"].values
y_motor  = motor_df["is_at_risk"].values

print(f" Speech features : {X_speech.shape}  |  At-risk: {y_speech.sum()}/{len(y_speech)}")
print(f" Motor features  : {X_motor.shape}   |  At-risk: {y_motor.sum()}/{len(y_motor)}")



def evaluate_model(model, X, y, model_name, n_splits=5):
    """
    Run stratified k-fold cross-validation and print all metrics.
    """
    print(f"\n{'─'*60}")
    print(f"   {model_name}")
    print(f"{'─'*60}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results = {
        "accuracy":  [],
        "precision": [],
        "recall":    [],
        "f1":        [],
        "auc_roc":   [],
        "log_loss":  [],
        "avg_prec":  [],
    }

    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        fold_results["accuracy"].append(accuracy_score(y_test, y_pred))
        fold_results["precision"].append(precision_score(y_test, y_pred, zero_division=0))
        fold_results["recall"].append(recall_score(y_test, y_pred, zero_division=0))
        fold_results["f1"].append(f1_score(y_test, y_pred, zero_division=0))
        fold_results["auc_roc"].append(roc_auc_score(y_test, y_prob))
        fold_results["log_loss"].append(log_loss(y_test, y_prob))
        fold_results["avg_prec"].append(average_precision_score(y_test, y_prob))

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

        print(f"  Fold {fold_idx+1}: "
              f"Acc={fold_results['accuracy'][-1]:.3f}  "
              f"F1={fold_results['f1'][-1]:.3f}  "
              f"AUC={fold_results['auc_roc'][-1]:.3f}  "
              f"Loss={fold_results['log_loss'][-1]:.3f}")

    print(f"\n  {'Metric':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'─'*52}")
    for metric, values in fold_results.items():
        label = metric.replace("_", " ").title()
        direction = "↓" if metric == "log_loss" else "↑"
        print(f"  {label+' '+direction:<20} "
              f"{np.mean(values):>8.4f} "
              f"{np.std(values):>8.4f} "
              f"{np.min(values):>8.4f} "
              f"{np.max(values):>8.4f}")

    cm = confusion_matrix(all_y_true, all_y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion Matrix (all folds combined):")
    print(f"  ┌─────────────────────────────┐")
    print(f"  │           Predicted          │")
    print(f"  │         Typical  At-Risk     │")
    print(f"  │ Typical   {tn:4d}    {fp:4d}        │")
    print(f"  │ At-Risk   {fn:4d}    {tp:4d}        │")
    print(f"  └─────────────────────────────┘")
    print(f"  True Positives (caught at-risk) : {tp}")
    print(f"  False Negatives (missed at-risk): {fn}  ← most important to minimize")
    print(f"  False Positives (false alarm)   : {fp}")
    print(f"  True Negatives  (correct typical): {tn}")

    # ── Overfitting check ──
    model.fit(X, y)
    train_pred = model.predict(X)
    train_prob = model.predict_proba(X)[:, 1]
    train_acc  = accuracy_score(y, train_pred)
    train_loss = log_loss(y, train_prob)
    val_acc    = np.mean(fold_results["accuracy"])
    val_loss   = np.mean(fold_results["log_loss"])
    gap        = train_acc - val_acc

    print(f"\n  Overfitting Check:")
    print(f"  {'Metric':<15} {'Train':>8} {'Val (CV)':>10} {'Gap':>8}")
    print(f"  {'─'*43}")
    print(f"  {'Accuracy':<15} {train_acc:>8.4f} {val_acc:>10.4f} {gap:>+8.4f}")
    print(f"  {'Log Loss':<15} {train_loss:>8.4f} {val_loss:>10.4f} {val_loss-train_loss:>+8.4f}")

    if gap > 0.15:
        print(f"   Possible overfitting (gap = {gap:.3f} > 0.15)")
    elif gap > 0.05:
        print(f"   Mild overfitting (gap = {gap:.3f})")
    else:
        print(f"   No significant overfitting (gap = {gap:.3f})")

    return fold_results


speech_model = RandomForestClassifier(
    n_estimators=100, max_depth=6,
    min_samples_leaf=3, class_weight="balanced",
    random_state=42
)
speech_results = evaluate_model(
    speech_model, X_speech, y_speech,
    "Speech Model (RandomForest)"
)

motor_model = GradientBoostingClassifier(
    n_estimators=100, max_depth=4,
    learning_rate=0.1, subsample=0.8,
    random_state=42
)
motor_results = evaluate_model(
    motor_model, X_motor, y_motor,
    "Motor Model (GradientBoosting)"
)

print(f"\n{'─'*60}")
print(f"  Fused Model (Meta-Classifier)")
print(f"{'─'*60}")

from fusion.risk_model import SpeechRiskModel, MotorRiskModel, MultimodalFusion
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
min_len = min(len(y_speech), len(y_motor), X_speech.shape[0], X_motor.shape[0])
X_s = X_speech[:min_len]
X_m = X_motor[:min_len]
y_f = y_speech[:min_len]

fused_metrics = {"accuracy":[],"f1":[],"auc_roc":[],"log_loss":[]}

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_s, y_f)):
    
    sm_fold = CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=100, max_depth=6,
                               class_weight="balanced", random_state=42),
        cv=3, method="sigmoid"
    )
    sm_fold.fit(X_s[train_idx], y_f[train_idx])

    mm_fold = CalibratedClassifierCV(
        GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                   random_state=42),
        cv=3, method="sigmoid"
    )
    mm_fold.fit(X_m[train_idx], y_f[train_idx])

    
    p_s_train = sm_fold.predict_proba(X_s[train_idx])[:,1]
    p_m_train = mm_fold.predict_proba(X_m[train_idx])[:,1]
    p_s_test  = sm_fold.predict_proba(X_s[test_idx])[:,1]
    p_m_test  = mm_fold.predict_proba(X_m[test_idx])[:,1]

    meta = LogisticRegression(C=1.0, random_state=42)
    meta.fit(np.column_stack([p_s_train, p_m_train]), y_f[train_idx])

    y_prob = meta.predict_proba(np.column_stack([p_s_test, p_m_test]))[:,1]
    y_pred = (y_prob > 0.5).astype(int)
    y_test = y_f[test_idx]

    fused_metrics["accuracy"].append(accuracy_score(y_test, y_pred))
    fused_metrics["f1"].append(f1_score(y_test, y_pred, zero_division=0))
    fused_metrics["auc_roc"].append(roc_auc_score(y_test, y_prob))
    fused_metrics["log_loss"].append(log_loss(y_test, y_prob))

    print(f"  Fold {fold_idx+1}: "
          f"Acc={fused_metrics['accuracy'][-1]:.3f}  "
          f"F1={fused_metrics['f1'][-1]:.3f}  "
          f"AUC={fused_metrics['auc_roc'][-1]:.3f}  "
          f"Loss={fused_metrics['log_loss'][-1]:.3f}")

print(f"\n  {'Metric':<20} {'Mean':>8} {'Std':>8}")
print(f"  {'─'*36}")
for metric, values in fused_metrics.items():
    label = metric.replace("_"," ").title()
    print(f"  {label:<20} {np.mean(values):>8.4f} {np.std(values):>8.4f}")


print(f"\n{'='*60}")
print(f"  FINAL MODEL COMPARISON")
print(f"{'='*60}")
print(f"  {'Metric':<15} {'Speech RF':>12} {'Motor GB':>12} {'Fused':>12}")
print(f"  {'─'*51}")

metrics_to_compare = [
    ("Accuracy ↑",  "accuracy"),
    ("F1 Score ↑",  "f1"),
    ("AUC-ROC ↑",   "auc_roc"),
    ("Log Loss ↓",  "log_loss"),
]

for label, key in metrics_to_compare:
    s = np.mean(speech_results[key])
    m = np.mean(motor_results[key])
    f = np.mean(fused_metrics[key])

    vals = [s, m, f]
    best = max(vals) if key != "log_loss" else min(vals)

    def fmt(v): return f"{'→ ' if v==best else '   '}{v:.4f}"
    print(f"  {label:<15} {fmt(s):>12} {fmt(m):>12} {fmt(f):>12}")

print(f"\n  → = best value for that metric")
print(f"\n{'='*60}")
print(f"   EVALUATION COMPLETE")
print(f"{'='*60}\n")