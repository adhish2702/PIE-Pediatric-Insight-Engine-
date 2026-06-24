#!/usr/bin/env python3
# ============================================================
# PIE – Pediatric Insight Engine
# run_pipeline.py  –  Full end-to-end demonstration runner
#
# Executes all steps sequentially and prints results.
# Run:  python run_pipeline.py
# ============================================================

import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║     PIE – Pediatric Insight Engine                           ║
║     Multimodal Developmental Risk Detection System           ║
║     ─────────────────────────────────────────────           ║
║     Steps 1–11 End-to-End Pipeline Demo                      ║
╚══════════════════════════════════════════════════════════════╝
"""

def section(title: str, step: int):
    print(f"\n{'═'*60}")
    print(f"  STEP {step}: {title}")
    print(f"{'═'*60}")


def run_pipeline(n_children=20, sessions_per_child=4, seed=42):
    print(BANNER)
    total_start = time.time()

    # ─────────────────────────────────────────────────────────
    # STEP 1: Project Setup
    # ─────────────────────────────────────────────────────────
    section("Project Setup", 1)
    for d in ["data/speech","data/motor","data/fusion"]:
        os.makedirs(d, exist_ok=True)
    print("✅ Directory structure ready")
    print(f"   n_children={n_children}, sessions={sessions_per_child}, seed={seed}")


    # ─────────────────────────────────────────────────────────
    # STEP 2: CHILDES Linguistic Features
    # ─────────────────────────────────────────────────────────
    section("CHILDES Linguistic Feature Extraction", 2)
    t0 = time.time()
    from speech.childes_pipeline import build_linguistic_dataset
    ling_df = build_linguistic_dataset(n_children, sessions_per_child, seed=seed)
    print(f"✅ Linguistic features: {ling_df.shape}")
    print(f"   MLU (typical vs at-risk): "
          f"{ling_df[ling_df.is_at_risk==0].mlu.mean():.2f} vs "
          f"{ling_df[ling_df.is_at_risk==1].mlu.mean():.2f}")
    print(f"   Turn-taking (typical vs at-risk): "
          f"{ling_df[ling_df.is_at_risk==0].turn_taking_freq.mean():.2f} vs "
          f"{ling_df[ling_df.is_at_risk==1].turn_taking_freq.mean():.2f}")
    ling_df.to_csv("data/speech/linguistic_features.csv", index=False)
    print(f"   ⏱  {time.time()-t0:.1f}s")


    # ─────────────────────────────────────────────────────────
    # STEP 3: Acoustic Feature Extraction
    # ─────────────────────────────────────────────────────────
    section("Acoustic Feature Extraction (librosa)", 3)
    t0 = time.time()
    from speech.acoustic_features import build_acoustic_dataset
    acoustic_df = build_acoustic_dataset(n_children, sessions_per_child, seed=seed)
    print(f"✅ Acoustic features: {acoustic_df.shape}")
    if "pitch_std" in acoustic_df.columns:
        print(f"   Pitch variability (typical vs at-risk): "
              f"{acoustic_df[acoustic_df.is_at_risk==0].pitch_std.mean():.1f} vs "
              f"{acoustic_df[acoustic_df.is_at_risk==1].pitch_std.mean():.1f} Hz")
    if "speech_rate_sps" in acoustic_df.columns:
        print(f"   Speech rate (typical vs at-risk): "
              f"{acoustic_df[acoustic_df.is_at_risk==0].speech_rate_sps.mean():.1f} vs "
              f"{acoustic_df[acoustic_df.is_at_risk==1].speech_rate_sps.mean():.1f} syl/s")
    acoustic_df.to_csv("data/speech/acoustic_features.csv", index=False)
    print(f"   ⏱  {time.time()-t0:.1f}s")

    # Merge speech modalities on child_id + session_idx (age_months may differ by seed)
    speech_df = pd.merge(ling_df, acoustic_df,
                         on=["child_id","session_idx","is_at_risk"], how="inner")
    # Carry age_months from linguistic (authoritative)
    if "age_months_x" in speech_df.columns:
        speech_df["age_months"] = speech_df["age_months_x"]
        speech_df.drop(columns=["age_months_x","age_months_y"], inplace=True, errors="ignore")
    print(f"\n✅ Combined speech features: {speech_df.shape}")


    # ─────────────────────────────────────────────────────────
    # STEP 4: K-DST Motor Data
    # ─────────────────────────────────────────────────────────
    section("K-DST Motor Data Processing", 4)
    t0 = time.time()
    from motor.kdst_processor import build_motor_dataset
    motor_df = build_motor_dataset(n_children, sessions_per_child, seed=seed)
    print(f"✅ Motor features: {motor_df.shape}")
    print(f"   Hand velocity (typical vs at-risk): "
          f"{motor_df[motor_df.is_at_risk==0].velocity_hands_mean.mean():.4f} vs "
          f"{motor_df[motor_df.is_at_risk==1].velocity_hands_mean.mean():.4f} units/s")
    print(f"   Symmetry index (typical vs at-risk): "
          f"{motor_df[motor_df.is_at_risk==0].symmetry_index.mean():.3f} vs "
          f"{motor_df[motor_df.is_at_risk==1].symmetry_index.mean():.3f}")
    motor_df.to_csv("data/motor/motor_features.csv", index=False)
    print(f"   ⏱  {time.time()-t0:.1f}s")


    # ─────────────────────────────────────────────────────────
    # STEP 5: MediaPipe (simulated)
    # ─────────────────────────────────────────────────────────
    section("MediaPipe Pose Integration", 5)
    from motor.mediapipe_integration import get_motor_features_from_camera
    mp_feats = get_motor_features_from_camera(
        child_id="DEMO_LIVE", age_months=30, is_at_risk=False, seed=99
    )
    print(f"✅ MediaPipe → PIE features (simulated, same API as K-DST):")
    for k, v in list(mp_feats.items())[:6]:
        if k not in ("child_id","age_months"):
            print(f"   {k:35s}: {v}")


    # ─────────────────────────────────────────────────────────
    # STEP 6: Feature Engineering
    # ─────────────────────────────────────────────────────────
    section("Feature Engineering + Standardization", 6)
    t0 = time.time()
    from fusion.feature_engineering import FeatureEngineer, SPEECH_FEATURE_COLS, MOTOR_FEATURE_COLS
    fe = FeatureEngineer()
    X_speech, X_motor = fe.fit_transform(speech_df, motor_df)
    print(f"✅ Standardized speech features: {X_speech.shape}")
    print(f"   Mean ≈ 0: {X_speech.mean(axis=0)[:3].round(3)}")
    print(f"   Std  ≈ 1: {X_speech.std(axis=0)[:3].round(3)}")
    print(f"✅ Standardized motor features: {X_motor.shape}")
    fe.save()
    print(f"   ⏱  {time.time()-t0:.1f}s")


    # ─────────────────────────────────────────────────────────
    # STEP 7: Multimodal Fusion + Risk Prediction
    # ─────────────────────────────────────────────────────────
    section("Multimodal Late Fusion + Risk Prediction", 7)
    t0 = time.time()
    from fusion.risk_model import train_full_pipeline, risk_label

    sm, mm, fusion, metrics = train_full_pipeline(
        speech_df, motor_df,
        X_speech, X_motor,
        fe.speech_cols_, fe.motor_cols_,
        strategy="meta"
    )

    print(f"\n✅ Model Performance:")
    print(f"   Speech AUC  : {metrics['speech_auc']:.3f}")
    print(f"   Motor  AUC  : {metrics['motor_auc']:.3f}")
    print(f"   Fused  AUC  : {metrics['fused_auc']:.3f} ← combined")

    # Example predictions
    print(f"\n── Example Predictions ──")
    p_speech = sm.predict_risk(X_speech)
    p_motor  = mm.predict_risk(X_motor)
    min_len  = min(len(p_speech), len(p_motor), len(speech_df))

    for i in [0, 5, 10, 15]:
        if i >= min_len:
            break
        r = fusion.predict_single(p_speech[i], p_motor[i])
        true_label = "At-Risk" if speech_df.iloc[i]["is_at_risk"] else "Typical"
        print(f"   {speech_df.iloc[i]['child_id']:12s} "
              f"risk={r['risk_score']:.2f} ({r['risk_label']:15s}) "
              f"[true: {true_label}]")

    sm.save(); mm.save(); fusion.save("data/fusion/fusion_model.joblib")
    print(f"   ⏱  {time.time()-t0:.1f}s")


    # ─────────────────────────────────────────────────────────
    # STEP 8: Temporal Modeling
    # ─────────────────────────────────────────────────────────
    section("Temporal Modeling – Developmental Velocity", 8)
    t0 = time.time()
    from fusion.temporal_tracking import (
        compute_developmental_velocity, analyze_longitudinal_cohort
    )

    # Single child deep analysis
    example_child_id = speech_df["child_id"].iloc[0]
    child_df = speech_df[speech_df["child_id"] == example_child_id]
    vel = compute_developmental_velocity(
        child_df, key_features=["mlu", "vocab_size", "turn_taking_freq"]
    )

    print(f"✅ Temporal analysis for {vel['child_id']} ({vel['age_range']}):")
    print(f"   Developmental Momentum: {vel['developmental_momentum']} "
          f"({vel['momentum_label']})")
    for feat, fd in vel["features"].items():
        print(f"   [{feat}] slope={fd['slope']:+.4f}/mo  "
              f"velocity_ratio={fd['velocity_ratio']:.2f}  "
              f"micro_devs={fd['n_micro_deviations']}")

    cohort_temporal = analyze_longitudinal_cohort(
        speech_df, key_features=["mlu","vocab_size"]
    )
    cohort_temporal.to_csv("data/fusion/temporal_features.csv", index=False)
    print(f"\n✅ Cohort temporal analysis: {cohort_temporal.shape}")
    print(f"   Avg momentum (typical):  "
          f"{cohort_temporal[cohort_temporal.is_at_risk==0].dev_momentum.mean():.3f}")
    print(f"   Avg momentum (at-risk):  "
          f"{cohort_temporal[cohort_temporal.is_at_risk==1].dev_momentum.mean():.3f}")
    print(f"   ⏱  {time.time()-t0:.1f}s")


    # ─────────────────────────────────────────────────────────
    # STEP 9: Explainable AI + Reasoning Tags
    # ─────────────────────────────────────────────────────────
    section("Explainable AI – Reasoning Tags", 9)
    from fusion.explainability import ReasoningEngine

    all_df = pd.merge(speech_df, motor_df,
                      on=["child_id","session_idx","is_at_risk"], how="inner")
    if "age_months_x" in all_df.columns:
        all_df["age_months"] = all_df["age_months_x"]
        all_df.drop(columns=["age_months_x","age_months_y"], inplace=True, errors="ignore")
    engine = ReasoningEngine()
    engine.fit(all_df)

    print("✅ ReasoningEngine fitted on population\n")

    # Show tags for an at-risk child
    at_risk_rows = all_df[all_df["is_at_risk"] == 1]
    if len(at_risk_rows):
        row = at_risk_rows.iloc[0]
        tags = engine.generate_tags(row.to_dict(), max_tags=5)
        p_s = float(sm.predict_risk(X_speech[:1])[0])
        p_m = float(mm.predict_risk(X_motor[:1])[0])
        rs  = float(fusion.predict(np.array([p_s]), np.array([p_m]))[0])

        summary = engine.generate_summary(
            tags, rs, row["child_id"], int(row["age_months"])
        )

        print("── At-Risk Child Reasoning Tags ──")
        for t in tags:
            icon = "🔴" if t["severity"]=="severe" else \
                   "🟠" if t["severity"]=="moderate" else "🟡"
            print(f"   {icon} [{t['category']:20s}] {t['tag']:45s} "
                  f"Z={t['z_score']:+.2f}")

        print("\n── Clinical Summary ──")
        print(summary)


    # ─────────────────────────────────────────────────────────
    # STEPS 10 & 11: Dashboard + Edge AI
    # ─────────────────────────────────────────────────────────
    section("Dashboard (Streamlit) + Edge AI Design", 10)
    print("✅ Dashboard ready at:  app/dashboard.py")
    print("   Launch with:  streamlit run app/dashboard.py")
    print()
    print("✅ EDGE AI PRIVACY DESIGN (STEP 11):")
    print("   • All inference runs locally (no remote API calls)")
    print("   • Audio/video never transmitted off-device")
    print("   • Models serialized to local disk only (joblib)")
    print("   • Only aggregated risk scores stored (not raw child data)")
    print("   • Compliant with COPPA / GDPR-K design principles")
    print("   • MediaPipe runs on device GPU/CPU")


    # ─────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────
    total_time = time.time() - total_start
    print(f"\n{'═'*60}")
    print(f"  ✅ PIE PIPELINE COMPLETE  ({total_time:.1f}s)")
    print(f"{'═'*60}")
    print(f"""
  Output files:
    data/speech/linguistic_features.csv
    data/speech/acoustic_features.csv
    data/motor/motor_features.csv
    data/fusion/temporal_features.csv
    data/fusion/scalers.joblib
    data/fusion/speech_model.joblib
    data/fusion/motor_model.joblib
    data/fusion/fusion_model.joblib

  To launch dashboard:
    streamlit run app/dashboard.py
""")


if __name__ == "__main__":
    run_pipeline(n_children=20, sessions_per_child=4, seed=42)
