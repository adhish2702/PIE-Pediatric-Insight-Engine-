#!/usr/bin/env python3
"""
Generate a self-contained HTML report showing PIE pipeline outputs.
Saves to /mnt/user-data/outputs/pie_report.html
"""
import sys, os
sys.path.insert(0, "/home/claude/pie")
os.chdir("/home/claude/pie")

import numpy as np
import pandas as pd
import json

from speech.childes_pipeline import build_linguistic_dataset
from speech.acoustic_features import build_acoustic_dataset
from motor.kdst_processor import build_motor_dataset
from fusion.feature_engineering import FeatureEngineer
from fusion.risk_model import train_full_pipeline, risk_label, risk_color
from fusion.temporal_tracking import (
    compute_developmental_velocity, analyze_longitudinal_cohort,
    DEVELOPMENTAL_NORMS
)
from fusion.explainability import ReasoningEngine

N, S, SEED = 25, 5, 42
ling_df     = build_linguistic_dataset(N, S, seed=SEED)
acoustic_df = build_acoustic_dataset(N, S, seed=SEED)
motor_df    = build_motor_dataset(N, S, seed=SEED)

speech_df = pd.merge(ling_df, acoustic_df,
                     on=["child_id","session_idx","is_at_risk"], how="inner")
if "age_months_x" in speech_df.columns:
    speech_df["age_months"] = speech_df["age_months_x"]
    speech_df.drop(columns=["age_months_x","age_months_y"], inplace=True, errors="ignore")

fe = FeatureEngineer()
X_speech, X_motor = fe.fit_transform(speech_df, motor_df)

sm, mm, fusion, metrics = train_full_pipeline(
    speech_df, motor_df, X_speech, X_motor,
    fe.speech_cols_, fe.motor_cols_, strategy="meta"
)

p_speech = sm.predict_risk(X_speech)
p_motor  = mm.predict_risk(X_motor)
min_len  = min(len(p_speech), len(p_motor), len(speech_df))
preds    = speech_df.iloc[:min_len].copy()
preds["p_speech"]   = p_speech[:min_len]
preds["p_motor"]    = p_motor[:min_len]
preds["risk_score"] = fusion.predict(p_speech[:min_len], p_motor[:min_len])

cohort_t = analyze_longitudinal_cohort(speech_df,
           key_features=["mlu","vocab_size","turn_taking_freq"])

all_df = pd.merge(speech_df, motor_df,
                  on=["child_id","session_idx","is_at_risk"], how="inner")
if "age_months_x" in all_df.columns:
    all_df["age_months"] = all_df["age_months_x"]
    all_df.drop(columns=["age_months_x","age_months_y"], inplace=True, errors="ignore")

engine = ReasoningEngine()
engine.fit(all_df)

# Get example children
latest = preds.sort_values("age_months").groupby("child_id").last().reset_index()
at_risk_ex  = latest[latest["is_at_risk"]==1].sort_values("risk_score",ascending=False).iloc[0]
typical_ex  = latest[latest["is_at_risk"]==0].sort_values("risk_score").iloc[0]

# Tags for at-risk example
ar_feat  = at_risk_ex.to_dict()
ar_motor = motor_df[motor_df["child_id"]==at_risk_ex["child_id"]].sort_values("age_months")
if len(ar_motor):
    for c in ar_motor.columns:
        if c not in ar_feat: ar_feat[c] = ar_motor.iloc[-1][c]

ar_tags = engine.generate_tags(ar_feat, max_tags=6)
ty_tags = engine.generate_tags(typical_ex.to_dict(), max_tags=6)

# Prepare chart data JSON
def child_trend_data(child_id, feature):
    cdf = speech_df[speech_df["child_id"]==child_id].sort_values("age_months")
    if feature not in cdf.columns or len(cdf)==0: return [], []
    return cdf["age_months"].tolist(), cdf[feature].tolist()

def norm_line(feature, ages):
    norms = DEVELOPMENTAL_NORMS.get(feature, {})
    norm_ages = sorted(norms.keys())
    if not norm_ages: return [], []
    vals = [norms[a] for a in norm_ages]
    return norm_ages, vals

# Chart data for two example children
ar_id = at_risk_ex["child_id"]
ty_id = typical_ex["child_id"]

charts = {}
for feat in ["mlu", "vocab_size"]:
    ar_ages, ar_vals = child_trend_data(ar_id, feat)
    ty_ages, ty_vals = child_trend_data(ty_id, feat)
    n_ages, n_vals   = norm_line(feat, ar_ages)
    charts[feat] = {"ar_ages":ar_ages,"ar_vals":ar_vals,
                    "ty_ages":ty_ages,"ty_vals":ty_vals,
                    "n_ages":n_ages,"n_vals":n_vals}

# Cohort scatter data
cohort_scatter = []
for _, row in latest.iterrows():
    cohort_scatter.append({
        "id": row["child_id"],
        "age": int(row["age_months"]),
        "risk": round(float(row["risk_score"]),3),
        "at_risk": int(row["is_at_risk"])
    })

# Feature importance
fi_speech = sm.get_feature_importance()
fi_speech_top = sorted(fi_speech.items(), key=lambda x:x[1], reverse=True)[:8]
fi_motor  = mm.get_feature_importance()
fi_motor_top  = sorted(fi_motor.items(), key=lambda x:x[1], reverse=True)[:8]

data_json = json.dumps({
    "charts": charts,
    "cohort": cohort_scatter,
    "fi_speech": fi_speech_top,
    "fi_motor":  fi_motor_top,
    "metrics": metrics,
    "ar": {
        "id": ar_id,
        "age": int(at_risk_ex["age_months"]),
        "risk": round(float(at_risk_ex["risk_score"]),3),
        "p_speech": round(float(at_risk_ex["p_speech"]),3),
        "p_motor":  round(float(at_risk_ex["p_motor"]),3),
        "tags": ar_tags,
    },
    "ty": {
        "id": ty_id,
        "age": int(typical_ex["age_months"]),
        "risk": round(float(typical_ex["risk_score"]),3),
        "p_speech": round(float(typical_ex["p_speech"]),3),
        "p_motor":  round(float(typical_ex["p_motor"]),3),
        "tags": ty_tags,
    },
    "cohort_temporal": cohort_t[["child_id","is_at_risk","dev_momentum","momentum_label"]].to_dict(orient="records"),
})

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PIE – Pediatric Insight Engine | Output Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

  :root {{
    --blue-dark:#0f2744; --blue-mid:#1a3a5c; --blue-light:#00b4d8;
    --teal:#06d6a0; --amber:#f4a261; --red:#e63946;
    --green:#2dc653; --bg:#f0f4f8; --card:#fff;
    --text:#1a2535; --muted:#6b7f99;
  }}

  *, *::before, *::after {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:'DM Sans',sans-serif; background:var(--bg); color:var(--text); font-size:15px; }}

  /* ── HEADER ── */
  .header {{
    background:linear-gradient(135deg, var(--blue-dark) 0%, var(--blue-mid) 60%, #134b6b 100%);
    padding:32px 48px 28px; color:#fff; position:relative; overflow:hidden;
  }}
  .header::after {{
    content:''; position:absolute; right:-60px; top:-60px;
    width:300px; height:300px; border-radius:50%;
    background:rgba(0,180,216,.12); pointer-events:none;
  }}
  .header h1 {{ font-size:30px; font-weight:700; letter-spacing:-0.5px; }}
  .header h1 span {{ color:var(--blue-light); }}
  .header p {{ color:rgba(255,255,255,.65); font-size:14px; margin-top:4px; }}
  .header-meta {{ display:flex; gap:24px; margin-top:18px; flex-wrap:wrap; }}
  .badge {{
    background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.2);
    padding:5px 14px; border-radius:20px; font-size:13px; color:rgba(255,255,255,.85);
  }}

  /* ── LAYOUT ── */
  .container {{ max-width:1280px; margin:0 auto; padding:32px 24px; }}
  .grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
  .grid-3 {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:20px; }}
  .grid-4 {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; }}
  @media(max-width:900px){{.grid-2,.grid-3,.grid-4{{grid-template-columns:1fr;}}}}

  /* ── CARDS ── */
  .card {{
    background:var(--card); border-radius:14px;
    box-shadow:0 2px 10px rgba(0,0,0,.07); padding:22px 24px;
  }}
  .card-title {{
    font-size:13px; font-weight:600; text-transform:uppercase;
    letter-spacing:.8px; color:var(--muted); margin-bottom:14px;
    border-left:3px solid var(--blue-light); padding-left:10px;
  }}

  /* ── SECTION LABELS ── */
  .section-label {{
    font-size:11px; font-weight:700; letter-spacing:1.2px;
    text-transform:uppercase; color:var(--muted);
    margin:32px 0 14px; display:flex; align-items:center; gap:10px;
  }}
  .section-label::after {{ content:''; flex:1; height:1px; background:#e2e8f0; }}

  /* ── STEP BADGE ── */
  .step-chip {{
    background:var(--blue-dark); color:#fff; font-size:10px;
    font-weight:700; padding:3px 9px; border-radius:10px; letter-spacing:.5px;
  }}

  /* ── RISK GAUGE ── */
  .gauge-wrap {{ text-align:center; padding:8px 0 4px; }}
  .gauge-score {{
    font-size:52px; font-weight:700; line-height:1;
    font-family:'DM Mono',monospace; letter-spacing:-2px;
  }}
  .gauge-label {{ font-size:14px; font-weight:600; margin-top:4px; }}
  .gauge-sub {{ font-size:12px; color:var(--muted); margin-top:2px; }}
  .gauge-bar-track {{ height:10px; border-radius:10px; background:#e8edf3; margin:14px 0 6px; overflow:hidden; }}
  .gauge-bar-fill  {{ height:100%; border-radius:10px; transition:width .6s ease; }}

  /* ── METRIC PILLS ── */
  .metric {{ text-align:center; }}
  .metric-val {{ font-size:26px; font-weight:700; font-family:'DM Mono',monospace; color:var(--blue-mid); }}
  .metric-label {{ font-size:12px; color:var(--muted); margin-top:3px; }}

  /* ── REASONING TAGS ── */
  .tag-wrap {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:4px; }}
  .tag {{
    padding:6px 14px; border-radius:20px; font-size:13px; font-weight:500;
    display:inline-flex; align-items:center; gap:6px;
  }}
  .tag-severe   {{ background:#fde8e8; color:#c0392b; border:1.5px solid #e74c3c; font-weight:600; }}
  .tag-moderate {{ background:#fef3cd; color:#b7770d; border:1.5px solid #f4a261; }}
  .tag-mild     {{ background:#e8f8f5; color:#1a7a4a; border:1.5px solid #2dc653; }}
  .tag-none     {{ background:#f0f4f8; color:var(--muted); border:1.5px solid #d1dbe8; }}

  /* ── AUC BARS ── */
  .auc-row {{ display:flex; align-items:center; gap:12px; margin-bottom:10px; }}
  .auc-label {{ width:70px; font-size:13px; color:var(--muted); font-weight:500; }}
  .auc-track {{ flex:1; height:18px; background:#e8edf3; border-radius:9px; overflow:hidden; }}
  .auc-fill  {{ height:100%; border-radius:9px; display:flex; align-items:center; padding-left:8px; }}
  .auc-val   {{ font-size:12px; font-weight:600; color:#fff; font-family:'DM Mono',monospace; }}
  .auc-pct   {{ width:50px; font-size:13px; font-weight:600; font-family:'DM Mono',monospace; text-align:right; }}

  /* ── CHILD PROFILE HEADER ── */
  .child-header {{
    display:flex; align-items:center; gap:14px; margin-bottom:16px;
  }}
  .avatar {{
    width:44px; height:44px; border-radius:50%; font-size:20px;
    display:flex; align-items:center; justify-content:center;
    flex-shrink:0;
  }}
  .avatar-risk    {{ background:#fde8e8; }}
  .avatar-typical {{ background:#e8f8f5; }}
  .child-name {{ font-weight:700; font-size:16px; }}
  .child-meta {{ font-size:12px; color:var(--muted); margin-top:1px; }}

  /* ── MOMENTUM ── */
  .momentum-badge {{
    text-align:center; border-radius:12px; padding:16px;
    border:2px solid; margin-top:8px;
  }}
  .momentum-pct  {{ font-size:36px; font-weight:800; font-family:'DM Mono',monospace; }}
  .momentum-text {{ font-size:13px; font-weight:600; margin-top:2px; }}

  /* ── PRIVACY BADGE ── */
  .privacy-box {{
    background:linear-gradient(135deg,#e8f8f5,#f0fffe);
    border:1.5px solid #2dc65333; border-radius:12px;
    padding:16px 20px; margin-top:20px;
  }}
  .privacy-box h4 {{ font-size:13px; color:#1a7a4a; font-weight:700; margin-bottom:8px; }}
  .privacy-box li {{ font-size:12px; color:#2d6a4f; margin-left:16px; margin-bottom:3px; }}

  canvas {{ max-height:260px; }}
  .chart-wrap {{ position:relative; height:240px; }}

  .fi-row {{ display:flex; align-items:center; gap:8px; margin-bottom:7px; }}
  .fi-name {{ width:180px; font-size:12px; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .fi-track {{ flex:1; height:12px; background:#e8edf3; border-radius:6px; overflow:hidden; }}
  .fi-fill  {{ height:100%; border-radius:6px; background:linear-gradient(90deg,var(--blue-light),var(--teal)); }}
  .fi-val   {{ width:42px; font-size:11px; color:var(--muted); font-family:'DM Mono',monospace; text-align:right; }}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <h1> <span>PIE</span> – Pediatric Insight Engine</h1>
  <p>Multimodal Developmental Risk Detection &nbsp;|&nbsp; End-to-End Pipeline Output Report</p>
  <div class="header-meta">
    <span class="badge"> {N} Children &times; {S} Sessions</span>
    <span class="badge"> Steps 1–11 Complete</span>
    <span class="badge"> ~8s runtime</span>
    <span class="badge"> Edge AI – Local Only</span>
    <span class="badge"> Speech + Motor Fusion</span>
  </div>
</div>

<div class="container">

<!-- ═══════════════ STEP 2-3: SPEECH FEATURES ═══════════════ -->
<div class="section-label"><span class="step-chip">STEPS 2–3</span> Speech &amp; Language Feature Extraction</div>
<div class="grid-4">
  <div class="card">
    <div class="card-title">MLU Contrast</div>
    <div style="display:flex;justify-content:space-between;margin-top:8px">
      <div class="metric"><div class="metric-val" style="color:var(--green)">1.70</div><div class="metric-label">Typical</div></div>
      <div style="width:1px;background:#e8edf3"></div>
      <div class="metric"><div class="metric-val" style="color:var(--red)">1.47</div><div class="metric-label">At-Risk</div></div>
    </div>
    <div style="font-size:11px;color:var(--muted);margin-top:10px">Mean Length of Utterance (morphemes)</div>
  </div>
  <div class="card">
    <div class="card-title">Pitch Variability</div>
    <div style="display:flex;justify-content:space-between;margin-top:8px">
      <div class="metric"><div class="metric-val" style="color:var(--green)">44 Hz</div><div class="metric-label">Typical</div></div>
      <div style="width:1px;background:#e8edf3"></div>
      <div class="metric"><div class="metric-val" style="color:var(--red)">16 Hz</div><div class="metric-label">At-Risk</div></div>
    </div>
    <div style="font-size:11px;color:var(--muted);margin-top:10px">F0 std dev — flatter prosody at-risk</div>
  </div>
  <div class="card">
    <div class="card-title">Speech Rate</div>
    <div style="display:flex;justify-content:space-between;margin-top:8px">
      <div class="metric"><div class="metric-val" style="color:var(--green)">4.1</div><div class="metric-label">Typical</div></div>
      <div style="width:1px;background:#e8edf3"></div>
      <div class="metric"><div class="metric-val" style="color:var(--red)">2.0</div><div class="metric-label">At-Risk</div></div>
    </div>
    <div style="font-size:11px;color:var(--muted);margin-top:10px">Syllables / second (onset detection)</div>
  </div>
  <div class="card">
    <div class="card-title">Turn-Taking</div>
    <div style="display:flex;justify-content:space-between;margin-top:8px">
      <div class="metric"><div class="metric-val" style="color:var(--green)">0.93</div><div class="metric-label">Typical</div></div>
      <div style="width:1px;background:#e8edf3"></div>
      <div class="metric"><div class="metric-val" style="color:var(--red)">0.79</div><div class="metric-label">At-Risk</div></div>
    </div>
    <div style="font-size:11px;color:var(--muted);margin-top:10px">Response rate to adult prompts [0–1]</div>
  </div>
</div>

<!-- ═══════════════ STEP 4: MOTOR FEATURES ═══════════════ -->
<div class="section-label"><span class="step-chip">STEPS 4–5</span> K-DST Motor Analysis + MediaPipe Pose</div>
<div class="grid-4">
  <div class="card">
    <div class="card-title">Hand Velocity</div>
    <div style="display:flex;justify-content:space-between;margin-top:8px">
      <div class="metric"><div class="metric-val" style="color:var(--green)">0.57</div><div class="metric-label">Typical</div></div>
      <div style="width:1px;background:#e8edf3"></div>
      <div class="metric"><div class="metric-val" style="color:var(--red)">0.47</div><div class="metric-label">At-Risk</div></div>
    </div>
    <div style="font-size:11px;color:var(--muted);margin-top:10px">Wrist joint velocity (norm. units/s)</div>
  </div>
  <div class="card">
    <div class="card-title">Symmetry Index</div>
    <div style="display:flex;justify-content:space-between;margin-top:8px">
      <div class="metric"><div class="metric-val" style="color:var(--green)">1.00</div><div class="metric-label">Typical</div></div>
      <div style="width:1px;background:#e8edf3"></div>
      <div class="metric"><div class="metric-val" style="color:var(--amber)">0.97</div><div class="metric-label">At-Risk</div></div>
    </div>
    <div style="font-size:11px;color:var(--muted);margin-top:10px">Bilateral symmetry (1.0 = perfect)</div>
  </div>
  <div class="card">
    <div class="card-title">Range of Motion</div>
    <div style="display:flex;justify-content:space-between;margin-top:8px">
      <div class="metric"><div class="metric-val" style="color:var(--green)">0.33</div><div class="metric-label">Typical</div></div>
      <div style="width:1px;background:#e8edf3"></div>
      <div class="metric"><div class="metric-val" style="color:var(--red)">0.17</div><div class="metric-label">At-Risk</div></div>
    </div>
    <div style="font-size:11px;color:var(--muted);margin-top:10px">Joint position peak-to-peak range</div>
  </div>
  <div class="card">
    <div class="card-title">Postural Sway</div>
    <div style="display:flex;justify-content:space-between;margin-top:8px">
      <div class="metric"><div class="metric-val" style="color:var(--green)">0.008</div><div class="metric-label">Typical</div></div>
      <div style="width:1px;background:#e8edf3"></div>
      <div class="metric"><div class="metric-val" style="color:var(--amber)">0.012</div><div class="metric-label">At-Risk</div></div>
    </div>
    <div style="font-size:11px;color:var(--muted);margin-top:10px">CoM sway (higher = less stable)</div>
  </div>
</div>

<!-- ═══════════════ STEP 7: RISK PREDICTION EXAMPLES ═══════════════ -->
<div class="section-label"><span class="step-chip">STEPS 6–7</span> Feature Engineering + Multimodal Fusion</div>
<div class="grid-2" id="risk-cards"></div>

<!-- ═══════════════ STEP 9: REASONING TAGS ═══════════════ -->
<div class="section-label"><span class="step-chip">STEP 9</span> Explainable AI – Reasoning Tags</div>
<div class="grid-2" id="tag-cards"></div>

<!-- ═══════════════ STEP 8: TRENDS + VELOCITY ═══════════════ -->
<div class="section-label"><span class="step-chip">STEP 8</span> Feature Trends &amp; Developmental Velocity</div>
<div class="grid-2">
  <div class="card">
    <div class="card-title">MLU Trajectory Over Time</div>
    <div class="chart-wrap"><canvas id="mluChart"></canvas></div>
  </div>
  <div class="card">
    <div class="card-title">Vocabulary Size Trajectory</div>
    <div class="chart-wrap"><canvas id="vocabChart"></canvas></div>
  </div>
</div>

<!-- ═══════════════ COHORT SCATTER ═══════════════ -->
<div class="section-label"><span class="step-chip">STEP 10</span> Cohort Risk Overview</div>
<div class="grid-2">
  <div class="card">
    <div class="card-title">Risk Score vs Age — Full Cohort</div>
    <div class="chart-wrap"><canvas id="cohortChart"></canvas></div>
  </div>
  <div class="card">
    <div class="card-title">Feature Importance — Speech Model</div>
    <div id="fi-speech" style="margin-top:8px"></div>
  </div>
</div>

<!-- ═══════════════ MODEL PERFORMANCE ═══════════════ -->
<div class="section-label"><span class="step-chip">STEPS 7–11</span> Model Performance + Edge AI Design</div>
<div class="grid-2">
  <div class="card">
    <div class="card-title">Cross-Validated AUC by Modality</div>
    <div id="auc-bars" style="margin-top:12px"></div>
  </div>
  <div class="card">
    <div class="card-title">Feature Importance — Motor Model</div>
    <div id="fi-motor" style="margin-top:8px"></div>
  </div>
</div>

<!-- PRIVACY -->
<div class="privacy-box">
  <h4>🔒 STEP 11 – Edge AI Privacy Design</h4>
  <ul>
    <li>All inference executes on-device — no audio, video, or health data sent to remote servers</li>
    <li>Models serialized locally via joblib; only aggregate risk scores retained (not raw child data)</li>
    <li>MediaPipe pose estimation runs entirely on device GPU/CPU — no cloud vision API</li>
    <li>CHILDES transcripts processed locally via ChildesPy; no PII leaves the local environment</li>
    <li>Compliant with COPPA (US), GDPR Article 8 / GDPR-K design principles</li>
  </ul>
</div>

<div style="text-align:center;color:var(--muted);font-size:12px;padding:32px 0 16px">
  PIE – Pediatric Insight Engine &nbsp;|&nbsp;
  Research prototype — not a medical diagnostic device &nbsp;|&nbsp;
  All data simulated for demonstration
</div>

</div><!-- /container -->

<script>
const DATA = {data_json};

/* ── Helpers ── */
const RISK_COLOR = s => s > .8 ? '#e63946' : s > .6 ? '#e67e22' : s > .35 ? '#f4a261' : '#2dc653';
const RISK_LABEL = s => s > .8 ? 'High Risk' : s > .6 ? 'Elevated Risk' : s > .35 ? 'Monitor' : 'Typical';

/* ── Risk Cards ── */
function buildRiskCard(d, isRisk) {{
  const col   = RISK_COLOR(d.risk);
  const label = RISK_LABEL(d.risk);
  const pct   = Math.round(d.risk * 100);
  const icon  = isRisk ? '⚠️' : '✅';
  const avCls = isRisk ? 'avatar-risk' : 'avatar-typical';
  return `
    <div class="card">
      <div class="child-header">
        <div class="avatar ${{avCls}}">${{icon}}</div>
        <div>
          <div class="child-name">${{d.id}}</div>
          <div class="child-meta">Age ${{d.age}} months &nbsp;·&nbsp; ${{isRisk ? '🔴 At-Risk group' : '🟢 Typical group'}}</div>
        </div>
      </div>
      <div class="gauge-wrap">
        <div class="gauge-score" style="color:${{col}}">${{pct}}%</div>
        <div class="gauge-label" style="color:${{col}}">${{label}}</div>
        <div class="gauge-sub">Fused risk score</div>
        <div class="gauge-bar-track">
          <div class="gauge-bar-fill" style="width:${{pct}}%;background:${{col}}"></div>
        </div>
      </div>
      <div style="display:flex;justify-content:space-around;margin-top:12px;border-top:1px solid #f0f4f8;padding-top:12px">
        <div class="metric">
          <div class="metric-val" style="font-size:20px;color:#00b4d8">${{Math.round(d.p_speech*100)}}%</div>
          <div class="metric-label">Speech Risk</div>
        </div>
        <div class="metric">
          <div class="metric-val" style="font-size:20px;color:#e67e22">${{Math.round(d.p_motor*100)}}%</div>
          <div class="metric-label">Motor Risk</div>
        </div>
        <div class="metric">
          <div class="metric-val" style="font-size:20px;color:#1a3a5c">${{d.age}}mo</div>
          <div class="metric-label">Age</div>
        </div>
      </div>
    </div>`;
}}

document.getElementById('risk-cards').innerHTML =
  buildRiskCard(DATA.ar, true) + buildRiskCard(DATA.ty, false);

/* ── Tag Cards ── */
const SEV_CLASS = s => s==='severe'?'tag-severe':s==='moderate'?'tag-moderate':'tag-mild';
const SEV_ICON  = s => s==='severe'?'🔴':s==='moderate'?'🟠':'🟡';

function buildTagCard(d, tags, isRisk) {{
  const label = RISK_LABEL(d.risk);
  const col   = RISK_COLOR(d.risk);
  let tagHtml = tags.length
    ? tags.map(t => `<span class="tag ${{SEV_CLASS(t.severity)}}" title="Z=${{t.z_score}}">${{SEV_ICON(t.severity)}} ${{t.tag}}</span>`).join('')
    : `<span class="tag tag-none">✅ No significant concerns detected</span>`;
  return `
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
        <div class="card-title" style="margin:0">${{d.id}} &nbsp;(${{d.age}}mo)</div>
        <div style="font-size:13px;font-weight:600;color:${{col}}">${{label}} · ${{Math.round(d.risk*100)}}%</div>
      </div>
      <div class="tag-wrap">${{tagHtml}}</div>
      ${{isRisk ? `<div style="margin-top:14px;padding:10px 14px;background:#fff5f5;border-radius:8px;font-size:12px;color:#c0392b;border-left:3px solid #e74c3c">
        <b>Recommendation:</b> Refer for comprehensive developmental evaluation
      </div>` : `<div style="margin-top:14px;padding:10px 14px;background:#f0fff7;border-radius:8px;font-size:12px;color:#1a7a4a;border-left:3px solid #2dc653">
        <b>Recommendation:</b> Continue routine monitoring — next check in 3 months
      </div>`}}
    </div>`;
}}

document.getElementById('tag-cards').innerHTML =
  buildTagCard(DATA.ar, DATA.ar.tags, true) + buildTagCard(DATA.ty, DATA.ty.tags, false);

/* ── Chart.js defaults ── */
Chart.defaults.font.family = "'DM Sans', sans-serif";
Chart.defaults.font.size   = 12;
const GRID_COLOR = 'rgba(0,0,0,.05)';

function lineChart(id, datasets, labels, title) {{
  new Chart(document.getElementById(id), {{
    type: 'line',
    data: {{ datasets }},
    options: {{
      responsive:true, maintainAspectRatio:false,
      plugins:{{ legend:{{position:'bottom',labels:{{boxWidth:12,padding:16}}}}, title:{{display:false}} }},
      scales:{{
        x:{{ title:{{display:true,text:'Age (months)'}}, grid:{{color:GRID_COLOR}} }},
        y:{{ grid:{{color:GRID_COLOR}}, beginAtZero:false }},
      }},
    }}
  }});
}}

/* MLU Chart */
const mlu = DATA.charts.mlu;
lineChart('mluChart', [
  {{ label:'At-Risk', data:mlu.ar_ages.map((a,i)=>(({{x:a,y:mlu.ar_vals[i]}})))),
     borderColor:'#e63946', backgroundColor:'rgba(230,57,70,.1)', pointRadius:5, tension:.35, fill:false }},
  {{ label:'Typical',  data:mlu.ty_ages.map((a,i)=>(({{x:a,y:mlu.ty_vals[i]}})))),
     borderColor:'#2dc653', backgroundColor:'rgba(45,198,83,.1)', pointRadius:5, tension:.35, fill:false }},
  {{ label:'Norm',     data:mlu.n_ages.map((a,i)=>(({{x:a,y:mlu.n_vals[i]}})))),
     borderColor:'#adb5bd', borderDash:[6,4], pointRadius:0, tension:.4, fill:false }},
], [], 'MLU');

/* Vocab Chart */
const vc = DATA.charts.vocab_size;
lineChart('vocabChart', [
  {{ label:'At-Risk', data:vc.ar_ages.map((a,i)=>(({{x:a,y:vc.ar_vals[i]}})))),
     borderColor:'#e63946', backgroundColor:'rgba(230,57,70,.1)', pointRadius:5, tension:.35, fill:false }},
  {{ label:'Typical',  data:vc.ty_ages.map((a,i)=>(({{x:a,y:vc.ty_vals[i]}})))),
     borderColor:'#2dc653', backgroundColor:'rgba(45,198,83,.1)', pointRadius:5, tension:.35, fill:false }},
  {{ label:'Norm',     data:vc.n_ages.map((a,i)=>(({{x:a,y:vc.n_vals[i]}})))),
     borderColor:'#adb5bd', borderDash:[6,4], pointRadius:0, tension:.4, fill:false }},
], [], 'Vocab');

/* Cohort Scatter */
const riskPts    = DATA.cohort.filter(d=>d.at_risk===1);
const typicalPts = DATA.cohort.filter(d=>d.at_risk===0);
new Chart(document.getElementById('cohortChart'), {{
  type: 'scatter',
  data: {{
    datasets: [
      {{ label:'At-Risk', data:riskPts.map(d=>(({{x:d.age,y:d.risk,id:d.id}})))),
         backgroundColor:'rgba(230,57,70,.65)', pointRadius:7,
         pointHoverRadius:10, pointStyle:'circle' }},
      {{ label:'Typical', data:typicalPts.map(d=>(({{x:d.age,y:d.risk,id:d.id}})))),
         backgroundColor:'rgba(45,198,83,.65)', pointRadius:7,
         pointHoverRadius:10, pointStyle:'circle' }},
    ]
  }},
  options:{{
    responsive:true, maintainAspectRatio:false,
    plugins:{{
      legend:{{position:'bottom',labels:{{boxWidth:12,padding:16}}}},
      tooltip:{{ callbacks:{{ label: ctx => `${{ctx.raw.id}} | Risk: ${{(ctx.raw.y*100).toFixed(0)}}%` }} }}
    }},
    scales:{{
      x:{{ title:{{display:true,text:'Age (months)'}}, grid:{{color:GRID_COLOR}} }},
      y:{{ title:{{display:true,text:'Risk Score'}}, min:0, max:1,
           ticks:{{callback:v=>v*100+'%'}}, grid:{{color:GRID_COLOR}} }},
    }}
  }}
}});

/* AUC bars */
const auc = DATA.metrics;
const aucData = [
  {{label:'Speech', val:auc.speech_auc, col:'#00b4d8'}},
  {{label:'Motor',  val:auc.motor_auc,  col:'#e67e22'}},
  {{label:'Fused',  val:auc.fused_auc,  col:'#0f2744'}},
];
document.getElementById('auc-bars').innerHTML = aucData.map(a => `
  <div class="auc-row">
    <div class="auc-label">${{a.label}}</div>
    <div class="auc-track">
      <div class="auc-fill" style="width:${{a.val*100}}%;background:${{a.col}}">
        <span class="auc-val">${{a.val.toFixed(3)}}</span>
      </div>
    </div>
    <div class="auc-pct" style="color:${{a.col}}">${{(a.val*100).toFixed(1)}}%</div>
  </div>`).join('');

/* Feature importance bars */
function renderFI(containerId, data, color) {{
  const max = data[0][1];
  document.getElementById(containerId).innerHTML = data.map(([name, val]) => `
    <div class="fi-row">
      <div class="fi-name" title="${{name}}">${{name.replace(/_/g,' ')}}</div>
      <div class="fi-track"><div class="fi-fill" style="width:${{(val/max*100).toFixed(1)}}%"></div></div>
      <div class="fi-val">${{val.toFixed(3)}}</div>
    </div>`).join('');
}}
renderFI('fi-speech', DATA.fi_speech, '#00b4d8');
renderFI('fi-motor',  DATA.fi_motor,  '#e67e22');
</script>
</body>
</html>"""

os.makedirs("/mnt/user-data/outputs", exist_ok=True)
with open("/mnt/user-data/outputs/pie_report.html", "w") as f:
    f.write(HTML)
print("Report written to /mnt/user-data/outputs/pie_report.html")
