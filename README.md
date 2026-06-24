# PIE – Pediatric Insight Engine

A multimodal AI system for early developmental risk detection in children,
analyzing speech/language patterns and motor behavior over time.

## Project Structure

```
pie/
├── data/
│   ├── speech/          # Raw and processed speech/transcript data
│   ├── motor/           # Raw and processed motor/skeletal data
│   └── fusion/          # Combined feature outputs
├── speech/              # Speech & language analysis modules
├── motor/               # Motor behavior analysis modules
├── fusion/              # Multimodal fusion & risk prediction
├── utils/               # Shared utilities
├── app/                 # Streamlit dashboard
├── tests/               # Unit tests
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app/dashboard.py
```

## Components
1. Speech & Language Analysis (CHILDES + librosa)
2. Motor Behavior Analysis (K-DST + MediaPipe)
3. Multimodal Fusion + Risk Prediction
4. Temporal Tracking (Developmental Velocity)
5. Explainable AI (SHAP + Reasoning Tags)
6. Streamlit Dashboard
