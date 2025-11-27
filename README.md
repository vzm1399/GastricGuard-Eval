# GastricGuard-Eval
Automated safety & reliability evaluator for LLMs in early gastric cancer screening and risk communication.

## Features
- 130 clinically-inspired scenarios (benign → red-team)
- Fully evidence-based expected behavior (ACG 2024, NCCN 3.2025, CSCO 2024, etc.)
- Automated scoring for accuracy, harm, anxiety induction
- Live Streamlit demo
- Results on GPT-4o-mini, Claude-3.5-Sonnet, Llama-3-8B, etc.

## Quick start
```bash
pip install -r requirements.txt
streamlit run app/streamlit_demo.py
