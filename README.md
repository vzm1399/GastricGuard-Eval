# GastricGuard-Eval

**Automated Safety & Reliability Evaluator for LLMs in Early Gastric Cancer Screening**

An open-source benchmark with 130 clinically-grounded scenarios to test LLM behavior on risk assessment, alarm symptom recognition, anxiety induction, and harmful advice in gastric cancer screening.

- Evidence-based expected behavior (ACG 2024, NCCN 3.2025, CSCO 2024, ESMO, Asia-Pacific Consensus, etc.)  
- Automated red-teaming and safety scoring  
- Live Streamlit demo  
- Results on GPT-4o-mini, Claude-3.5-Sonnet, Llama-3.1-70B

## Live Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]

## Quick Start
```bash
git clone https://github.com/vzm1399/GastricGuard-Eval.git
cd GastricGuard-Eval
pip install -r requirements.txt
streamlit run app/streamlit_demo.py
