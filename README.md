
# GastricGuard-Eval

**Automated Safety & Reliability Evaluator for LLMs in Early Gastric Cancer Screening**

An open-source benchmark with 130 clinically-grounded, evidence-based scenarios to test LLM behavior in gastric cancer risk communication — specifically designed for AI safety and alignment research.

# Live Demo
Local demo available:  
```bash
streamlit run app/streamlit_demo.py

(Public cloud deployment coming in 24h)

# Key Features

130 English prompts across 6 safety-critical categories
Fully grounded in global clinical guidelines (2024–2025)
Automated red-teaming and ethical behavior testing
Live comparison of frontier models
Ready-to-run evaluation pipeline

# Dataset

130 expert-crafted scenarios
Sources include:
ACG 2024, AGA 2024, NCCN 3.2025, NICE NG83, CSCO 2024, ESMO 2022, Asia-Pacific Consensus, Brazilian Guidelines, Japanese high-incidence studies
File: data/gastriccancer_benchmark.jsonl

# Quick Start
Bashgit clone https://github.com/vzm1399/GastricGuard-Eval.git
cd GastricGuard-Eval
pip install -r requirements.txt
streamlit run app/streamlit_demo.py
Run Full Evaluation
Bashpython run_evaluation.py
→ Results automatically saved in results/ folder

# Author
PhD Candidate in Medical Informatics
Author of CEMET’s guideline on ethical use of generative AI in higher education.
Focused on safe, equitable, and trustworthy Health AI
Work in Progress (24–48h)

Full automated safety scoring (hallucination, harm, anxiety induction)
Comparative results dashboard

## Latest Results (130 scenarios, Nov 2025)
| Model                  | Avg Safety Score | Harm Detected | Anxiety Induced | Unnecessary Tests |
|------------------------|------------------|---------------|-----------------|-------------------|
| gpt-4o-mini            | 9.1              | 3             | 12              | 4                 |
| claude-3-5-sonnet      | 9.6              | 0             | 5               | 1                 |

Full results: [`results/complete_results.csv`](results/complete_results.csv)


Public Streamlit/Hugging Face deployment
arXiv pre-print submission
