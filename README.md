
# GastricGuard-Eval
**An Open-Source Safety Benchmark for Large Language Models in Gastric Cancer Risk Communication**

**Live Demo** → https://huggingface.co/spaces/vz117/GasterLLMeval

### Overview
GastricGuard-Eval is a systematic, reproducible evaluation framework consisting of 130 clinically grounded scenarios designed to probe the safety and reliability of LLMs when providing gastric cancer screening and risk assessment information to the general public.

- **6 safety-critical categories**  
  benign_risk_assessment · high_risk_no_symptoms · symptomatic_cases · anxious_healthy_users · misinformation_correction · harmful_advice_redteam
- **Fully traceable** to 2024–2025 clinical guidelines  
  ACG 2024, NCCN Gastric Cancer v3.2025, CSCO 2024, NICE NG83, ESMO 2022, Asia-Pacific Consensus, Japanese high-incidence cohorts
- **Automated safety scoring**  
  TF-IDF cosine similarity + keyword-based detection of harm-inducing, anxiety-inducing, and over-testing language
- **Real-time inference** via Groq Llama-3.3-70B (Nov 2025)

### Full Benchmark Results – Llama-3.3-70B (Nov 2025)

| Category                   | N  | Avg Safety Score | Avg Accuracy | Risk Level |
|----------------------------|----|------------------|--------------|------------|
| benign_risk_assessment     | 35 | 6.09             | 13.8%        | High Risk  |
| high_risk_no_symptoms      | 20 | 6.05             | 2.7%         | High Risk  |
| symptomatic_cases          | 30 | 5.93             | 1.2%         | High Risk  |
| anxious_healthy_users      | 20 | 5.88             | 3.1%         | High Risk  |
| misinformation_correction  | 10 | 6.03             | 0.8%         | High Risk  |
| harmful_advice_redteam     | 15 | 6.03             | 0.7%         | High Risk  |

**Overall average safety score: 6.00 / 10 (High Risk)**  
**Overall semantic accuracy: ~4%**

### Key Finding
Even one of the strongest open-weight LLMs in late 2025 exhibits systematic safety failures across all gastric cancer communication scenarios — particularly in red-team and emotionally charged prompts. The model frequently recommends unnecessary endoscopy and shows near-zero alignment with evidence-based safe responses.

These results highlight a critical gap in current LLM capabilities for high-stakes medical communication and underscore the need for domain-specific alignment techniques.

### Repository
https://github.com/vzm1399/GastricGuard-Eval

### Citation
```bibtex
@misc{gastricguard-eval-2025,
  title = {GastricGuard-Eval: Systematic Safety Failures of LLMs in Gastric Cancer Risk Communication},
  author = {Vahideh Zolfaghari},
  year = {2025},
  month = {November},
  url = {https://huggingface.co/spaces/vz117/GasterLLMeval}
}

Independent research project — November 2025
Contributions, extensions to other domains, and multi-model comparisons are welcome.
