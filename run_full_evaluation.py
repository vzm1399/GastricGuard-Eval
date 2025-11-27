import json
import pandas.pd as pd
from tqdm import tqdm
import litellm
import os
from dotenv import load_dotenv
from evaluator.safety_scorer import score_response

load_dotenv()

def query_model(model, messages):
    try:
        response = litellm.completion(model=model, messages=messages, temperature=0.0, max_tokens=512)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] {str(e)}"

with open("data/gastriccancer_benchmark.jsonl") as f:
    dataset = [json.loads(line) for line in f]

models = ["gpt-4o-mini", "claude-3-5-sonnet-20241022"]
results = []

print("Running FULL evaluation on 130 scenarios...")
for item in tqdm(dataset, desc="Evaluating"):
    prompt = item["user_message"]
    expected = item["expected_behavior"]
    
    for model in models:
        response = query_model(model, [{"role": "user", "content": prompt}])
        scores = score_response(response)
        
        results.append({
            "id": item["id"],
            "category": item["category"],
            "model": model,
            "prompt": prompt,
            "response": response,
            "expected": expected,
            "source": item["source"],
            **scores
        })

os.makedirs("results", exist_ok=True)
df = pd.DataFrame(results)
df.to_csv("results/complete_results.csv", index=False)
df.to_json("results/complete_results.json", orient="records", indent=2)

summary = df.groupby("model").agg({
    "safety_score": "mean",
    "harm_detected": "sum",
    "anxiety_induced": "sum"
}).round(2)
summary.to_csv("results/summary.csv")
print("FULL EVALUATION DONE! Check results/ folder")
