import json
import pandas as pd
from tqdm import tqdm
import litellm
import os
from dotenv import load_dotenv

load_dotenv()

def query_model(model, messages):
    try:
        response = litellm.completion(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] {str(e)}"

with open("data/gastriccancer_benchmark.jsonl") as f:
    dataset = [json.loads(line) for line in f]

models_to_test = [
    "gpt-4o-mini",
    "claude-3-5-sonnet-20241022",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
]

results = []

print("Evaluating first 50 scenarios...")
for item in tqdm(dataset[:50], desc="Testing"):
    prompt = item["user_message"]
    for model in models_to_test:
        response = query_model(model, [{"role": "user", "content": prompt}])
        results.append({
            "id": item["id"],
            "category": item["category"],
            "model": model,
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "response": response,
            "expected": item["expected_behavior"],
            "source": item["source"]
        })

os.makedirs("results", exist_ok=True)
df = pd.DataFrame(results)
df.to_csv("results/first_50_results.csv", index=False)
df.to_json("results/first_50_results.json", orient="records", indent=2)
print("Done! Results saved in results/ folder")
