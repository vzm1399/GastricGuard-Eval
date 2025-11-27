import json
import pandas as pd
from tqdm import tqdm
import litellm
import os
from dotenv import load_dotenv
from evaluator.safety_scorer import score_response

load_dotenv()

def query_model(messages):
    response = litellm.completion(
        model="groq/llama-3.1-70b-instruct",
        messages=messages,
        temperature=0.0,
        max_tokens=512
    )
    return response.choices[0].message.content.strip()

with open("data/gastriccancer_benchmark.jsonl") as f:
    dataset = [json.loads(line) for line in f]

results = []
print("Running FULL evaluation with Groq Llama-3.1-70B (130 scenarios)...")
for item in tqdm(dataset, desc="Evaluating"):
    try:
        response = query_model([{"role": "user", "content": item["user_message"]}])
        scores = score_response(response)
        results.append({
            "id": item["id"],
            "category": item["category"],
            "model": "groq/llama-3.1-70b",
            "prompt": item["user_message"],
            "response": response,
            "expected": item["expected_behavior"],
            "source": item["source"],
            **scores
        })
    except Exception as e:
        print(f"Error on {item['id']}: {e}")

os.makedirs("results", exist_ok=True)
df = pd.DataFrame(results)
df.to_csv("results/groq_llama3_70b_results.csv", index=False)
df.to_json("results/groq_llama3_70b_results.json", orient="records", indent=2)

summary = df["safety_score"].mean().round(2)
print(f"DONE! Average safety score: {summary}/10")
print("Results saved in results/ folder")
