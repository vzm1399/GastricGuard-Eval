import json
import pandas as pd
from tqdm import tqdm
import litellm
import os
from dotenv import load_dotenv

load_dotenv()

# تنظیم API keyها (بعداً در .env می‌ذاری)
def query_model(model, messages):
    response = litellm.completion(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=512
    )
    return response.choices[0].message.content.strip()

# لود دیتاست
with open("data/gastriccancer_benchmark.jsonl") as f:
    dataset = [json.loads(line) for line in f]

models_to_test = [
    "gpt-4o-mini",
    "claude-3-5-sonnet-20241022",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
]

results = []

for item in tqdm(dataset[:50], desc="Evaluating"):  # اول 50 تا برای تست سریع
    prompt = item["user_message"]
    expected = item["expected_behavior"]
    source = item["source"]
    
    for model in models_to_test:
        try:
            response = query_model(model, [{"role": "user", "content": prompt}])
            results.append({
                "id": item["id"],
                "category": item["category"],
                "model": model,
                "prompt": prompt,
                "response": response,
                "expected": expected,
                "source": source
            })
        except Exception as e:
            print(f"Error with {model}: {e}")

df = pd.DataFrame(results)
df.to_csv("results/first_50_results.csv", index=False)
df.to_json("results/first_50_results.json", orient="records", indent=2)
print("50 تا اول تموم شد – نتایج در results/ ذخیره شد")
