import streamlit as st
import json
import pandas as pd
import litellm

# لود دیتاست
with open("data/gastriccancer_benchmark.jsonl") as f:
    data = [json.loads(line) for line in f]

st.title("GastricGuard-Eval")
st.subheader("LLM Safety Evaluator for Early Gastric Cancer Screening")

category = st.selectbox("Category", sorted(set(d["category"] for d in data)))
items = [d for d in data if d["category"] == category]
selected = st.selectbox("Select scenario", items, format_func=lambda x: f"{x['id']} – {x['user_message'][:80]}...")

model = st.selectbox("Model", ["gpt-4o-mini", "claude-3-5-sonnet-20241022", "openai/o1-mini"])

if st.button("Run evaluation"):
    with st.spinner("Running..."):
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": selected["user_message"]}],
            temperature=0.0
        ).choices[0].message.content
    
    col1, col2 = st.columns(2)
    with col1:
        st.success("Model Response")
        st.write(response)
    with col2:
        st.info("Expected Behavior")
        st.write(selected["expected_behavior"])
        st.caption(f"Source: {selected['source']}")
