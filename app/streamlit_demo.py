import streamlit as st
import json
import litellm

with open("data/gastriccancer_benchmark.jsonl") as f:
    data = [json.loads(line) for line in f]

st.title("GastricGuard-Eval")
st.caption("Live safety evaluation using Groq Llama-3.1-70B")

category = st.selectbox("Category", sorted({d["category"] for d in data}))
items = [d for d in data if d["category"] == category]
selected = st.selectbox("Scenario", items, format_func=lambda x: f"{x['id']} – {x['user_message'][:100]}...")

if st.button("Run with Groq Llama-3.1-70B"):
    with st.spinner("Thinking..."):
        response = litellm.completion(
            model="groq/llama-3.1-70b-instruct",
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
