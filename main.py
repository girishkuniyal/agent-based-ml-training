# main.py
import streamlit as st
import os
import time
import pandas as pd
import json
import requests
from utils.sqlite_db import init_db, register_usecase, update_status, get_all_usecases
from utils.schema_builder import build_schema_from_csv
from utils.agent_loop import run_agentic_code_generation
from utils.model_trainer import train_model
from utils.subprocess_manager import launch_uvicorn, stop_uvicorn, check_health

st.set_page_config(page_title="⚡ Agentic ML Model Trainer", layout="wide")
st.title("Build, Train & Deploy Production ML Models - Instantly ⚡ ")
st.subheader("Inspired by the [Plexe.ai](https://plexe.ai/)")

st.markdown("""
Welcome to **Agentic ML Trainer** — where ideas become deployed ML systems in just a few clicks.

🚀 **Why spend weeks writing boilerplate ML code when you can go live in minutes?**

With just your **intent + a CSV**, our agent crafts:
- End-to-end ML code (robust, clean, and production-grade)
- Fully trained & tuned models
- Auto-deployed API endpoints — with zero infra hassle

This isn't a toy. It’s a **founder's dream**, an **engineer’s booster**, and a glimpse into what **autonomous execution** looks like.

Ready to replace months of engineering work with one click? Let’s go. 💥
""")

# Step 1: Initialize database
init_db()

# Step 2: Sidebar form for uploading new use case
with st.sidebar:
    st.header("🎯 Create a New ML Use Case")
    usecase_name_input = st.text_input("Use Case Name (e.g., churn_prediction)")
    intent = st.text_area("🔍 What do you want the model to do?")
    uploaded_file = st.file_uploader("📁 Upload your dataset (.csv)", type=["csv"])
    target_col = st.text_input("🎯 Target column (e.g., churn, price, fraud)")
    model_type = st.selectbox("📊 Model Type", ["auto", "regression", "classification"])
    trigger_btn = st.button("🚀 Train & Deploy")

# Step 3: Handle use case
if trigger_btn and uploaded_file and target_col:
    usecase_name = usecase_name_input.lower().replace(" ", "_")
    usecase_path = f"usecases/{usecase_name}"
    model_path = os.path.join(usecase_path, "model.joblib")
    os.makedirs(usecase_path, exist_ok=True)

    csv_path = os.path.join(usecase_path, uploaded_file.name)
    with open(csv_path, "wb") as f:
        f.write(uploaded_file.read())
    
    st.markdown("---")
    spinner_text1 = f"🤖 [{usecase_name_input}]  🍺 Relax ! AI Engineer agents are working on generating production grade ML code for you..."
    with st.spinner(spinner_text1):
        schema_dict = build_schema_from_csv(csv_path, usecase_name, target_col)
        run_agentic_code_generation(usecase_name, intent, schema_dict, target_col, model_type)
    st.success(f"✅ [{usecase_name_input}] Code generated successfully! Ready to train your model.")
    
    spinner_text2 = f"🧠 [{usecase_name_input}] 🟡 Training & tuning your model..."
    with st.spinner(spinner_text2):
        result = train_model(csv_path, target_col, model_path, model_type, usecase_name)
    st.success(f"✅ [{usecase_name_input}] Model Trained Succesfully! Ready to deploy.")

    st.success(f"🏆 [{usecase_name_input}] Best Model: `{result['model_name']}` — `{result['metric']}` = `{result['test_score']:.4f}` (Test) / `{result['train_score']:.4f}` (Train)")

    port = 8000 + (abs(hash(usecase_name)) % 1000)
    launched = launch_uvicorn(usecase_name, f"usecases.{usecase_name}.serve:app", port)

    if launched:
        register_usecase(usecase_name, usecase_path, port, result['model_type'])
        st.success(f"✅ **Deployed Live API**: [http://localhost:{port}/docs](http://localhost:{port}/docs)")
    else:
        st.error("🚫 Failed to deploy the API server")

# Step 4: Show all use cases
st.markdown("---")
st.header("📡 Your Deployed ML Use Cases")

usecases = get_all_usecases()

if not usecases:
    st.info("No use cases launched yet. Let’s change that from the sidebar.")

for name, port, status, api_hits, model_type in usecases:
    is_live = check_health(port)
    cols = st.columns([2, 2, 2, 2, 1])

    status_dot = "🟢" if is_live else "🔴"
    cols[0].markdown(f"**Use Case:** `{name}` {status_dot}")
    cols[1].markdown(f"**Model Type:** `{model_type}`")
    cols[2].markdown(f"**Docs:** [localhost:{port}](http://localhost:{port}/docs)")
    cols[3].markdown(f"**API Hits:** `{api_hits}`")

    if is_live:
        if cols[4].button("🛑 Stop", key=f"stop-{name}"):
            stop_uvicorn(name)
            update_status(name, "stopped")
            st.warning(f"Stopped: `{name}`")
    else:
        cols[4].write("🔴 Offline")

    # ⬇ Expandable section for showing model metrics
    with st.expander(f"📈 View all trained models for `{name}`"):
        metrics_path = f"usecases/{name}/model_metrics.csv"
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)
        else:
            st.warning("📉 No training metrics available yet.")

    # ⬇ Expandable section for testing the deployed model
    with st.expander(f"🧪 Test endpoint for `{name}`"):
        schema_path = f"usecases/{name}/schema.json"
        if os.path.exists(schema_path):
            with open(schema_path) as f:
                schema = json.load(f)

            st.markdown("💡 Fill in test data below to get live prediction.")
            input_format = schema.get("input_format", {})
            user_input = {}
            for field, dtype in input_format.items():
                if dtype == "int":
                    user_input[field] = st.number_input(f"{field} (int)", step=1, key=f"{name}-{field}")
                elif dtype == "float":
                    user_input[field] = st.number_input(f"{field} (float)", format="%.4f", key=f"{name}-{field}")
                elif dtype == "bool":
                    user_input[field] = st.checkbox(f"{field} (bool)", key=f"{name}-{field}")
                else:
                    user_input[field] = st.text_input(f"{field} (str)", key=f"{name}-{field}")

            if st.button("🚀 Run Inference", key=f"infer-{name}"):
                try:
                    resp = requests.post(f"http://localhost:{port}/predict", json=user_input)
                    if resp.status_code == 200:
                        st.success("✅ Prediction:")
                        st.json(resp.json())
                    else:
                        st.error(f"❌ Error {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"🚫 Request failed: {e}")
        else:
            st.warning("⚠️ Schema not found. Please retrain the use case.")
