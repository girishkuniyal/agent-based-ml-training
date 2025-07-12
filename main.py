# main.py
import streamlit as st
import os
import time
import pandas as pd
from utils.sqlite_db import init_db, register_usecase, update_status, get_all_usecases
from utils.schema_builder import build_schema_from_csv
from utils.agent_loop import run_agentic_code_generation
from utils.model_trainer import train_model
from utils.subprocess_manager import launch_uvicorn, stop_uvicorn, check_health

st.set_page_config(page_title="Agentic ML Model Trainer", layout="wide")
st.title("Build, Train & Launch ML models in Minutes")

st.markdown("""
Imagine reducing your ML build cycle from **weeks to minutes** — this demo app tries to do exactly that.

Welcome to the **Agentic ML Trainer**, where:
- A simple intent + CSV is all you need to launch an expert ML model.
- Every click results in real infrastructure, not just toy demos.
- Each use case becomes a production-ready endpoint, auto-deployed and ready to scale.

If you're a startup founder or builder, this could be inspiration for what **execution at the edge of autonomy** looks like.

Let’s build, train and launch — your next ML model, right now.
""")

# Step 1: Initialize database
init_db()

# Step 2: Sidebar form for uploading new use case
with st.sidebar:
    st.header("Create a New ML Use Case")
    usecase_name_input = st.text_input("Use Case Name (e.g., churn_prediction, price_forecasting)")
    intent = st.text_area("Describe what kind of model you want to build and data description")
    uploaded_file = st.file_uploader("Upload your dataset (.csv)", type=["csv"])
    target_col = st.text_input("Target column (e.g., churn, price, fraud)")
    model_type = st.selectbox("Model Type", ["auto", "regression", "classification"])
    trigger_btn = st.button("🚀 Train & Deploy")

# Step 3: Handle new use case creation
if trigger_btn and uploaded_file and target_col:
    usecase_name = usecase_name_input.lower().replace(" ", "_")
    usecase_path = f"usecases/{usecase_name}"
    model_path = os.path.join(usecase_path, "model.joblib")

    os.makedirs(usecase_path, exist_ok=True)
    csv_path = os.path.join(usecase_path, uploaded_file.name)

    # Save uploaded CSV
    with open(csv_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Training model and deploying an endpoint just for you..."):
        schema_dict = build_schema_from_csv(csv_path, usecase_name, target_col)
        run_agentic_code_generation(usecase_name, intent, schema_dict, target_col, model_type)
        result = train_model(csv_path, target_col, model_path, model_type, usecase_name)

    st.success(f"Best Trained Model `{result['model_name']}` — {result['metric']} Score: `{result['test_score']:.4f}/{result['train_score']:.4f}`")

    port = 8000 + (abs(hash(usecase_name)) % 1000)
    launched = launch_uvicorn(usecase_name, f"usecases.{usecase_name}.serve:app", port)

    if launched:
        register_usecase(usecase_name, usecase_path, port, result['model_type'])
        st.success(f"🟢 Deployed at: [http://localhost:{port}/docs](http://localhost:{port}/docs)")
    else:
        st.error("🚫 Failed to launch the API server")

# Step 4: Show active and inactive use cases
st.markdown("---")
st.header("📊 Active Use Cases — Your Live ML Endpoints")

usecases = get_all_usecases()

if not usecases:
    st.info("No use cases deployed yet. Launch one from the sidebar.")

for name, port, status, api_hits, model_type in usecases:
    is_live = check_health(port)
    cols = st.columns([2, 2, 2, 2, 1])

    cols[0].markdown(f"**Use Case:** `{name}` {'🟢' if is_live else '🔴'}`")
    cols[1].markdown(f"**Model Type:** `{model_type}`")
    cols[2].markdown(f"**API Docs:** [localhost:{port}](http://localhost:{port}/docs)")
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
            st.warning("No training metrics logged yet.")
