# main.py
import streamlit as st
import os
import time
import pandas as pd
import json
import requests
from utils.sqlite_db import (
    init_db, register_usecase, update_status, 
    get_all_usecases, update_pid, get_pid
)
from utils.schema_builder import build_schema_from_csv
from utils.agent_loop import run_agentic_code_generation
from utils.model_trainer import train_model
from utils.subprocess_manager import launch_uvicorn, stop_uvicorn, check_health

st.set_page_config(page_title="⚡ Agentic ML Model Trainer", layout="wide")
st.title("⚡ Build, Train & Deploy Production ML Models Instantly")
st.caption("Built with ❤️ | Inspired by [Plexe.ai](https://plexe.ai/)")

with st.container():
    st.markdown("""
    Welcome to **Text2Model : Agentic ML Trainer** — where ideas become deployed ML systems in just a few clicks.

    ✅ **Production-Ready Code**  
    🧠 **Trained & Tuned ML Models**  
    🚀 **Auto-Deployed API Endpoints**

    > No infra hassle. No boilerplate. Just results. 
    """)

# Step 1: Initialize database
init_db()

# Step 2: Sidebar form for uploading new use case
with st.sidebar:
    st.header("📌 Create a New ML Use Case")
    st.markdown("Upload your dataset and define the task")
    usecase_name_input = st.text_input("📝 Use Case Name", placeholder="e.g., churn_prediction")
    intent = st.text_area("🧠 Modeling Intent", placeholder="e.g., Predict customer churn based on demographics")
    uploaded_file = st.file_uploader("📁 Upload Dataset (.csv)", type=["csv"])
    target_col = st.text_input("🎯 Target Column", placeholder="e.g., churn")
    model_type = st.radio("⚙️ Task Type", ["auto", "classification", "regression"])
    trigger_btn = st.button("🚀 Train & Deploy Model")

# Step 3: Handle use case
if trigger_btn and uploaded_file and target_col:
    usecase_name = usecase_name_input.lower().replace(" ", "_")
    usecase_path = f"usecases/{usecase_name}"
    model_path = os.path.join(usecase_path, "model.joblib")
    os.makedirs(usecase_path, exist_ok=True)

    csv_path = os.path.join(usecase_path, "training_data.csv")
    with open(csv_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.status(f"🤖 Generating ML code for `{usecase_name_input}`...", expanded=True) as status:
        schema_dict = build_schema_from_csv(csv_path, usecase_name, target_col)
        run_agentic_code_generation(usecase_name, intent, schema_dict, target_col, model_type)
        status.update(label="✅ Code generated", state="complete")

    with st.status(f"🧠 Training model for `{usecase_name_input}`...", expanded=True) as status:
        result = train_model(csv_path, target_col, model_path, model_type, usecase_name)
        status.update(label="✅ Model trained", state="complete")

    st.success(f"🏆 Best Model: `{result['model_name']}` | `{result['metric']}` = Test: `{result['test_score']}` | Train: `{result['train_score']}`")

    port = 8000 + (abs(hash(usecase_name)) % 1000)
    proc = launch_uvicorn(usecase_name, f"usecases.{usecase_name}.serve:app", port)

    if proc:
        register_usecase(usecase_name, usecase_path, port, result['model_name'], proc.pid)
    else:
        st.error("🚫 Failed to deploy API server")

# Step 4: Show deployed use cases
st.markdown("---")
st.header("📡 Your Deployed ML Use Cases")

if "start_clicked" not in st.session_state:
    st.session_state.start_clicked = {}
if "stop_clicked" not in st.session_state:
    st.session_state.stop_clicked = {}

usecases = get_all_usecases()

if not usecases:
    st.info("No use cases launched yet. Use the sidebar to create one.")

for name, port, status_val, api_hits, model_type in usecases:
    # Determine status
    manual_interaction = st.session_state.start_clicked.get(name) or st.session_state.stop_clicked.get(name)
    if not manual_interaction:
        is_live = check_health(port)
        status = "live" if is_live else "offline"
        update_status(name, status)
    else:
        status = status_val
        is_live = status == "live"

    st.session_state.start_clicked[name] = False
    st.session_state.stop_clicked[name] = False

    status_icon = "🟢" if is_live else "🔴"

    with st.container():
        st.markdown(f"### `{name}` {status_icon}")
        cols = st.columns([2, 2, 2, 2, 1])
        cols[0].markdown(f"**Model Deployed:** `{model_type}`")
        cols[1].markdown(f"**Docs:** [localhost:{port}](http://localhost:{port}/docs)")
        cols[2].markdown(f"**API Hits:** `{api_hits}`")

        if is_live:
            if cols[4].button("🛑 Stop", key=f"stop-{name}"):
                st.session_state.stop_clicked[name] = True
                pid = get_pid(name)
                if stop_uvicorn(pid):
                    update_status(name, "stopped")
                    update_pid(name, None)
                    st.warning(f"Stopped: `{name}`")
                else:
                    st.error("❌ Failed to stop process")
        else:
            if cols[4].button("▶️ Start", key=f"start-{name}"):
                st.session_state.start_clicked[name] = True
                proc = launch_uvicorn(name, f"usecases.{name}.serve:app", port)
                if proc:
                    update_status(name, "live")
                    update_pid(name, proc.pid)
                    st.success(f"Started: `{name}`")
                else:
                    st.error(f"Failed to start: `{name}`")

        with st.expander(f"📊 Training Summary - `{name}`"):
            metrics_path = f"usecases/{name}/experiments.json"
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    raw_metrics = json.load(f)

                records = []
                for model_name, details in raw_metrics.items():
                    record = {
                        "Model": model_name,
                        "Train Score": details.get("train_score"),
                        "Test Score": details.get("test_score"),
                        "Best Params": json.dumps(details.get("best_params", {}), indent=0)
                    }
                    records.append(record)

                df = pd.DataFrame(records)
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("No training metrics available yet.")

        with st.expander(f"🧪 Test Prediction API - `{name}`"):
            schema_path = f"usecases/{name}/schema.json"
            if os.path.exists(schema_path):
                with open(schema_path) as f:
                    schema = json.load(f)

                st.markdown(f"Fill in input fields to test your model. ENDPOINT: `http://localhost:{port}/predict`")
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

                if st.button("🔍 Run Prediction", key=f"infer-{name}"):
                    try:
                        resp = requests.post(f"http://localhost:{port}/predict", json=user_input)
                        if resp.status_code == 200:
                            st.success("✅ Prediction Result")
                            st.json(resp.json())
                        else:
                            st.error(f"❌ Error {resp.status_code}: {resp.text}")
                    except Exception as e:
                        st.error(f"🚫 Request failed: {e}")
            else:
                st.warning("⚠️ Schema not found. Please retrain the use case.")
