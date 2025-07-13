# Text2Model: Describe your use case, get a deployed model

**Text2Model (Agentic ML Trainer)** lets you build production-grade machine learning models from just plain English. Upload your dataset, describe your intent, and we’ll take care of the rest — from generating model code to training, tuning, and deploying your model as a fully working REST API.

🚀 Say goodbye to boilerplate and infrastructure headaches — Text2Model is the fastest way to go from CSV to prediction endpoint.

> 📽️ **Watch Demo**: [Click here to see it in action](https://www.youtube.com/watch?v=fmmTLo7SlBg&ab_channel=GIRISHkuniyal)

> 🖼️ **Screenshot**:
> ![Demo Screenshot](resource/ui_demo.png)

> 🧪 *Note: This is a basic POC demo project created for learning and showcasing agentic ML pipelines.*

---

## ✨ Features

* 🧠 **Plain English to Model**: Just describe your ML use case — we generate the full training code automatically.
* 🧪 **Auto Model Selection & Tuning**: Automatically chooses the best model and hyperparameters.
* 🚀 **One-Click Deployment**: Exposes your trained model instantly via REST API with OpenAPI docs.
* 📊 **Test Interface**: Built-in UI to test prediction APIs instantly.
* 🔄 **Persistent Use Cases**: Models stay available even after system restarts.

---

## 🛠️ Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/agentic-ml.git
cd agentic-ml
```

### 2. Set up Environment

```bash
pip install poetry
poetry install
```

### 3. Add OpenAI API Key

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_key_here
```

### 4. Run the App

```bash
poetry run streamlit run main.py
```

---

## 🧱 Architecture Overview

```
[User Input via Streamlit UI]
      ↓
[Schema Extraction from CSV]
      ↓
[Code Generation via LLM Agent (prompt + context)]
      ↓
[Model Training and Evaluation (AutoML)]
      ↓
[FastAPI App Generation per Use Case]
      ↓
[Serve via Uvicorn + REST Endpoint]
```

* **Model Registry**: SQLite-based tracking of models, ports, statuses, and hits.
* **Persistent Deployments**: PIDs stored in DB for safe start/stop handling.
* **Health Monitoring**: Every model has a `/health` endpoint.
* **Custom JSON Schema**: Automatically derived from uploaded CSV.

---

## 🚧 Future Scope & Improvements

✅ **Extended Context Window**: Support for large-scale codebases and multi-file context windows for accurate agentic understanding.

✅ **Advanced Model Versioning & Rollback**: Seamless version control with revert capability and model lineage tracking.

✅ **Cloud-Native Deployments**: Plug-and-play deployment across GCP, AWS, and Azure with autoscaling, CI/CD and Containerization.

✅ **Visual Workflow Orchestration**: Drag-and-drop GUI to define, chain, and monitor ML pipelines end-to-end.

✅ **Realtime Notifications**: Integrated Webhook & Slack alerting for training milestones, errors, and deployment events.

✅ **Resilient Failure Recovery**: Granular error tracing with intelligent retries and fallback mechanisms.

✅ **Semantic Caching**: Context-aware caching of past model intents and pipeline structures for faster iterations.

✅ **Advanced Generator-Validator Agents**: Modular agentic architecture with reflective validation, refactoring, and 
self-debugging loops.

✅ **Expanding ML Use Case Coverage**: Scalable templates and code for supervised, unsupervised, time-series, and custom ML pipelines.

✅ **Robust Persistent State Management**: SQLite/PostgreSQL-based unified storage for tracking metadata, endpoints, and pipeline states.

---

## 🤝 Contributing

We welcome contributions! Please open issues or submit PRs for suggestions, improvements, or bug fixes.

---

## 🪪 License

MIT License

---

Made with ❤️ by Girish
