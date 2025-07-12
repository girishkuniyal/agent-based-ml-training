# Agent-Based ML Training Demo

This is a mini-project inspired by [Plexe](https://plexe.ai) — a Y Combinator startup building open-source AI agents to train and deploy ML models from a natural language prompt.

The goal of this demo is to simulate how an AI agent can take in an `intent` and a dataset, autonomously build a machine learning model, and serve predictions over a REST API — all in one smooth pipeline.

---

## What It Does

1. **Understand Intent**
   The agent interprets the task described in plain English.

2. **Load & Preprocess Data**
   Cleans the CSV, handles missing values, encodes categoricals, and prepares train/test splits.

3. **Train a Predictive Model**
   Based on the problem type inferred from the intent, the agent selects and trains a suitable ML algorithm.

4. **Serve the Model via API**
   A FastAPI service exposes an `/infer` endpoint where users can get real-time predictions.

---

## Example Usage

### Step 1: Build the Model

```python
from agent import build

build(
    intent="Predict if a customer will buy again",
    data_path="user_transactions.csv"
)
```

This triggers:

* Dataset loading and cleaning
* Classification model training (XGBoost or Logistic Regression)
* Model saved as `model.joblib`

---

### 🛁 Step 2: Run the Inference API

```bash
uvicorn serve:app --reload
```

#### POST `/infer`

```json
{
  "user_id": "U123",
  "total_spent": 4213,
  "days_since_last_purchase": 45
}
```

#### Response

```json
{
  "prediction": 1,
  "confidence": 0.87
}
```

---

## Project Structure

TODO

---

## Tech Stack

| Component      | Technology                                  |
| -------------- | ------------------------------------------- |
| Language       | Python 3.9+                                 |
| Model Training | Scikit-learn / XGBoost                      |
| API Server     | FastAPI                                     |
| Model Serving  | joblib                                      |
| Agent Logic    | (Simple rule-based) or optionally LangChain |
| Dataset        | Synthetic tabular data                      |

---

## Why This Demo?

Plexe’s vision of reducing ML development from *months to hours* through agentic workflows is really interesting, challenging and game-changing. This demo shows about:

* Multi-agent orchestration for ML pipelines
* Seamless data-to-deployment flows
* Developer-first tools with clean, composable interfaces

I’d love to contribute to challenging ai problem — both as an engineer and product thinker. This repo is a starting point.

---

## Contact

**Girish Kuniyal**
[LinkedIn](https://linkedin.com/in/girishchandra1)

---

> Built with ❤️ from India. Open to feedback, improvements, and collaboration!
