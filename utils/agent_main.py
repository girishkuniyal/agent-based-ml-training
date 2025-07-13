# utils/agent_main.py

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
import time
from dotenv import load_dotenv
from utils.sqlite_db import log_llm_call


load_dotenv()  

# Now this will work as expected
openai_key = os.getenv("OPENAI_API_KEY")

def clean_generated_code(code: str) -> str:
    return (
        code.replace("```python", "")
            .replace("```", "")
            .strip()
    )




TEMPLATE = """
You are a senior ML engineer.

Generate robust, production-ready Python code to train a machine learning model using the CSV located at:
  ➤ usecases/{usecase_name}/training_data.csv

The model should solve the following ML task:
  ➤ Task: {model_type} (classification, regression, or auto)
  ➤ Target column: {target_col}
  ➤ Schema: {schema_dict}
  ➤ Use case intent: {intent}

----------------------------------------
Output Expectations:
1. Define a function `train_pipeline(df, target_col, model_type)` that:
   - Trains and evaluates multiple ML models using scikit-learn
   - Returns: best model object, model name, train score, test score, and evaluation metric name

2. Preprocessing:
   - Handle missing values
   - Remove identity columns and column with no name or only spaces
   - Encode categorical features
   - Scale numerical features
   - Build a `Pipeline` with `ColumnTransformer` where needed

3. Training:
   - Use GridSearchCV or RandomizedSearchCV for hyperparameter tuning
   - Compare multiple models (e.g., RandomForest, SVC, LogisticRegression, GradientBoosting)

4. Saving:
   - Save the **entire fitted pipeline** (preprocessor + model) to:
     ➤ `usecases/{usecase_name}/model.joblib`
   - Save a summary of all model performances to:
     ➤ `usecases/{usecase_name}/experiments.json`

----------------------------------------
Best Practices:
- Use only scikit-learn, pandas, joblib
- Python 3.8+ and scikit-learn ≥ 1.0 compatible
- No separate saving of the preprocessor — the entire pipeline must be saved
- The pipeline should support direct `.predict()` on raw DataFrame input
- Write clean, modular, and safe code with useful comments

----------------------------------------
If there is an error to correct from previous code, fix it:
{error_log}

Use this if partial code already exists (safely patch/extend it):
{train_model_code}

Output **only valid executable Python code** — no markdown, text, or formatting symbols.
"""


prompt = PromptTemplate(
    input_variables=["target_col", "model_type", "usecase_name", "schema_dict", "error_log", "train_model_code"],
    template=TEMPLATE
)

llm = ChatOpenAI(temperature=0.1, model_name="gpt-4o",
                 openai_api_key=openai_key)

def generate_training_code(intent, target_col, model_type, usecase_name, 
                           schema_dict, error_log, train_model_code):
    prompt_text = prompt.format(
        intent=intent,
        target_col=target_col,
        model_type=model_type,
        usecase_name=usecase_name,
        schema_dict=schema_dict,
        error_log=error_log ,
        train_model_code = train_model_code
    )


    result = llm.predict(prompt_text)

    log_llm_call(usecase_name, prompt_text, result, success=True, retries=0)


    

    clean_result = clean_generated_code(result)

    # Save generated code to train_model.py
    usecase_dir = f"usecases/{usecase_name}"
    os.makedirs(usecase_dir, exist_ok=True)
    with open(os.path.join(usecase_dir, "train_model.py"), "w") as f:
        f.write(clean_result)

    return clean_result
