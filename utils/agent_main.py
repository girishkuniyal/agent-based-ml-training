# utils/agent_main.py

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env into environment

# Now this will work as expected
openai_key = os.getenv("OPENAI_API_KEY")

TEMPLATE = """
You're a senior ML engineer.
Based on the use case intent and schema, generate Python code that trains a model on the CSV and saves the model to disk.
Ensure the code is production-ready and robust, supporting both classification and regression.

### Metadata
- Input CSV path: {csv_path}
- Target column: {target_col}
- Model type: {model_type}
- Usecase folder: usecases/{usecase_name}
- Schema: {schema_dict}

### Requirements
1. Automatically handle preprocessing:
   - Handle missing values
   - Encode categorical variables (if needed)
   - Scale numerical variables (if needed)
2. Use only **scikit-learn**.
3. Save the best model to `model.joblib`.
4. Also save any feature transformers (e.g., scalers, encoders, imputers) used during preprocessing to disk in the same folder.
5. During training:
   - Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
   - Evaluate and compare multiple models (e.g., Regression, SVC,  RandomForest, Gradientboosting if allowed).
   - Save each model's performance in a JSON file `experiments.json` inside the usecase folder with:
     - Model name
     - Hyperparameters
     - Train/Test scores
     - Evaluation metric
6. Define a function `train_pipeline(df, target_col, model_type)` that returns:
    - best model object
    - model name like "RandomForestRegressor" or "SVC" etc
    - training score
    - test score
    - evaluation metric name
7. Create a `predict_preprocessing` function that:
   - Accepts raw input as a **Pydantic object** or JSON
   - Loads the same saved transformers
   - Returns a **processed Pydantic object** or dict, ready for model prediction

### Best Practices
- Write clean, modular code

### Error to fix (if any):
{error_log}

Only return valid and executable Python code for `train_model.py`.
"""

prompt = PromptTemplate(
    input_variables=["csv_path", "target_col", "model_type", "usecase_name", "schema_dict", "error_log"],
    template=TEMPLATE
)

llm = ChatOpenAI(temperature=0, model_name="gpt-4",
                 openai_api_key=openai_key)

def generate_training_code(csv_path, target_col, model_type, usecase_name, 
                           schema_dict, error_log):
    prompt_text = prompt.format(
        csv_path=csv_path,
        target_col=target_col,
        model_type=model_type,
        usecase_name=usecase_name,
        schema_dict=schema_dict,
        error_log=error_log  
    )

    result = llm.predict(prompt_text)

    # Save generated code to train_model.py
    usecase_dir = f"usecases/{usecase_name}"
    os.makedirs(usecase_dir, exist_ok=True)
    with open(os.path.join(usecase_dir, "train_model.py"), "w") as f:
        f.write(result)

    return result
