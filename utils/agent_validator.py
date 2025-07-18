# utils/agent_validator.py

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.messages import SystemMessage, HumanMessage, AIMessage
from langchain.schema.output_parser import StrOutputParser
from utils.sqlite_db import log_llm_call
import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env into environment

# Now this will work as expected
openai_key = os.getenv("OPENAI_API_KEY")



def get_role(message):
    if isinstance(message, SystemMessage):
        return "SYSTEM"
    elif isinstance(message, HumanMessage):
        return "HUMAN"
    elif isinstance(message, AIMessage):
        return "AI"
    else:
        return "UNKNOWN"

VALIDATOR_TEMPLATE = """
You are a senior MLOps engineer responsible for validating Python training scripts for tabular ML use cases.

Your job is to:
- Review the provided code thoroughly
- Determine if the code is **production-ready**
- Validate that `model_type` supports three options: `auto`, `classification`, and `regression`

A production-grade training script **must**:
- Be syntactically correct and logically sound
- Validate the data tranformations and data handling steps.
- Handle missing/null values properly
- Encode categorical variables appropriately
- Perform train-test split using a fixed `random_state`
- Dynamically select model type based on `model_type` argument
- Use a suitable evaluation metric (accuracy for classification, RMSE/R² for regression, etc.)
- Leverage a full `sklearn.Pipeline` including preprocessing and model
- Save the trained pipeline to disk using `joblib`
- Be modular, readable, and free from unnecessary global variables or hardcoded values
- Avoid anti-patterns like:
  - Saving preprocessor separately
  - Fitting model outside of pipeline
  - Mixing training and inference logic

Additional checks:
- No comments, markdown syntax, or extra text should be included in the final script
- Code must be ready for direct execution without modification

Your output **must** strictly follow this format:

[RESULT]
Valid: True or False

[REASON]
- Point-by-point explanation of why the code is valid or invalid
- Be objective and specific (e.g., "Missing pipeline usage", "No handling for categorical features", "Commented lines detected")

Only return the `[RESULT]` and `[REASON]` sections. Do not add any other text.
"""

def validate_code(training_file: str, usecase_name: str):
    with open(training_file, "r") as f:
        code = f.read()

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         VALIDATOR_TEMPLATE
         ),
        ("human", "Here is the code to review:\n\n{code}")
    ])

    input_dict = {"code": code}
    rendered_prompt = prompt.format_messages(**input_dict)
    prompt_str = "\n\n".join([f"{get_role(m)}: {m.content}" for m in rendered_prompt])


    llm = ChatOpenAI(temperature=0.1, model="gpt-4o",
                     openai_api_key=openai_key)
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({"code": code})

    log_llm_call(usecase_name, prompt_str, response, success=True, retries=0)

    # Parse result block from LLM response
    valid_flag = "valid: true" in response.lower()
    return valid_flag, response
