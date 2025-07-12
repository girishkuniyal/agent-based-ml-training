# utils/agent_validator.py

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env into environment

# Now this will work as expected
openai_key = os.getenv("OPENAI_API_KEY")


VALIDATOR_TEMPLATE = """
You are a senior MLOps engineer responsible for validating Python training scripts for tabular ML use cases.

Your job is to:
- Review the provided code thoroughly
- Determine if the code is **production-ready**
- Validate that `model_type` supports three options: `auto`, `classification`, and `regression`

A production-grade training script **must**:
- Be syntactically correct and logically sound
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

def validate_code(training_file: str):
    with open(training_file, "r") as f:
        code = f.read()

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         VALIDATOR_TEMPLATE
         ),
        ("human", "Here is the code to review:\n\n{code}")
    ])

    llm = ChatOpenAI(temperature=0.1, model="gpt-4o",
                     openai_api_key=openai_key)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"code": code})

    # Parse result block from LLM response
    valid_flag = "valid: true" in response.lower()
    return valid_flag, response
