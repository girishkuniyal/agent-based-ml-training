from jinja2 import Template

def render_serve_template(usecase_name: str, schema_dict: dict) -> str:
    """
    Generates a FastAPI `serve.py` file as a string using Jinja2 templating.

    Args:
        usecase_name (str): Name of the ML use case (used in model path).
        schema_dict (dict): Dictionary where keys are column names and values are Python types.

    Returns:
        str: Rendered FastAPI code for the given use case.
    """
    template = Template('''from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from utils.sqlite_db import increment_hit

app = FastAPI()
model = joblib.load("usecases/{{ usecase_name }}/model.joblib")

class InputSchema(BaseModel):
{% for field, dtype in schema.items() %}
    {{ field }}: {{ dtype }}
{% endfor %}

@app.post("/predict")
def predict(input: InputSchema):
    # Convert Pydantic model to DataFrame
    input_df = pd.DataFrame([input.dict()])

    # Predict using the loaded pipeline (which includes preprocessing + model)
    prediction = model.predict(input_df)[0]
                        
    increment_hit("{{ usecase_name }}")

    return {"prediction": prediction}
                        
@app.get("/health")
def health_check():
    return {"status": "ok", "usecase": "{{ usecase_name }}"}
''')

    # Ensure proper type mapping for pydantic
    type_mapping = {
        'int': 'int',
        'float': 'float',
        'bool': 'bool',
        'str': 'str'
    }

    schema_str_dict = {
        key: type_mapping.get(value.__name__, 'str')
        for key, value in schema_dict.items()
    }

    return template.render(usecase_name=usecase_name, schema=schema_str_dict)
