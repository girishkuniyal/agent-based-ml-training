# utils/schema_builder.py
import pandas as pd
import json
from pydantic import BaseModel, create_model
from typing import Dict, Any
import os


def infer_dtype(val):
    if pd.api.types.is_integer_dtype(val):
        return int
    elif pd.api.types.is_float_dtype(val):
        return float
    elif pd.api.types.is_bool_dtype(val):
        return bool
    else:
        return str


def build_schema_from_csv(csv_path: str, usecase_name: str, target_col : str) -> Dict[str, Any]:
    df = pd.read_csv(csv_path).head(1000) # Limit to first 1000 rows for performance
    df = df.dropna(axis=1, how="all")
    
    schema_dict = {col: infer_dtype(df[col]) for col in df.columns if col != target_col}
    target_column_schema_dict = {col: infer_dtype(df[col]) for col in df.columns if col == target_col}

    # Save schema.json
    schema_json = {
        "input_format": {k: v.__name__ for k, v in schema_dict.items()},
        "output_format": {"prediction": target_column_schema_dict[target_col]}  
    }
    
    os.makedirs(f"usecases/{usecase_name}", exist_ok=True)
    with open(f"usecases/{usecase_name}/schema.json", "w") as f:
        json.dump(schema_json, f, indent=2)

    return schema_dict


def generate_pydantic_model(schema_dict: Dict[str, Any], model_name="InferenceInput") -> BaseModel:
    return create_model(model_name, **{k: (v, ...) for k, v in schema_dict.items()})
