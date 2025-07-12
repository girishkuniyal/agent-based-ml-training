# utils/schema_builder.py

import pandas as pd
import json
from pydantic import BaseModel, create_model
from typing import Dict, Any
import os


def infer_dtype(series):
    """Infer simplified data type from pandas series."""
    if pd.api.types.is_integer_dtype(series):
        return int
    elif pd.api.types.is_float_dtype(series):
        return float
    elif pd.api.types.is_bool_dtype(series):
        return bool
    else:
        return str


def build_schema_from_csv(csv_path: str, usecase_name: str, target_col: str) -> Dict[str, Any]:
    df = pd.read_csv(csv_path).head(1000)
    df = df.dropna(axis=1, how="all")
    
    schema_dict = {col: infer_dtype(df[col]) for col in df.columns if col != target_col}
    target_dtype = infer_dtype(df[target_col])

    print(f"Schema for use case '{usecase_name}': Target type = {target_dtype.__name__}")

    # Save schema.json with stringified types
    schema_json = {
        "input_format": {k: v.__name__ for k, v in schema_dict.items()},
        "output_format": {"prediction": target_dtype.__name__}
    }

    os.makedirs(f"usecases/{usecase_name}", exist_ok=True)
    with open(f"usecases/{usecase_name}/schema.json", "w") as f:
        json.dump(schema_json, f, indent=2)

    return schema_dict


def generate_pydantic_model(schema_dict: Dict[str, Any], model_name="InferenceInput") -> BaseModel:
    return create_model(model_name, **{k: (v, ...) for k, v in schema_dict.items()})
