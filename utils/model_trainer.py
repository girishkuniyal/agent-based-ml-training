# utils/model_trainer.py

import pandas as pd
import joblib
import importlib.util
import os

def load_train_module(usecase_name):
    """
    Dynamically loads the `train_model.py` module from a given usecase.
    """
    module_path = f"usecases/{usecase_name}/train_model.py"
    spec = importlib.util.spec_from_file_location("train_model", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def train_model(csv_path, target_col, model_output_path, model_type="auto", usecase_name=""):
    """
    Load usecase-specific training module and run training pipeline.

    Args:
        csv_path (str): Path to training dataset CSV
        target_col (str): Column to predict
        model_output_path (str): Path to save the trained model
        model_type (str): 'regression', 'classification' or 'auto'
        usecase_name (str): Identifier for the ML use case

    Returns:
        dict: Contains model_type, metric, and test_score
    """
    # Load dataset
    df = pd.read_csv(csv_path)

    # Dynamically import the agent-generated train_model.py
    training_module = load_train_module(usecase_name)

    # Run training pipeline
    model_object, model_name, train_score, test_score, metric = training_module.train_pipeline(df, target_col, model_type)

    # Save model
    joblib.dump(model_object, model_output_path)

    return {
        "model_name": model_name,
        "train_score": train_score,
        "test_score": test_score,
        "metric": metric
    }
