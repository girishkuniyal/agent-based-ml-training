# utils/agent_loop.py

import os
from utils.generate_serving import render_serve_template
from utils.agent_validator import validate_code
from utils.agent_main import generate_training_code


def save_code(filepath: str, code: str):
    with open(filepath, "w") as f:
        f.write(code)


def log_validation_errors(log_path: str, attempt: int, error_log: str):
    with open(log_path, "a") as log_file:
        log_file.write(f"\n--- Attempt {attempt} Validation Errors ---\n{error_log}\n")


def run_agentic_code_generation(
    usecase_name: str,
    intent: str,
    schema_dict: dict,
    target_col: str,
    model_type: str = "auto",
    max_retries: int = 20
):
    """
    Main loop: Generates ML training pipeline via agent, validates it,
    retries with fixes until validation passes, and finally generates a FastAPI serve file.
    """
    usecase_path = f"usecases/{usecase_name}"
    os.makedirs(usecase_path, exist_ok=True)
    error_log = None
    train_model_code = None

    training_file = os.path.join(usecase_path, "train_model.py")
    validation_log = os.path.join(usecase_path, "validation_errors.log")


    for attempt in range(1, max_retries + 1):
        # Step 1: Generate training code
        training_code = generate_training_code(
            intent=intent,
            schema_dict=schema_dict,
            target_col=target_col,
            model_type=model_type,
            usecase_name=usecase_name,
            error_log=error_log,
            train_model_code = train_model_code
        )


        print(f"[Agent Loop] Attempt {attempt}/{max_retries} → {usecase_name}")

        # Step 2: Validate the generated code
        is_valid, error_log = validate_code(training_file)

        if is_valid:
            print("Validation Passed — Training Code Ready.")
            break

        print("Validation Failed — Attempting to Fix...")


        train_model_code = training_code
        log_validation_errors(validation_log, attempt, error_log)

    else:
        print(f"❌ Failed to generate valid training code after {max_retries} retries for: {usecase_name}")

    # Step 4: Render serve.py file using template
    serve_code = render_serve_template(usecase_name, schema_dict)
    serve_path = os.path.join(usecase_path, "serve.py")
    save_code(serve_path, serve_code)

    print(f"🚀 Completed: `{usecase_name}` training + serve code ready.")
