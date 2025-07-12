import json
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs/llm_calls")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def log_prompt_response(usecase_name: str, prompt: str, response: str, success: bool, retries: int):
    log = {
        "timestamp": datetime.utcnow().isoformat(),
        "prompt": prompt,
        "response": response,
        "success": success,
        "retries": retries
    }
    log_path = LOG_DIR / f"{usecase_name}.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(log) + "\n")
