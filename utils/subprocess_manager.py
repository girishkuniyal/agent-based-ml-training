# utils/subprocess_manager.py
import subprocess
import signal
import time
import requests
from typing import Dict

PROCESS_REGISTRY: Dict[str, subprocess.Popen] = {}


def launch_uvicorn(usecase_name: str, module_path: str, port: int) -> bool:
    try:
        proc = subprocess.Popen([
            "uvicorn", module_path, "--host", "0.0.0.0", "--port", str(port)
        ])
        PROCESS_REGISTRY[usecase_name] = proc
        return True
    except Exception as e:
        print(f"Failed to launch {usecase_name}: {e}")
        return False


def stop_uvicorn(usecase_name: str) -> bool:
    proc = PROCESS_REGISTRY.get(usecase_name)
    if not proc:
        print(f"No process found for {usecase_name}")
        return False
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=5)
        del PROCESS_REGISTRY[usecase_name]
        return True
    except Exception as e:
        print(f"Failed to stop {usecase_name}: {e}")
        return False


def check_health(port: int) -> bool:
    try:
        res = requests.get(f"http://localhost:{port}/health", timeout=2)
        return res.status_code == 200
    except:
        return False
