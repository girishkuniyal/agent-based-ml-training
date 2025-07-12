# utils/subprocess_manager.py
import subprocess
import signal
import time
import requests
from typing import Dict
import os

PROCESS_REGISTRY: Dict[str, subprocess.Popen] = {}


def launch_uvicorn(usecase_name: str, module_path: str, port: int) -> subprocess.Popen | None:
    try:
        proc = subprocess.Popen([
            "uvicorn", module_path, "--host", "0.0.0.0", "--port", str(port)
        ])
        PROCESS_REGISTRY[usecase_name] = proc
        return proc
    except Exception as e:
        print(f"Failed to launch {usecase_name}: {e}")
        return None


def stop_uvicorn(pid: int) -> bool:
    try:
        # SIGINT allows for graceful shutdown of Uvicorn
        os.kill(pid, signal.SIGINT)
        time.sleep(2)
        return True
    except ProcessLookupError:
        print(f"No process found with PID: {pid}")
        return False
    except Exception as e:
        print(f"Failed to stop process {pid}: {e}")
        return False



def check_health(port: int) -> bool:
    try:
        res = requests.get(f"http://localhost:{port}/health", timeout=2)
        return res.status_code == 200
    except:
        return False
