import time
import requests
from config.settings import MEILI_URL, MASTER_HEADERS


def check_task(task_uid: int):
    """Get the current status of a task by its UID."""
    response = requests.get(
        f"{MEILI_URL}/tasks/{task_uid}",
        headers=MASTER_HEADERS
    )
    response.raise_for_status()
    return response.json()


def list_tasks(limit: int = 20):
    """List recent tasks."""
    response = requests.get(
        f"{MEILI_URL}/tasks",
        headers=MASTER_HEADERS,
        params={"limit": limit}
    )
    response.raise_for_status()
    return response.json()


def wait_for_task(task_uid: int, interval: int = 5, timeout: int = 1800):
    """
    Poll a task until it succeeds or fails.
    interval: seconds between polls (default 5)
    timeout:  max seconds to wait (default 30 min)
    Raises TimeoutError if task doesn't complete in time.
    """
    elapsed = 0
    while elapsed < timeout:
        task = check_task(task_uid)
        status = task["status"]
        print(f"[Task {task_uid}] status: {status} ({elapsed}s elapsed)")

        if status == "succeeded":
            print(f"[Task {task_uid}] completed successfully.")
            return task

        if status == "failed":
            error = task.get("error", {}).get("message", "Unknown error")
            raise RuntimeError(f"[Task {task_uid}] failed: {error}")

        time.sleep(interval)
        elapsed += interval

    raise TimeoutError(f"[Task {task_uid}] timed out after {timeout}s")