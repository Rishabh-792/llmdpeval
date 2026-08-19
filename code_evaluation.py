import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


GENERATED_CODE_DIR = Path("generated_code")
EVALUATION_LOGS_DIR = Path("evaluation_logs")
RUN_TIMEOUT_SECONDS = 120


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ts",
        required=True,
        help="Timestamp substring used to find the generated_code folder to evaluate.",
    )
    return parser.parse_args()


def find_generated_dir(timestamp: str) -> Path:
    matches = [
        path
        for path in GENERATED_CODE_DIR.iterdir()
        if path.is_dir() and timestamp in path.name
    ]

    if not matches:
        raise FileNotFoundError(
            f"No generated_code folder contains timestamp: {timestamp}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Timestamp matched multiple generated_code folders: "
            f"{', '.join(str(path) for path in matches)}"
        )

    return matches[0]


def parse_generated_dir(path: Path) -> dict | None:
    parts = path.name.split("_")
    if len(parts) != 4:
        return None

    model, rag, framework, timestamp = parts
    if not timestamp.isdigit():
        return None

    return {
        "model": model,
        "rag": bool(rag),
        "framework": framework,
        "timestamp": int(timestamp),
        "path": path,
    }


def sort_key(path: Path) -> tuple[int, str]:
    return (run_number(path), path.name)


def run_number(path: Path) -> int:
    stem = path.stem
    suffix = stem.rsplit("_code_", 1)[-1]

    if stem.startswith("code_"):
        suffix = stem.removeprefix("code_")

    return int(suffix) if suffix.isdigit() else sys.maxsize


def run_python_file(code_file: Path) -> dict:
    try:
        result = subprocess.run(
            [sys.executable, str(code_file)],
            cwd=Path.cwd(),
            text=True,
            capture_output=True,
            timeout=RUN_TIMEOUT_SECONDS,
        )
        return {
            "code_file": str(code_file),
            "runnable": str(result.returncode == 0),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired as error:
        return {
            "code_file": str(code_file),
            "runnable": "False",
            "returncode": None,
            "stdout": error.stdout or "",
            "stderr": f"Timed out after {RUN_TIMEOUT_SECONDS} seconds",
        }


def evaluate_generated_dir(generated_dir: dict) -> dict:
    code_files = sorted(generated_dir["path"].glob("*.py"), key=sort_key)
    runs = [run_python_file(code_file) for code_file in code_files]

    return {
        "model": generated_dir["model"],
        "rag": generated_dir["rag"],
        "framework": generated_dir["framework"],
        "timestamp": generated_dir["timestamp"],
        "folder": str(generated_dir["path"]),
        "runs": runs,
    }


def build_run_log(result: dict) -> dict:
    log = {
        "model": result["model"],
        "rag": result["rag"],
        "framework": result["framework"],
        "timestamp": result["timestamp"],
        "folder": result["folder"],
        "runs": [],
    }

    for run in result["runs"]:
        log["runs"].append(
            {
                "code_file": run["code_file"],
                "runnable": run["runnable"],
                "returncode": run["returncode"],
                "stdout": run["stdout"],
                "stderr": run["stderr"],
            }
        )

    return log


def save_run_log(result: dict) -> Path:
    EVALUATION_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = EVALUATION_LOGS_DIR / (
        f"{result['model']}_{result['framework']}_{time.time_ns()}.json"
    )
    log_path.write_text(
        json.dumps(build_run_log(result), indent=2),
        encoding="utf-8",
    )
    return log_path


def main() -> None:
    args = parse_args()
    if not GENERATED_CODE_DIR.exists():
        raise FileNotFoundError(f"Missing generated code directory: {GENERATED_CODE_DIR}")

    generated_dir_path = find_generated_dir(args.ts)
    generated_dir = parse_generated_dir(generated_dir_path)
    if generated_dir is None:
        raise ValueError(
            f"Generated folder name is not in expected format: {generated_dir_path}"
        )

    result = evaluate_generated_dir(generated_dir)
    runnable_count = sum(1 for run in result["runs"] if run["runnable"] == "True")
    failed_count = sum(1 for run in result["runs"] if run["runnable"] != "True")
    total_count = runnable_count + failed_count

    print("\nCode evaluation")
    print("=" * 80)
    print(f"Folder evaluated: {result['folder']}")
    print(f"Python files evaluated: {total_count}")
    print(f"Runnable: {runnable_count}")
    print(f"Failed: {failed_count}")
    print(f"Log file: {save_run_log(result)}")


if __name__ == "__main__":
    main()
