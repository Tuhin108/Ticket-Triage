"""Set up, train, validate, and start the customer-support web app.

Run from any directory with:
    python allinone.py

Use --skip-smart when sentence-transformers or its model download is not
available. The Classic model and web app still run in that mode.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
VENV_DIR = ROOT / ".venv"
VENV_REEXEC_ENV = "ALLINONE_VENV_REEXEC"
OUTPUT_DIRS = (ROOT / "model", ROOT / "screenshots")
RAW_DATASET = ROOT / "data" / "tickets.csv"
SAMPLE_BULK_DATASET = ROOT / "data" / "sample_bulk_tickets.csv"


def run_step(label: str, command: list[str]) -> None:
    print(f"\n=== {label} ===", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def venv_python(venv_dir: Path) -> Path:
    """Return the Python executable inside a virtual environment."""
    executable = "python.exe" if os.name == "nt" else "python"
    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    return venv_dir / scripts_dir / executable


def restart_in_venv(venv_dir: Path) -> None:
    """Create the venv and replace this process with its Python interpreter."""
    python_path = venv_python(venv_dir)
    if not python_path.exists():
        print(f"\n=== Creating virtual environment at {venv_dir} ===", flush=True)
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], cwd=ROOT, check=True)
    else:
        print(f"\n=== Using existing virtual environment at {venv_dir} ===", flush=True)

    if not python_path.exists():
        raise RuntimeError(f"Virtual environment Python was not created at {python_path}")

    print(f"\n=== Starting virtual environment at {venv_dir} ===", flush=True)
    env = os.environ.copy()
    env[VENV_REEXEC_ENV] = "1"
    os.execve(str(python_path), [str(python_path), str(Path(__file__).resolve()), *sys.argv[1:]], env)


def create_sample_bulk_dataset() -> None:
    """Create the app's downloadable bulk-upload example if it is missing."""
    if SAMPLE_BULK_DATASET.exists():
        return

    with RAW_DATASET.open(newline="", encoding="utf-8") as source:
        rows = csv.DictReader(source)
        descriptions = [
            row["ticket_description"]
            for row in rows
            if row.get("ticket_description")
        ][:5]

    with SAMPLE_BULK_DATASET.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(["ticket_description"])
        writer.writerows([[description] for description in descriptions])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Do not install requirements before running the pipeline.",
    )
    parser.add_argument(
        "--skip-smart",
        action="store_true",
        help="Skip sentence-embedding training and run Classic mode only.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser window when the web app starts.",
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    if not os.environ.get(VENV_REEXEC_ENV):
        restart_in_venv(VENV_DIR)

    for output_dir in OUTPUT_DIRS:
        output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_install:
        run_step("Installing dependencies", [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    if not RAW_DATASET.exists():
        (ROOT / "data").mkdir(parents=True, exist_ok=True)
        run_step("Generating the dataset", [sys.executable, str(SRC / "generate_dataset.py")])
    create_sample_bulk_dataset()

    run_step("Preparing the dataset", [sys.executable, str(SRC / "preprocessing.py")])
    run_step("Training and evaluating the Classic model", [sys.executable, str(SRC / "train.py")])

    if not args.skip_smart:
        run_step(
            "Training and evaluating the Smart model",
            [sys.executable, str(SRC / "train_embeddings.py")],
        )

    run_step("Running prediction smoke tests", [sys.executable, str(SRC / "predict.py"), "--test"])

    print("\n=== Starting the web app ===", flush=True)
    print("Open http://127.0.0.1:5000 in your browser. Press Ctrl+C to stop.", flush=True)
    env = os.environ.copy()
    if args.no_browser:
        env["DISABLE_BROWSER"] = "1"
    subprocess.run([sys.executable, str(SRC / "app.py")], cwd=ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
