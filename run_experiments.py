"""
run_experiments.py — Ejecuta p0→p2 y c0→c2 y guarda logs en results/.

Uso:
  python run_experiments.py           # todo
  python run_experiments.py --only p0,c0
  python run_experiments.py --skip-embeddings   # omite p1 y c1 (muy lentos en CPU)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
VENV_PY = ROOT / "venv" / "Scripts" / "python.exe"
PY = str(VENV_PY if VENV_PY.exists() else sys.executable)

FULL_ORDER = ["p0", "p1", "p2", "c0", "c1", "c2"]
EMBEDDING_SCRIPTS = {"p1", "c1"}


def run_one(name: str) -> int:
    script = ROOT / f"{name}.py"
    if not script.exists():
        print(f"[ERROR] No existe {script}")
        return 1

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = RESULTS / f"{name}_{stamp}.log"

    print(f"\n======== Ejecutando {name}.py → {log_path.name} ========")
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# {name}.py started {stamp}\n")
        proc = subprocess.run(
            [PY, str(script)],
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        elapsed = time.perf_counter() - t0
        log.write(f"\n# exit={proc.returncode} elapsed_s={elapsed:.3f}\n")

    print(f"  exit={proc.returncode} elapsed={elapsed:.1f}s → {log_path}")
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Runner del laboratorio CBDE Lab1")
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Lista separada por comas, p.ej. p0,c0",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Omite p1 y c1 (generación de 10k embeddings)",
    )
    args = parser.parse_args()

    if args.only:
        order = [x.strip() for x in args.only.split(",") if x.strip()]
    else:
        order = list(FULL_ORDER)

    if args.skip_embeddings:
        order = [x for x in order if x not in EMBEDDING_SCRIPTS]

    corpus = ROOT / "bookcorpus_10k.txt"
    if not corpus.exists() and any(x in ("p0", "c0") for x in order):
        print("Corpus ausente: ejecutando prepare_corpus.py ...")
        rc = subprocess.run([PY, str(ROOT / "prepare_corpus.py")], cwd=str(ROOT))
        if rc.returncode != 0:
            return rc.returncode

    failed = []
    for name in order:
        rc = run_one(name)
        if rc != 0:
            failed.append(name)
            print(f"[ERROR] {name} falló; se continúa con el siguiente.")

    if failed:
        print(f"\nFallidos: {failed}")
        return 1
    print("\nTodos los experimentos seleccionados terminaron OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
