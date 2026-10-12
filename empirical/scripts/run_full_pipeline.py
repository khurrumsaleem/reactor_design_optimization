#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single- and multi-target registered studies.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "buchwald_hartwig.csv",
                        help="Measured reaction CSV prepared by fetch_buchwald_hartwig.py")
    args = parser.parse_args()
    data_path = args.data.expanduser().resolve()
    configs = (ROOT / "configs" / "full_single.json", ROOT / "configs" / "full.json")
    for config in configs:
        command = [sys.executable, "-u", str(ROOT / "scripts" / "run_experiment.py"), "--config", str(config),
                   "--data", str(data_path)]
        if args.resume:
            command.append("--resume")
        print(f"\n>>> Running {config.name}", flush=True)
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
