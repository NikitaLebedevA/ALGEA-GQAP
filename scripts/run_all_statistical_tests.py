#!/usr/bin/env python3
"""Run Friedman/Nemenyi and Wilcoxon statistical tests on benchmark results."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    scripts = [
        SCRIPT_DIR / "statistical_tests.py",
        SCRIPT_DIR / "wilcoxon_tests.py",
    ]
    for script in scripts:
        print(f"\n{'=' * 60}\nRunning {script.name} ...\n{'=' * 60}")
        result = subprocess.run([sys.executable, str(script)], check=False)
        if result.returncode != 0:
            sys.exit(result.returncode)
    print("\nAll statistical tests finished.")


if __name__ == "__main__":
    main()
