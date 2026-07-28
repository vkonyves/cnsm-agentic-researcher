#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", type=Path, required=True)
    args = parser.parse_args()

    prereg = args.study_dir / "preregistration.json"
    hash_file = args.study_dir / "preregistration.sha256"

    expected = hash_file.read_text(encoding="utf-8").strip()
    actual = hashlib.sha256(prereg.read_bytes()).hexdigest()

    if actual != expected:
        raise SystemExit("Preregistration integrity check FAILED")

    print("Preregistration integrity check: OK")
    print("SHA-256:", actual)


if __name__ == "__main__":
    main()
