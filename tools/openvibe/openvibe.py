#!/usr/bin/env python3
"""
OpenVibe - Elevator ride-quality analyzer.

This file is kept for backwards compatibility.

Preferred usage (after installing the package):
    openvibe sample_data.csv --plot

Legacy usage:
    python openvibe.py sample_data.csv --plot
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    # Allow `python openvibe.py ...` without requiring an editable install.
    src_dir = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_dir))

    from openvibe.cli import main as _main

    _main()


if __name__ == "__main__":
    main()

