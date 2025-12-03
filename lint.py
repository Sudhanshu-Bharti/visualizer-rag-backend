#!/usr/bin/env python3
"""
Simple linting script for the backend.
Can be run directly: python lint.py
"""

import subprocess
import sys


def main():
    """Run all linting checks."""
    print("Backend Linting Script")
    print("=" * 50)

    # Check if ruff is available
    try:
        subprocess.run(["ruff", "--version"], check=True, capture_output=True)
        print("Ruff is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: ruff not found. Please install ruff:")
        print("  pip install ruff")
        return 1

    # Run ruff check
    print("\nRunning ruff check...")
    result = subprocess.run(["ruff", "check", "."])
    if result.returncode != 0:
        print("Linting failed!")
        return 1

    print("All linting checks passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
