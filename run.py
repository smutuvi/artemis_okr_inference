#!/usr/bin/env python3
"""Thin entrypoint: python run.py --tasks flowers,pods --sample 20 --visualize"""

from okr_inference.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
