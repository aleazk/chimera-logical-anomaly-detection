#!/usr/bin/env bash
set -euo pipefail
python -m py_compile $(find src demos experiments research -name '*.py')
pytest tests/test_semantics.py tests/test_chimera.py
