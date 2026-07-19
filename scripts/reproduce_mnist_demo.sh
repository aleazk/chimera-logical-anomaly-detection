#!/usr/bin/env bash
set -euo pipefail
python demos/mnist_forbidden_conjunction.py \
  --pairs "1,7" \
  --negatives chimeras_only \
  --seed 123
