#!/usr/bin/env bash
set -e

python run_debugger.py \
  --debug-mode multistep \
  --rewrite-max-steps 4 \
  --individual-json ./debug/examples/init_78ef8a1034d68e4ab8fd460391dedbcf34188c4c.json \
  --model-index 0 \
  --hcase-index 0 \
  --pcase-index 3 \
  --t-end 100 \
  --req-type-num 3 \
  --req-dist "[0.6,0.3,0.1]" \
  --lam 100 \
  --topk 20 \
  --report-json ./debug/rewrite_report.json \
  --save-dir ./debug/rewrite_candidates