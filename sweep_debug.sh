#!/usr/bin/env bash
set -e

# --debug-mode single
# --debug-mode multistep
# --debug-mode init_pattern
      # init_pattern debugger    按 stratum 过滤：          --stratum xp_root
      # init_pattern debugger    按具体 pattern 过滤：      --pattern dp_split_by_type
# --debug-mode numeric_pattern
      # numeric_pattern debugger 只看某个 node：            --node-id 3
      # numeric_pattern debugger 只看某个 numeric pattern： --pattern tp_binary_bias


python run_debugger.py \
  --debug-mode numeric_pattern \
  --rewrite-max-steps 4 \
  --individual-json ./debug/examples/init_a01b2739afada99ccf1c0cf86afd853d8dd9a8d6.json \
  --model-index 0 \
  --hcase-index 0 \
  --pcase-index 3 \
  --t-end 100 \
  --req-type-num 3 \
  --req-dist "[0.6,0.3,0.1]" \
  --lam 100 \
  --topk 20
#  --report-json ./debug/rewrite_report.json \
#  --save-dir ./debug/rewrite_candidates