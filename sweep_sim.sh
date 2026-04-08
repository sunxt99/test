#!/usr/bin/env bash
set -e

model_index=2
req_type_num=3
req_dist_configs=(
#  "1.0 0.0 0.0"
#  "0.0 1.0 0.0"
#  "0.0 0.0 1.0"
  "0.8 0.1 0.1"
  "0.6 0.3 0.1"
  "0.3 0.5 0.2"
#  "0.2 0.3 0.5"
)

# "hcase_index  pcase_index  arch_name"
hw_configs=(
#  "2 5 2npu2pim"
#  "5 10 2npu"
#  "6 10 2pim"
#  "0 15 WTF"
#  "0 3 4npu4pimP3"       # NeuPIM (4+4) + Baseline (P3)
#  "0 4 4npu4pimP4"       # NeuPIM (4+4) + P4
#  "0 12 4npu4pimP12"     # NeuPIM (4+4) + P12
#  "8 16 8npu8pimBaseline" # NeuPIM (8+8) + Baseline (P16)
#  "8 17 8npu8pimPP+TP"     # NeuPIM (8+8) + PP2+TP4+XP (P17)
  "8 18 8npu8pimDP"     # NeuPIM (8+8) + DP (P18)
)

for hw in "${hw_configs[@]}"; do
  read -r hcase_index pcase_index arch_name <<< "$hw"
  for req_dist in "${req_dist_configs[@]}"; do
    read -r req0 req1 req2 <<< "$req_dist"
    for lam in 20; do
      OUT="result/model${model_index}_single_req${req_type_num}_${arch_name}_lam${lam}_${req0}_${req1}_${req2}.jsonl"
      rm -f "$OUT"
#      for B in 448; do # model0
      for B in 1 2 4 8 16 32 64 96 128 256 320 384 448 512; do # model0 (4NPU+4PIM)
      # for B in 1 2 4 8 16 32 48 64 96 128 160 196 228 256 288 384; do # model1
      # for B in 1 2 4 8 16 32 48 64 96 128 160 196 228 256; do  # model2
      # for B in 1 2 4 8 16 32 48 64 96 112 128; do  # model3
        echo "Running req_dist=[$req0,$req1,$req2], lam=$lam, batch=$B"
        /Users/SXT/anaconda3/envs/mytrans/bin/python run_simulation.py \
        --model-index "$model_index" \
        --hcase-index "$hcase_index"\
        --pcase-index "$pcase_index"\
        --req-type-num "$req_type_num" \
        --req-dist "[$req0,$req1,$req2]"\
        --lam $lam \
        --t-end 100 \
        --priority-ratio 0.0 \
        --max-batch-lo "$B" \
        --out "$OUT"
      done
    done
  done
done
echo "Done. Wrote $OUT"