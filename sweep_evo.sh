#!/usr/bin/env bash
set -e

model_index=2
req_type_num=3
req_dist_configs=(
#  "1.0 0.0 0.0"
#  "0.0 1.0 0.0"
#  "0.0 0.0 1.0"
#  "0.8 0.1 0.1"
  "0.6 0.3 0.1"
#  "0.3 0.5 0.2"
#  "0.2 0.3 0.5"
)

# "hcase_index  pcase_index  arch_name"
hw_configs=(
#  "2 5 2npu2pim"
#  "5 10 2npu"
#  "6 10 2pim"
#  "0 3 4npu4pimP4" # P4
######### Experiment #########
  "8 18 8npu8pimDP"     # NeuPIM (8+8) + DP (P18)
#  "9 18 8npu8pimDP"     # AttAcc (8+8) + DP (P18)
)

for hw in "${hw_configs[@]}"; do
  read -r hcase_index pcase_index arch_name <<< "$hw"
  for req_dist in "${req_dist_configs[@]}"; do
    read -r req0 req1 req2 peak_len<<< "$req_dist"
    for lam in 50; do
      OUT="result/model${model_index}_pareto_wi_sgbs_req${req_type_num}_${arch_name}_lam${lam}_${req0}_${req1}_${req2}_npDP_noTP.jsonl"
#      DSE_OUT="result/dse/model${model_index}_pareto_wi_sgbs_req${req_type_num}_${arch_name}_lam${lam}_${req0}_${req1}_${req2}_noDP_noTP.jsonl"
#      OUT="result/model${model_index}_pareto_wo_sgbs_req${req_type_num}_${arch_name}_lam${lam}_${req0}_${req1}_${req2}.jsonl"
#      DSE_OUT="result/dse/model${model_index}_pareto_wo_sgbs_req${req_type_num}_${arch_name}_lam${lam}_${req0}_${req1}_${req2}.jsonl"
      # 实验一律使用 wo
#      OUT="result/model${model_index}_req${req_type_num}_helix_lam${lam}_${req0}_${req1}_${req2}.jsonl"
#      OUT="result/model${model_index}_req${req_type_num}_dynamo_lam${lam}_${req0}_${req1}_${req2}.jsonl"
#      OUT="result/model${model_index}_req${req_type_num}_hexgen_lam${lam}_${req0}_${req1}_${req2}.jsonl"
#      OUT="result/model${model_index}_req${req_type_num}_symbiotree_lam${lam}_${req0}_${req1}_${req2}.jsonl"

      rm -f "$OUT"
      rm -f "$DSE_OUT"
      echo "Running req_dist=[$req0,$req1,$req2], lam=$lam"
      # 这里的 pcase-index 和 max-batch-lo 是不重要的。这里只是为了共享接口。
      /Users/SXT/anaconda3/envs/mytrans/bin/python run_evolution.py \
        --model-index "$model_index" \
        --hcase-index "$hcase_index"\
        --pcase-index "$pcase_index"\
        --req-type-num "$req_type_num" \
        --req-dist "[$req0,$req1,$req2]"\
        --lam $lam \
        --t-end 100 \
        --priority-ratio 0.0 \
        --max-batch-lo 512 \
        --peak-seq-len "$peak_len" \
        --out "$OUT"
#        --dse-out "$DSE_OUT"
    done
  done
done
echo "Done. Wrote $OUT"