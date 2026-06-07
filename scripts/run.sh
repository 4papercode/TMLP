#!/bin/bash

export TOKENIZERS_PARALLELISM=false
GPU=${1:-0}

BA=CISO
YEAR=2018
SEQ_LEN=168
LABEL_LEN=48
PRED_LEN=24
ENC_IN=1
LR=0.001
L1=0.5
L2=0.5
dropout=0

mkdir -p logs

echo "========================================"
echo "$BA  year=$YEAR  GPU=$GPU"
echo "========================================"

python run.py \
  --task_name        long_term_forecast \
  --is_training      1 \
  --model            TMLP \
  --model_id         wecc_${BA}_${YEAR} \
  --data             WECC \
  --ba_name          $BA \
  --use_spatial      True \
  --use_topology     True \
  --pi_root_1        PI \
  --pi_root_2        PI_ba_groups \
  --features         S \
  --target           Cleaned_Demand_MWh \
  --enc_in           $ENC_IN \
  --dec_in           $ENC_IN \
  --c_out            1 \
  --seq_len          $SEQ_LEN \
  --label_len        $LABEL_LEN \
  --pred_len         $PRED_LEN \
  --year             $YEAR \
  --wecc_data_root   ./cleaned_data_16-24 \
  --similarity_path  utils/wecc_similarity.npy \
  --neighbors_path   utils/wecc_neighbors.json \
  --ba_names_path    utils/wecc_ba_names.json \
  --adj_path         utils/wecc_adj.csv \
  --vlm_type         CLIP \
  --d_model          128 \
  --dropout          $dropout \
  --learning_rate    $LR \
  --batch_size       32 \
  --train_epochs     20 \
  --patience         5 \
  --use_mem_gate     True \
  --learnable_image  True \
  --periodicity      24 \
  --align_lambda1    $L1 \
  --align_lambda2    $L2 \
  --gpu              $GPU \
  2>&1 | tee logs/${BA}_${YEAR}.log

echo "Finished Trial"
echo ""
