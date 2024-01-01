#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

models=("keras_tuner_CNN_LSTM" "keras_tuner_transformer_learnable_encoding" "keras_tuner_tsmixer_flattened")

data_params=(
  "0.01,0.025,ETHUSDT" \
  "0.02,0.05,ETHUSDT" \
  "0.03,0.05,ETHUSDT" \
  "0.01,0.025,BTCUSDT" \
  "0.02,0.05,BTCUSDT" \
  "0.03,0.05,BTCUSDT" \
)


for model in "${models[@]}"; do
    # Iterate over each tuple
    for pair in "${data_params[@]}"; do
        # Split the tuple into data parts and linked parameter
        IFS=',' read -r cumsum_threshold fixed_barrier currency <<< "$pair"

        # Run your Python script with the current model, data parts, and linked parameter
        python main_loop_ensemble.py \
          --cumsum_threshold "$cumsum_threshold" \
          --fixed_barrier "$fixed_barrier" \
          --currency "$currency" \
          --run_subtype "range_bar" \
          --model $model 
    done
done