#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

models=("keras_tuner_CNN_LSTM")

data_params=(

  "0.06,0.02,ETHUSDT" \
  "0.06,0.025,ETHUSDT" \
  "0.06,0.03,ETHUSDT" \
  "0.06,0.035,ETHUSDT" \
  "0.06,0.04,ETHUSDT" \
  "0.06,0.05,ETHUSDT" \
  "0.06,0.06,ETHUSDT" \
  "0.06,0.07,ETHUSDT" \
  "0.06,0.08,ETHUSDT" \
  "0.06,0.09,ETHUSDT" \
  "0.06,0.1,ETHUSDT" \
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
          --run_subtype "cumsum_triple_barrier" \
          --model $model 
    done
done