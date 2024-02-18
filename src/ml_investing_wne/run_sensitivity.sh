#!/bin/bash

models=("keras_tuner_CNN_LSTM")

# data_params=(

#   "0.01,0.02,ETHUSDT" \
#   "0.01,0.025,ETHUSDT" \
#   "0.01,0.035,ETHUSDT" \
#   "0.01,0.07,ETHUSDT" \
#   "0.01,0.08,ETHUSDT" \
#   "0.01,0.09,ETHUSDT" \
#   "0.01,0.1,ETHUSDT" \
#   "0.015,0.08,ETHUSDT" \
#   "0.015,0.09,ETHUSDT" \
#   "0.015,0.1,ETHUSDT" \
#   "0.02,0.02,ETHUSDT" \
#   "0.02,0.025,ETHUSDT" \
#   "0.02,0.035,ETHUSDT" \
#   "0.02,0.05,ETHUSDT" \
#   "0.02,0.09,ETHUSDT" \
#   "0.03,0.035,ETHUSDT" \
#   "0.03,0.05,ETHUSDT" \ 
#   "0.03,0.09,ETHUSDT" \
#   "0.06,0.08,ETHUSDT" \
# )


# data_params=(

#   "0.025,0.05,MATICUSDT" \
#   "0.025,0.06,MATICUSDT" \
#   "0.02,0.06,MATICUSDT" \
#   "0.025,0.05,LTCUSDT" \
#   "0.025,0.06,LTCUSDT" \
#   "0.02,0.06,LTCUSDT" \
# )


# data_params=(

#   "0.025,0.05,SOLUSDT" \
#   "0.025,0.06,SOLUSDT" \
#   "0.02,0.06,SOLUSDT" \
#   "0.025,0.05,DOGEUSDT" \
#   "0.025,0.06,DOGEUSDT" \
#   "0.02,0.06,DOGEUSDT" \
# )

# data_params=(

#   "0.025,0.05,ADAUSDT" \
#   "0.025,0.06,ADAUSDT" \
#   "0.02,0.06,ADAUSDT" \
#   "0.025,0.05,LTCUSDT" \
#   "0.025,0.06,LTCUSDT" \
#   "0.02,0.06,LTCUSDT" \
#   "0.025,0.05,LINKUSDT" \
#   "0.025,0.06,LINKUSDT" \
#   "0.02,0.06,LINKUSDT" \
# )


data_params=(

  "0.02,0.09,ADAUSDT" \
  "0.025,0.09,ADAUSDT" \
  "0.02,0.09,LTCUSDT" \
  "0.025,0.09,LTCUSDT" \
  "0.02,0.09,LINKUSDT" \
  "0.025,0.09,LINKUSDT" \
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

# data_params=(
#   "0.03,ETHUSDT" \
#   "0.04,ETHUSDT" \
#   "0.05,ETHUSDT" \
#   "0.06,ETHUSDT" \
# )


# for model in "${models[@]}"; do
#     # Iterate over each tuple
#     for pair in "${data_params[@]}"; do
#         # Split the tuple into data parts and linked parameter
#         IFS=',' read -r cumsum_threshold currency <<< "$pair"

#         # Run your Python script with the current model, data parts, and linked parameter
#         python main_loop_ensemble.py \
#           --cumsum_threshold "$cumsum_threshold" \
#           --currency "$currency" \
#           --run_subtype "cumsum" \
#           --model $model 
#     done
# done