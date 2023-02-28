#!/bin/bash


python  ../main.py --model_id transformer --exp_id ettm1_transformer --data ettm1_output_data_points.parquet --target_var OT --d_model 32 --n_heads 8 --e_layers 2 --d_layers 2 --d_ff 64

python  ../main.py --model_id transformer --exp_id ettm2_transformer --data ettm2_output_data_points.parquet --target_var OT --d_model 32 --n_heads 8 --e_layers 2 --d_layers 2 --d_ff 64

python  ../main.py --model_id transformer --exp_id solar_transformer --data solar_output_data_points.parquet  --d_model 256 --n_heads 8 --e_layers 2 --d_layers 2 --d_ff 128

python  ../main.py --model_id transformer --exp_id weather_transformer --data weather_output_data_points.parquet --target_var OT --d_model 128 --n_heads 4 --e_layers 3 --d_layers 3 --d_ff 512

python  ../main.py --model_id transformer --exp_id wind_transformer --data wind_output_data_points.parquet --target_var active_power --d_model 64 --n_heads 4 --e_layers 2 --d_layers 2 --d_ff 512


python  ../main.py --model_id nbeats --exp_id ettm1_nbeats --data ettm1_output_data_points.parquet --target_var OT --num_stacks 15 --num_blocks 1 --num_layers 4 --layer_widths 64

python  ../main.py --model_id nbeats --exp_id ettm2_nbeats --data ettm2_output_data_points.parquet --target_var OT --num_stacks 15 --num_blocks 1 --num_layers 4 --layer_widths 64

python  ../main.py --model_id nbeats --exp_id solar_nbeats --data solar_output_data_points.parquet  --num_stacks 15 --num_blocks 1 --num_layers 4 --layer_widths 64

python  ../main.py --model_id nbeats --exp_id weather_nbeats --data weather_output_data_points.parquet --target_var OT --num_stacks 30 --num_blocks 1 --num_layers 8 --layer_widths 64

python  ../main.py --model_id nbeats --exp_id wind_nbeats --data wind_output_data_points.parquet --target_var active_power --num_stacks 30 --num_blocks 1 --num_layers 4 --layer_widths 256


python  ../main.py --model_id gru --exp_id ettm1_gru --data ettm1_output_data_points.parquet --target_var OT --n_rnn_layers 2 --hidden_dim 32

python  ../main.py --model_id gru --exp_id ettm2_gru --data ettm2_output_data_points.parquet --target_var OT --n_rnn_layers 2 --hidden_dim 64

python  ../main.py --model_id gru --exp_id solar_gru --data solar_output_data_points.parquet --n_rnn_layers 3 --hidden_dim 64 --dropout 0.05

python  ../main.py --model_id gru --exp_id weather_gru --data weather_output_data_points.parquet --target_var OT --n_rnn_layers 1 --hidden_dim 32 --dropout 0.05

python  ../main.py --model_id gru --exp_id wind_gru --data wind_output_data_points.parquet --target_var active_power --n_rnn_layers 2 --hidden_dim 64 --dropout 0.05


python ../main.py --model_id dlinear  --data ettm1_output_data_points.parquet --target_var OT

python ../main.py --model_id dlinear  --data ettm2_output_data_points.parquet --target_var OT

python ../main.py --model_id dlinear  --data solar_output_data_points.parquet --target_var OT

python ../main.py --model_id dlinear  --data weather_output_data_points.parquet --target_var OT

python -u ../main.py --model_id dlinear  --data wind_output_data_points.parquet --target_var active_power


python -u ../main.py --model_id informer --data ettm1_output_data_points.parquet  --target_var OT --features S --seq_len 96 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 10

python -u ../main.py --model_id informer --data ettm2_output_data_points.parquet  --target_var OT --features S --seq_len 96 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 10

python -u ../main.py --model_id informer --data solar_output_data_points.parquet  --target_var OT --features M --seq_len 96 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 10

python -u ../main.py --model_id informer --data weather_output_data_points.parquet  --target_var OT --features S --seq_len 96 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 10

python -u ../main.py --model_id informer --data wind_output_data_points.parquet  --target_var active_power --features S --seq_len 96 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 10