#!/bin/bash

# This file specifies values on hyperparameters for replicating the study.

# Set the following variables.

save_root= # choose some directory
seed= # choose one of the following: 111, 222, 333, 444, 555.
vocab_size= # 32-256 (GRU) or 256-16384 (LSTM)
seq_length=64
time_encoding= # "concat" for the use of positional encoding. leave it blank for turning off positional encoding.
rnn_name= # "GRU" or "LSTM".
time_encoding_form=sinusoidal # Alternatively, "learnable" or "random" for replicating Fig. 10 in the appendix.

# Common parameters

num_held_out=1024

hidden_size=512
embed_size=512
num_layers=1
dropout=0.0

train_iterations=300000
batch_size=512
saving_interval=5000
num_workers=4
warmup_iters=1000
learning_rate=1e-3

device=cuda

train_job_id=seed-${seed}


# Training on the reverse-ordering task (Fig. 2)

settings=${rnn_name}/len-${seq_length}/vocab-${vocab_size}/`[ -z "$time_encoding" ] && echo "no-time" || echo $time_encoding`
save_dir=$save_root/time-stamped_rnn/palindrome/$settings/${train_job_id}
time_encoding_arg=`[ ! -z "$time_encoding" ] && echo "--time_encoding $time_encoding"`

python code/train_palindrome.py $vocab_size $seq_length $save_dir \
	--rnn_name $rnn_name \
	--hidden_size $hidden_size \
	--embed_size $embed_size \
	--num_layers $num_layers \
	--dropout $dropout \
	--num_iterations $train_iterations \
	--batch_size $batch_size --saving_interval $saving_interval --num_workers $num_workers \
	--learning_rate $learning_rate --warmup_iters $warmup_iters --device $device --seed $seed \
	$time_encoding_arg --time_encoding_form $time_encoding_form \
	--learnable_padding_token --num_held_out $num_held_out

# Training on the reverse-ordering task w/ a dual-frequency vocabulary (Fig. 3)
rarity=0.125 #=1/8
save_dir=$save_root/time-stamped_rnn/palindrome_dual-freq_single-target/$settings/${train_job_id}
python code/train_palindrome_dual-freq.py $vocab_size $seq_length $save_dir \
	--rnn_name $rnn_name \
	--hidden_size $hidden_size \
	--embed_size $embed_size \
	--num_layers $num_layers \
	--dropout $dropout \
	--num_iterations $train_iterations \
	--batch_size $batch_size --saving_interval $saving_interval --num_workers $num_workers \
	--learning_rate $learning_rate --warmup_iters $warmup_iters --device $device --seed $seed \
	$time_encoding_arg \
	--learnable_padding_token --num_held_out $num_held_out --rarity $rarity

# Analyze the gradients of the RNN trained on the dual-frequency vocabulary (Fig. 5)
num_seq_triplets=1024
batch_size=`[ ${rnn_name} = "RNN" ] && echo 32 || echo 16`
python code/test_jacobian_palindrome.py $save_dir --device $device $batch_size_option --num_seq_triplets $num_seq_triplets --batch_size $batch_size

## Variable input length (Fig. 9 in Appendix C)
min_length=32
max_length=64

settings=${rnn_name}/len-${min_length}-${max_length}/vocab-${vocab_size}/`[ -z "$time_encoding" ] && echo "no-time" || echo $time_encoding`
save_dir=$save_root/time-stamped_rnn/palindrome_variable-length/$settings/${train_job_id}

python code/train_palindrome_variable-length.py $vocab_size $min_length $max_length $save_dir \
	--rnn_name $rnn_name \
	--hidden_size $hidden_size \
	--embed_size $embed_size \
	--num_layers $num_layers \
	--dropout $dropout \
	--num_iterations $train_iterations \
	--batch_size $batch_size --saving_interval $saving_interval --num_workers $num_workers \
	--learning_rate $learning_rate --warmup_iters $warmup_iters --device $device --seed $seed \
	$time_encoding_arg \
	--learnable_padding_token --num_held_out $num_held_out

# Training on the sorting task. (Fig. 8 in Appendix A)

settings=${rnn_name}/len-${seq_length}/vocab-${vocab_size}/`[ -z "$time_encoding" ] && echo "no-time" || echo $time_encoding`
save_dir=$save_root/time-stamped_rnn/sort/$settings/${train_job_id}

python code/train_sort.py $vocab_size $seq_length $save_dir \
	--rnn_name $rnn_name \
	--hidden_size $hidden_size \
	--embed_size $embed_size \
	--num_layers $num_layers \
	--dropout $dropout \
	--num_iterations $train_iterations \
	--batch_size $batch_size --saving_interval $saving_interval --num_workers $num_workers \
	--learning_rate $learning_rate --warmup_iters $warmup_iters --device $device --seed $seed \
	$time_encoding_arg --time_encoding_form $time_encoding_form \
	--learnable_padding_token --num_held_out $num_held_out

# Reverse-ordering + Delayed-sum (Fig. 7 in Appendix A)
rnn_name=LSTM # Only LSTM was tested
vocab_size= # 896-1088
seq_length=16
train_iterations=600000

settings=${rnn_name}/len-${seq_length}/vocab-${vocab_size}/`[ -z "$time_encoding" ] && echo "no-time" || echo $time_encoding`
save_dir=$save_root/time-stamped_rnn/delayed-sum/$settings/${train_job_id}

python code/train_delayed-sum.py $vocab_size $seq_length $save_dir \
	--rnn_name $rnn_name \
	--hidden_size $hidden_size \
	--embed_size $embed_size \
	--num_layers $num_layers \
	--dropout $dropout \
	--num_iterations $train_iterations \
	--batch_size $batch_size --saving_interval $saving_interval --num_workers $num_workers \
	--learning_rate $learning_rate --warmup_iters $warmup_iters --device $device --seed $seed \
	$time_encoding_arg \
	--learnable_padding_token --num_held_out $num_held_out