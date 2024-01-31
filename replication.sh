#!/bin/bash

# This file specifies values on hyperparameters for replicating the study.

# Set the following variables.

save_root= # choose some directory
seed= # choose one of the following: 111, 222, 333, 444, 555.
vocab_size= # 2 to 16384
seq_length= # 64 by default. Set it to 128, 256, or 512 for replicating Fig. 3.
time_encoding= # "concat" for the use of positional encoding. leave it blank for turning off positional encoding.
rnn_name= # "LSTM", "GRU", or "RNN" (for Elman).

sphere_dim= # Dimenstionaly of the hypersphere (+1) used for investigating "infinite" vocabularies. 8-512.

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


# Reverse-ordering task

## Canonical training (Fig. 2-4)

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
	$time_encoding_arg \
	--learnable_padding_token --num_held_out $num_held_out

## Vocabulary extension (Fig. 5)

extra_vocab_size=$vocab_size # Double the vocabulary size.

settings=${rnn_name}/len-${seq_length}/vocab-${vocab_size}/`[ -z "$time_encoding" ] && echo "no-time" || echo $time_encoding`
pretrained=$save_root/time-stamped_rnn/palindrome/$settings/${train_job_id}/checkpoint.pt
save_dir=$save_root/time-stamped_rnn/palindrome_vocab-extension/$settings/${train_job_id}

python code/extend-vocab_palindrome.py $pretrained $extra_vocab_size $seq_length $save_dir \
	--num_iterations $train_iterations \
	--batch_size $batch_size --saving_interval $saving_interval --num_workers $num_workers \
	--learning_rate $learning_rate --warmup_iters $warmup_iters --device $device --seed $seed \
	--num_held_out $num_held_out

## Variable input length (Fig. 6)

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

## Autoregression (Fig. 8)

settings=${rnn_name}/len-${seq_length}/vocab-${vocab_size}/autoregression
save_dir=$save_root/time-stamped_rnn/palindrome_autoregression/$settings/${train_job_id}

python code/train_palindrome_autoregression.py $vocab_size $seq_length $save_dir \
	--rnn_name $rnn_name \
	--hidden_size $hidden_size \
	--embed_size $embed_size \
	--num_layers $num_layers \
	--dropout $dropout \
	--num_iterations $train_iterations \
	--batch_size $batch_size --saving_interval $saving_interval --num_workers $num_workers \
	--learning_rate $learning_rate --warmup_iters $warmup_iters --device $device --seed $seed \
	--learnable_padding_token --num_held_out $num_held_out

## Infinite vocabulary (Fig. 9)



settings=${rnn_name}/len-${seq_length}/sphere_dim-${sphere_dim}/`[ -z "$time_encoding" ] && echo "no-time" || echo $time_encoding`
save_dir=$save_root/time-stamped_rnn/palindrome_continuous/$settings/${train_job_id}

python code/train_palindrome_continuous.py $sphere_dim $seq_length $save_dir \
	--rnn_name $rnn_name \
	--hidden_size $hidden_size \
	--embed_size $embed_size \
	--num_layers $num_layers \
	--dropout $dropout \
	--num_iterations $train_iterations \
	--batch_size $batch_size --saving_interval $saving_interval --num_workers $num_workers \
	--learning_rate $learning_rate --warmup_iters $warmup_iters --device $device --seed $seed \
	$time_encoding_arg \
	--num_test_seqs $num_test_seqs


# Copying-Memory Task (Fig. 11)

settings=${rnn_name}/len-${seq_length}/vocab-${vocab_size}/delay-${delay}/`[ -z "$time_encoding" ] && echo "no-time" || echo $time_encoding`
save_dir=$save_root/time-stamped_rnn/copy-memory/$settings/${train_job_id}

python code/train_copy-memory.py $delay $save_dir \
	--vocab_size $vocab_size --seq_length $seq_length \
	--rnn_name $rnn_name \
	--hidden_size $hidden_size \
	--embed_size $embed_size \
	--num_layers $num_layers \
	--dropout $dropout \
	--num_iterations $train_iterations \
	--batch_size $batch_size --saving_interval $saving_interval --num_workers $num_workers \
	--learning_rate $learning_rate --warmup_iters $warmup_iters --device $device --seed $seed \
	$time_encoding_arg \
	--num_held_out $num_held_out


# Sorting task

## Canonical training (Fig. 12)

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
	$time_encoding_arg \
	--learnable_padding_token --num_held_out $num_held_out

### Check the order of the outputs of the trained model, ignoring the accuracy against the ground-truth target.
python code/test_sort_pairwise-order.py $save_dir/checkpoint.pt --device cuda


## Autoregression (Fig. 13)

settings=${rnn_name}/len-${seq_length}/vocab-${vocab_size}/autoregression
save_dir=$save_root/time-stamped_rnn/sort_autoregression/$settings/${train_job_id}

python code/train_sort_autoregression.py $vocab_size $seq_length $save_dir \
	--rnn_name $rnn_name \
	--hidden_size $hidden_size \
	--embed_size $embed_size \
	--num_layers $num_layers \
	--dropout $dropout \
	--num_iterations $train_iterations \
	--batch_size $batch_size --saving_interval $saving_interval --num_workers $num_workers \
	--learning_rate $learning_rate --warmup_iters $warmup_iters --device $device --seed $seed \
	--learnable_padding_token --num_held_out $num_held_out

