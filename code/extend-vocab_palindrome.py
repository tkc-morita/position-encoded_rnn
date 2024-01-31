# coding: utf-8

import os,argparse
import torch
from train_palindrome import Learner
from utils.logging import get_logger
from data.dataset import RandomSequence
from model.rnn import RNN

def load_pretrained_rnn(checkpoint_path):
	cp = torch.load(checkpoint_path, map_location='cpu')
	rnn = RNN(**cp['modules']['rnn']['init_args'])
	rnn.load_state_dict(cp['modules']['rnn']['state_dict'])
	return rnn,cp['modules']['rnn']['init_args']['vocab_size']

def reset_heldout_data(dataset: RandomSequence, num_held_out: int, extra_vocab_size: int):
	if not num_held_out:
		return None
	possible_patterns = extra_vocab_size**dataset.length
	assert possible_patterns>num_held_out, 'Cannot hold out {num_held_out} sequences from {possible_patterns} patterns.'.format(num_held_out=num_held_out, possible_patterns=possible_patterns)
	held_out = torch.randint(extra_vocab_size, size=(1, dataset.length))
	while held_out.size(0)<num_held_out:
		candidate = torch.randint(extra_vocab_size, size=(1, dataset.length))
		if (candidate!=held_out).any(dim=-1).all(dim=0).item(): # check duplication
			held_out = torch.cat([held_out,candidate], dim=0)
	dataset.held_out = held_out

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('pretrained', type=str, help='Path to the checkpoint of the pretrained model.')
	parser.add_argument('extra_vocab_size', type=int, help='Extra vocabulary size.')
	parser.add_argument('seq_length', type=int, help='Sequence length.')
	parser.add_argument('save_dir', type=str, help='Path to the directory where results are saved.')

	parser.add_argument('--num_held_out', type=int, default=0, help='# of random sequences to be held out for testing.')

	# parser.add_argument('--rnn_name', type=str, required=True, choices=['RNN','GRU','LSTM'], help='Type of RNN.')
	# parser.add_argument('--hidden_size', type=int, default=512, help='Dimensionality of hidden layer(s) in RNN.')
	# parser.add_argument('--embed_size', type=int, default=None, help='Dimensionality of input (& time) embeddings. Equals to hidden_size if not specified.')
	# parser.add_argument('--time_encoding', type=str, default=None, choices=['add','concat'], help='Specifies whether time encoding is added to or concatenated with the input embeddings. Time encoding is not used if this option is left unspecified.')
	# parser.add_argument('--num_layers', type=int, default=1, help='# of layers in RNN.')
	# parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate in RNN.')
	# parser.add_argument('--learnable_padding_token', action='store_true', help='Use a learnable embedding for the dummy token in the output phase. Otherwise, the dummy token is represented by the zero vector.')

	parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate of Adam optimizer.')
	parser.add_argument('--num_iterations', type=int, default=10000, help='# of training iterations.')
	parser.add_argument('--warmup_iters', type=int, default=0, help='# of warm-up iterations.')
	parser.add_argument('--saving_interval', type=int, default=1, help='Intervals of logging of the learning progress.')
	parser.add_argument('--batch_size', type=int, default=512, help='Batch size.')
	parser.add_argument('--num_workers', type=int, default=1, help='# of dataloading workers.')

	parser.add_argument('--device', type=str, default='cpu', help='Device.')
	parser.add_argument('--seed', type=int, default=111, help='Random seed.')
	args = parser.parse_args()

	os.makedirs(args.save_dir, exist_ok=True)
	logger = get_logger(args.save_dir)

	rnn,original_vocab_size = load_pretrained_rnn(args.pretrained)

	logger.info('Extend the vocabulary of RNN pretrained on palindrome.')
	logger.info('Original vocabulary size: {}'.format(original_vocab_size))
	logger.info('Extra vocabulary size: {}'.format(args.extra_vocab_size))
	logger.info('Sequence length: {}'.format(args.seq_length))

	model_configs = dict()
	model_configs['rnn'] = dict(module_name='VocabExtender',
								init_args=dict(
									base=rnn,
									extra_vocab_size=args.extra_vocab_size,
									vocab_size=args.extra_vocab_size+original_vocab_size, # NOTE: dummy kwarg for reference by the Learner.
									))
	optim_config = dict(lr=args.learning_rate, weight_decay=0.0, betas=(0.9,0.98), eps=1e-09)
	scheduler_config = dict(t_initial=args.num_iterations,
								warmup_t=args.warmup_iters,
								warmup_prefix=True, lr_min=0.0)
	learner = Learner(logger, args.save_dir, model_configs, optim_config, scheduler_config,
						device=args.device, seed=args.seed)
	dataset = RandomSequence(args.extra_vocab_size+original_vocab_size, args.seq_length, 0)
	reset_heldout_data(dataset, args.num_held_out, args.extra_vocab_size)
	learner(dataset, args.num_iterations, args.batch_size, args.saving_interval, args.num_workers)