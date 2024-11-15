# coding: utf-8

import os,argparse
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from train_palindrome import Learner as _Learner
from utils.logging import get_logger
from data.dataset import FreqentVSRare
from data.dataloader import get_data_loader

class Learner(_Learner):
	def __call__(self, dataset, num_iterations, batch_size, saving_interval, num_workers=1):
		if dataset.held_out.ndim==2:
			super().__call__(dataset, num_iterations, batch_size, saving_interval, num_workers=num_workers)
		if self.retrieval:
			start_iter = self.last_iteration + 1
			self.logger.info('To be restarted from the beginning of iteration #: {iteration}'.format(iteration=start_iter+1))
		else:
			self.logger.info("START LEARNING.")
			self.logger.info("max # of iterations: {ep}".format(ep=num_iterations))
			self.logger.info("batch size for training data: {size}".format(size=batch_size))
			self.checkpoint['held_out_data'] = dataset.held_out
			patterns_held_out = 0 if dataset.held_out is None \
								else 'x'.join(map(str, dataset.held_out.size()[:-2])) # <- NOTE: Change here
			self.logger.info('{} patterns are held out for test.'.format(patterns_held_out)) # <- NOTE: Change here
			start_iter = 0
		dataloader = get_data_loader(
								dataset,
								batch_size=batch_size,
								start_iter=start_iter,
								num_iterations=num_iterations,
								shuffle=True,
								num_workers=num_workers,
								random_seed=self.seed)
		self.train(dataloader, num_iterations, saving_interval, start_iter=start_iter)
		self.logger.info('END OF TRAINING')
		if not dataset.held_out is None:
			self.logger.info('START OF TEST ON HELD-OUT DATA')
			with torch.no_grad():
				self.test(dataset.held_out)
			self.logger.info('END OF TEST ON HELD-OUT DATA')

	def test(self, sequence):
		original_shape = sequence.size()
		if len(original_shape)==2: # NOTE: No split b/w frequent vs. rare.
			return super().test(sequence)
		sequence = sequence.to(self.device).view(-1,original_shape[-1])

		palindrome = sequence.flip(dims=(1,))
		vocab_size = self.checkpoint['modules']['rnn']['init_args']['vocab_size']
		dummy_input = torch.full_like(sequence, vocab_size) # NOTE: Mapped to a fixed zero vector or learnable filler.
		input = torch.cat([sequence, dummy_input], dim=1)

		logits = self.rnn(input)
		logits = logits[:,sequence.size(1):,:] # Strip-off the encoding phase.
		
		is_correct = (logits.argmax(dim=-1)==palindrome).view(*original_shape) # -> 2 x 2 x N x L x L
		is_correct = is_correct.flip(dims=(-1,) # back to the input order
								).diagonal(offset=0, dim1=-2, dim2=-1) # -> 2 x 2 x N x L
		token_accuracy = is_correct.float().mean((-2)) # -> 2 x 2 x L
		# seq_accuracy = is_correct.all(dim=-1).float().mean(-1) # -> 2 x 2
		# token_accuracy = is_correct.float().mean((-3,-1)) # -> 2 x 2 x 2
		# seq_accuracy = is_correct.all(dim=-1).float().mean(-2) # -> 2 x 2 x 2
		for target_type_ix in range(2):
			for disturbant_type_ix in range(2):
				# self.logger.info('Test accuracy of {target_type} tokens surrounded by {disturbant_type} tokens (sequence-wise full-match): {accuracy}'.format(
				# 		target_type=['frequent','rare'][target_type_ix],
				# 		disturbant_type=['frequent','rare'][disturbant_type_ix],
				# 		accuracy=seq_accuracy[target_type_ix,disturbant_type_ix].item()))
				for target_pos in range(token_accuracy.size(-1)):#range(2):
					self.logger.info('Test accuracy of {target_type} tokens input at t={target_pos} surrounded by {disturbant_type} token (token): {accuracy}'.format(
						target_type=['frequent','rare'][target_type_ix],
						disturbant_type=['frequent','rare'][disturbant_type_ix],
						target_pos=target_pos,
						accuracy=token_accuracy[target_type_ix,disturbant_type_ix,target_pos].item()))
					# self.logger.info('Test accuracy of {prefix_or_suffix} in {target_type}-{disturbant_type} (sequence-wise full-match): {accuracy}'.format(
					# 	target_type=['frequent','rare'][target_type_ix],
					# 	disturbant_type=['frequent','rare'][disturbant_type_ix],
					# 	prefix_or_suffix=['suffix','prefix'][target_pos],
					# 	accuracy=seq_accuracy[target_type_ix,disturbant_type_ix,target_pos]))

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('vocab_size', type=int, help='Vocabulary size.')
	parser.add_argument('seq_length', type=int, help='Sequence length.')
	parser.add_argument('save_dir', type=str, help='Path to the directory where results are saved.')

	parser.add_argument('--num_held_out', type=int, default=0, help='TOTAL # of random sequences (frequent+rare) to be held out for testing.')
	parser.add_argument('--mixed_held_out', action='store_true', help='Hold out sequences consisting of a mixture of frequent and rare tokens.')
	parser.add_argument('--rarity', type=float, default=1/2, help='Probability of choosing the rare vocabulary.')
	parser.add_argument('--num_frequent', type=int, default=None, help='# of frequent vocabulary items. Equals to vocab_size/2 by default.')

	parser.add_argument('--rnn_name', type=str, required=True, choices=['RNN','GRU','LSTM'], help='Type of RNN.')
	parser.add_argument('--hidden_size', type=int, default=512, help='Dimensionality of hidden layer(s) in RNN.')
	parser.add_argument('--embed_size', type=int, default=None, help='Dimensionality of input (& time) embeddings. Equals to hidden_size if not specified.')
	parser.add_argument('--time_encoding', type=str, default=None, choices=['add','concat'], help='Specifies whether time encoding is added to or concatenated with the input embeddings. Time encoding is not used if this option is left unspecified.')
	parser.add_argument('--time_encoding_form', type=str, default='sinusoidal', choices=['sinusoidal','learnable','random'], help='Implementation of time encoding.')
	parser.add_argument('--num_layers', type=int, default=1, help='# of layers in RNN.')
	parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate in RNN.')
	parser.add_argument('--learnable_padding_token', action='store_true', help='Use a learnable embedding for the dummy token in the output phase. Otherwise, the dummy token is represented by the zero vector.')

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

	logger.info('Learns palindrome w/ frequent and rare vocabulary items.')
	logger.info('Vocabulary size: {}'.format(args.vocab_size))
	logger.info('Sequence length: {}'.format(args.seq_length))
	logger.info('Rarity: {}'.format(args.rarity))
	if not args.num_frequent is None:
		logger.info('Vocabulary is split unevenly into two partitions of size {num_frequent} vs. {vocab_size}-{num_frequent}.'.format(
					num_frequent=args.num_frequent,vocab_size=args.vocab_size))

	model_configs = dict()
	model_configs['rnn'] = dict(module_name='RNN',
								init_args=dict(
									vocab_size=args.vocab_size,
									hidden_size=args.hidden_size,
									rnn_name=args.rnn_name,
									embed_size=args.embed_size,
									time_encoding=args.time_encoding,
									time_encoding_form=args.time_encoding_form,
									max_length=args.seq_length*2,
									num_layers=args.num_layers,
									dropout=args.dropout,
									learnable_padding_token=args.learnable_padding_token,
								))
	optim_config = dict(lr=args.learning_rate, weight_decay=0.0, betas=(0.9,0.98), eps=1e-09)
	scheduler_config = dict(t_initial=args.num_iterations,
								warmup_t=args.warmup_iters,
								warmup_prefix=True, lr_min=0.0)
	learner = Learner(logger, args.save_dir, model_configs, optim_config, scheduler_config,
						device=args.device, seed=args.seed)
	dataset = FreqentVSRare(args.vocab_size, args.seq_length, args.num_held_out,
								mixed_held_out=args.mixed_held_out,
								rarity=args.rarity, num_frequent=args.num_frequent,
								dummy_datasize=max(512,args.batch_size))
	learner(dataset, args.num_iterations, args.batch_size, args.saving_interval, args.num_workers)