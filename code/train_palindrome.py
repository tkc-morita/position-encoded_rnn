# coding: utf-8

import os,argparse
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from utils.training_template import Learner as _Learner
from utils.logging import get_logger
from data.dataset import RandomSequence

class Learner(_Learner):
	def train_per_iteration(self, sequence, records, iteration):
		self.optimizer.zero_grad()
		sequence = sequence.to(self.device)

		palindrome = sequence.flip(dims=(1,))
		vocab_size = self.checkpoint['modules']['rnn']['init_args']['vocab_size']
		dummy_input = torch.full_like(sequence, vocab_size) # NOTE: Mapped to a fixed zero vector or learnable filler.
		input = torch.cat([sequence, dummy_input], dim=1)

		logits = self.rnn(input)
		logits = logits[:,sequence.size(1):,:] # Strip-off the encoding phase.
		loss = F.cross_entropy(logits.reshape(-1,vocab_size), palindrome.view(-1))
		self.update_records(records, 'loss', loss.item())
		accuracy = (logits.argmax(dim=-1)==palindrome).float().mean()
		self.update_records(records, 'accuracy', accuracy.item())

		loss.backward()
		clip_grad_norm_(self.get_parameters(), 1.0)
		self.optimizer.step()
		self.scheduler.step(iteration)
		return records

	def log_training_stats(self, records, saving_interval):
		self.logger.info('Cross entropy loss: {:0.6f}'.format(records['loss']/saving_interval))
		self.logger.info('Accuracy: {:0.6f}'.format(records['accuracy']/saving_interval))

	def test(self, sequence):
		sequence = sequence.to(self.device)

		palindrome = sequence.flip(dims=(1,))
		vocab_size = self.checkpoint['modules']['rnn']['init_args']['vocab_size']
		dummy_input = torch.full_like(sequence, vocab_size) # NOTE: Mapped to a fixed zero vector or learnable filler.
		input = torch.cat([sequence, dummy_input], dim=1)

		logits = self.rnn(input)
		logits = logits[:,sequence.size(1):,:] # Strip-off the encoding phase.
		
		is_correct = logits.argmax(dim=-1)==palindrome
		token_accuracy = is_correct.float().mean().item()
		seq_accuracy = is_correct.all(dim=-1).float().mean().item()
		self.logger.info('Test accuracy (token): {}'.format(token_accuracy))
		self.logger.info('Test accuracy (sequence-wise full-match): {}'.format(seq_accuracy))


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('vocab_size', type=int, help='Vocabulary size.')
	parser.add_argument('seq_length', type=int, help='Sequence length.')
	parser.add_argument('save_dir', type=str, help='Path to the directory where results are saved.')

	parser.add_argument('--num_held_out', type=int, default=0, help='# of random sequences to be held out for testing.')

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

	logger.info('Learns palindrome.')
	logger.info('Vocabulary size: {}'.format(args.vocab_size))
	logger.info('Sequence length: {}'.format(args.seq_length))

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
	dataset = RandomSequence(args.vocab_size, args.seq_length, args.num_held_out)
	learner(dataset, args.num_iterations, args.batch_size, args.saving_interval, args.num_workers)