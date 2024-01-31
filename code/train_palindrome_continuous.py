# coding: utf-8

import os,argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from utils.training_template import Learner as _Learner
from utils.logging import get_logger
from data.dataset import RandomSphere

class Learner(_Learner):
	def train_per_iteration(self, sequence, records, iteration):
		self.optimizer.zero_grad()
		sequence = sequence.to(self.device)

		palindrome = sequence.flip(dims=(1,))
		dummy_input = torch.zeros_like(sequence)
		input = torch.cat([sequence, dummy_input], dim=1)

		prediction = self.rnn(input)
		prediction = prediction[:,sequence.size(1):,:] # Strip-off the encoding phase.
		prediction = F.normalize(prediction, p=2.0, dim=-1)
		similarity = (prediction*palindrome).sum(dim=-1).mean()
		self.update_records(records, 'similarity', similarity.item())

		(-similarity).backward()
		clip_grad_norm_(self.get_parameters(), 1.0)
		self.optimizer.step()
		self.scheduler.step(iteration)
		return records

	def log_training_stats(self, records, saving_interval):
		self.logger.info('Cosine similarity: {:0.6f}'.format(records['similarity']/saving_interval))

	def test(self, sequence):
		sequence = sequence.to(self.device)

		palindrome = sequence.flip(dims=(1,))
		dummy_input = torch.zeros_like(sequence)
		input = torch.cat([sequence, dummy_input], dim=1)

		prediction = self.rnn(input)
		prediction = prediction[:,sequence.size(1):,:] # Strip-off the encoding phase.
		prediction = F.normalize(prediction, p=2.0, dim=-1)
		
		similarity = (prediction*palindrome).sum(dim=-1)
		self.logger.info('Test cosine similarity (token): {}'.format(similarity.mean().item()))
		np.save(os.path.join(self.save_dir, 'test_similarity.npy'), similarity.cpu().numpy())
		self.logger.info('Full similarity results were saved in an numpy array.')

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('input_size', type=int, help='Dimensionality of input tokens.')
	parser.add_argument('seq_length', type=int, help='Sequence length.')
	parser.add_argument('save_dir', type=str, help='Path to the directory where results are saved.')

	parser.add_argument('--num_test_seqs', type=int, default=1024, help='# of random sequences to be tested.')

	parser.add_argument('--rnn_name', type=str, required=True, choices=['RNN','GRU','LSTM'], help='Type of RNN.')
	parser.add_argument('--hidden_size', type=int, default=512, help='Dimensionality of hidden layer(s) in RNN.')
	parser.add_argument('--embed_size', type=int, default=None, help='Dimensionality of input (& time) embeddings. Equals to hidden_size if not specified.')
	parser.add_argument('--time_encoding', type=str, default=None, choices=['add','concat'], help='Specifies whether time encoding is added to or concatenated with the input embeddings. Time encoding is not used if this option is left unspecified.')
	parser.add_argument('--num_layers', type=int, default=1, help='# of layers in RNN.')
	parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate in RNN.')

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

	logger.info('Learns palindrome of random points on hyper-sphere.')
	logger.info('Input dimensionality: {}'.format(args.input_size))
	logger.info('Sequence length: {}'.format(args.seq_length))

	model_configs = dict()
	model_configs['rnn'] = dict(module_name='RNN_w_ContinuousInput',
								init_args=dict(
									input_size=args.input_size,
									hidden_size=args.hidden_size,
									rnn_name=args.rnn_name,
									embed_size=args.embed_size,
									time_encoding=args.time_encoding,
									num_layers=args.num_layers,
									dropout=args.dropout,
								))
	optim_config = dict(lr=args.learning_rate, weight_decay=0.0, betas=(0.9,0.98), eps=1e-09)
	scheduler_config = dict(t_initial=args.num_iterations,
								warmup_t=args.warmup_iters,
								warmup_prefix=True, lr_min=0.0)
	learner = Learner(logger, args.save_dir, model_configs, optim_config, scheduler_config,
						device=args.device, seed=args.seed)
	dataset = RandomSphere(args.input_size, args.seq_length, args.num_test_seqs)
	learner(dataset, args.num_iterations, args.batch_size, args.saving_interval, args.num_workers)