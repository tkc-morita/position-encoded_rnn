# coding: utf-8

import os,argparse
import torch
import torch.nn.functional as F
from utils.logging import get_logger
from train_sort import Learner

class Tester(Learner):
	def __init__(self, logger, checkpoint_path, device='cpu'):
		self.logger = logger
		self.device = torch.device(device)
		self.retrieve_model(checkpoint_path)

	def __call__(self):
		sequence = self.checkpoint['held_out_data']
		sequence = sequence.to(self.device)

		with torch.no_grad():
			vocab_size = self.checkpoint['modules']['rnn']['init_args']['vocab_size']
			dummy_input = torch.full_like(sequence, vocab_size) # NOTE: Mapped to a fixed zero vector or learnable filler.
			input = torch.cat([sequence, dummy_input], dim=1)

			logits = self.rnn(input)
			logits = logits[:,sequence.size(1):,:] # Strip-off the encoding phase.

			prediction = F.one_hot(logits.argmax(dim=-1), # batch_size x duration
								num_classes=vocab_size).sum(dim=1) # batch_size x vocab_size

			counts = F.one_hot(sequence, num_classes=vocab_size).sum(dim=1) # batch_size x vocab_size

			deviation = (prediction-counts).float().abs().sum(dim=-1)*0.5 # NOTE: Each mismatch increments +2 deviations.

			self.logger.info('Mean half L1 distance in counts per sequence: {}'.format(deviation.mean().item()))
			self.logger.info('STD half L1 distance in counts per sequence: {}'.format(deviation.std().item()))

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('checkpoint', type=str, help='Path to the checkpoint of the trained model.')
	# parser.add_argument('--log_path', type=str, default=None, help='Path to the .log file where results are logged.')
	parser.add_argument('--device', type=str, default='cpu', help='Device.')
	args = parser.parse_args()

	# if not args.log_path is None:
		# os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
	logger = get_logger()#args.log_path,filename='bag-of-words.log')

	tester = Tester(logger, args.checkpoint, device=args.device)
	tester()