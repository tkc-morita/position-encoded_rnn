# coding: utf-8

import argparse
import torch
from rapidfuzz.distance import DamerauLevenshtein
from utils.logging import get_logger
from train_palindrome import Learner

class Tester(Learner):
	def __init__(self, logger, checkpoint_path, device='cpu'):
		self.logger = logger
		self.device = torch.device(device)
		self.retrieve_model(checkpoint_path)

	def __call__(self):
		sequence = self.checkpoint['held_out_data']
		sequence = sequence.to(self.device)

		with torch.no_grad():
			target = sequence.flip(dims=(1,))
			network_type = 'ssm' if 'ssm' in self.checkpoint['modules'] else 'rnn'
			vocab_size = self.checkpoint['modules'][network_type]['init_args']['vocab_size']
			dummy_input = torch.full_like(sequence, vocab_size) # NOTE: Mapped to a fixed zero vector or learnable filler.
			input = torch.cat([sequence, dummy_input], dim=1)

			logits = getattr(self, network_type)(input)
			logits = logits[:,sequence.size(1):,:] # Strip-off the encoding phase.

			prediction = logits.argmax(dim=-1) # batch_size x duration

			dist = torch.tensor([DamerauLevenshtein.distance(p.tolist(),t.tolist())
					for p,t in zip(prediction.cpu().unbind(0),target.cpu().unbind(0))]
					, dtype=torch.float32).mean().item()

			self.logger.info('Mean Damerau-Levenshtein distance b/w prediction and target: {}'.format(dist))


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('checkpoint', type=str, help='Path to the checkpoint of the trained model.')
	# parser.add_argument('--log_path', type=str, default=None, help='Path to the .log file where results are logged.')
	parser.add_argument('--device', type=str, default='cpu', help='Device.')
	args = parser.parse_args()

	logger = get_logger()#args.log_path)

	tester = Tester(logger, args.checkpoint, device=args.device)
	tester()