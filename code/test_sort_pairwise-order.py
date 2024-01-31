# coding: utf-8

import os,argparse
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchaudio.functional import edit_distance
from utils.logging import get_logger
from data.dataset import RandomSequence
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
			sorted = sequence.sort(dim=1)[0]
			vocab_size = self.checkpoint['modules']['rnn']['init_args']['vocab_size']
			dummy_input = torch.full_like(sequence, vocab_size) # NOTE: Mapped to a fixed zero vector or learnable filler.
			input = torch.cat([sequence, dummy_input], dim=1)

			time_mask = False
			if self.checkpoint.get('mask_encode_time', False):
				time_mask = time_mask | torch.cat([torch.ones_like(sequence, dtype=bool),
												torch.zeros_like(dummy_input, dtype=bool),
												], dim=1)
			if self.checkpoint.get('mask_decode_time', False):
				time_mask = time_mask | torch.cat([torch.zeros_like(sequence, dtype=bool),
												torch.ones_like(dummy_input, dtype=bool),
												], dim=1)
			if isinstance(time_mask, bool):
				time_mask = None

			logits = self.rnn(input, time_mask=time_mask)
			logits = logits[:,sequence.size(1):,:] # Strip-off the encoding phase.

			prediction = logits.argmax(dim=-1) # batch_size x duration
			# pred_freq = F.one_hot(prediction, num_classes=vocab_size).float().mean(dim=1)
			# pred_perplex = (-pred_freq * pred_freq.masked_fill(pred_freq==0.0,1).log()).sum(-1).mean(0).exp().item()
			# self.logger.info('Sequence-wise prediction perplexity: {}'.format(pred_perplex))
			mean_variation = torch.tensor([seq.unique().nelement() for seq in prediction.unbind(dim=0)]
								).float().mean().item()
			self.logger.info('Mean type counts per sequence: {}'.format(mean_variation))

			L1_dist = (prediction-sorted).float().abs().mean().item()
			self.logger.info('Mean L1 distance b/w tokens: {}'.format(L1_dist))
			# token_accuracy = (prediction==sorted).float().mean().item()
			# self.logger.info('Test accuracy (token): {}'.format(token_accuracy))

			pairwise_accuracy = (prediction.unsqueeze(-1)<=prediction.unsqueeze(1)).float()
			pairwise_accuracy = pairwise_accuracy.masked_select(
								torch.ones_like(pairwise_accuracy, dtype=torch.bool).triu(diagonal=1)
								).mean()
			self.logger.info('Pairwise prediction order accuracy: {}'.format(pairwise_accuracy))

			# bag_of_words_accuracy = 0.0
			# mean_levenshtein_dist = 0.0
			# for target_seq,pred_seq in zip(sorted.cpu().tolist(),prediction.cpu().tolist()):
			# 	mean_levenshtein_dist += edit_distance(pred_seq,target_seq)
			# 	for target_token in target_seq:
			# 		if target_token in pred_seq:
			# 			bag_of_words_accuracy += 1.0
			# 			pred_seq.remove(target_token) # NOTE: Only remove 1st occurrence.
			# bag_of_words_accuracy /= sorted.nelement()
			# self.logger.info('Bag-of-Words accuracy: {}'.format(bag_of_words_accuracy))
			# mean_levenshtein_dist /= sorted.size(0)
			# self.logger.info('Mean Levenshtein distance: {}'.format(mean_levenshtein_dist))


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('checkpoint', type=str, help='Path to the checkpoint of the trained model.')
	parser.add_argument('--log_path', type=str, default=None, help='Path to the .log file where results are logged.')
	parser.add_argument('--device', type=str, default='cpu', help='Device.')
	args = parser.parse_args()

	if not args.log_path is None:
		os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
	logger = get_logger(args.log_path)

	tester = Tester(logger, args.checkpoint, device=args.device)
	tester()