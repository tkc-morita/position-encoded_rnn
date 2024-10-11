# coding: utf-8

import torch.nn as nn
from .rnn import RNN
from .s4 import S4D

class SSM(RNN):
	def __init__(self, vocab_size, hidden_size, num_layers=1, dropout=0.0, **kwargs):
		super().__init__(vocab_size, hidden_size, 'RNN', **kwargs)
		input_size = self.rnn.input_size
		self.ssm = nn.Sequential(*[S4D(hidden_size, dropout=dropout, transposed=True)
									for l in range(num_layers)])
		self.in_proj = nn.Linear(input_size, hidden_size)
		del self.rnn
		# self.rnn = self._ssm # NOTE: Replacing an attribute for a deleted module with a method raises some device mismatch errors with nn.DataParallel.

	def _rnn(self, input, time_mask):
		"""
		x: batch_size x length x input_size
		"""
		input = self._encode_time(input, time_mask)
		input = self.in_proj(input)
		out = self.ssm(input.transpose(1,2))[0].transpose(1,2) # NOTE: Like CNN, S4D operates in BxDxL format.
		return out#,None # NOTE: Dummy 2nd output filling the slot for RNN's hidden.