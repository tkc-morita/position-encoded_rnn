# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from .rnn import RNN

class VocabExtender(nn.Module):
	def __init__(self, base: RNN, extra_vocab_size: int, **kwargs):
		super().__init__()
		self.base = base
		for p in base.parameters(): # Freeze the base model.
			p.requires_grad = False
		self.extra_embedding = nn.Embedding(extra_vocab_size, self.base.embedding.embedding_dim)
		self.to_extra_logits = nn.Linear(self.base.to_logits.in_features, extra_vocab_size)
		self.extra_vocab_size = extra_vocab_size

	def forward(self, input: torch.Tensor, **kwargs):
		# NOTE: extra vocab is PREPENDED to the original.
		in_extra_vocab = input<self.extra_vocab_size
		input = torch.where(in_extra_vocab.unsqueeze(-1),
					self.extra_embedding(input.clamp_max(self.extra_vocab_size-1)),
					self.base.embedding((input-self.extra_vocab_size).clamp_min(0)),
					)
		features = self.base(input, return_feature=True, **kwargs) # NOTE: base model skips embedding when input is float-valued.
		base_logits = self.base.to_logits(features)
		extra_logits = self.to_extra_logits(features)
		return torch.cat([extra_logits,base_logits], dim=-1)

