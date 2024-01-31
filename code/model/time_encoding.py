# coding: utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionEncoder(nn.Module):
	def __init__(self, embed_size):
		super().__init__()
		frequencies = torch.exp(torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size))
		self.register_buffer('frequencies', frequencies)

	def forward(self, reference):
		"""
		reference: batch_size x max_length x embed_size
		"""
		position = torch.arange(reference.size(1), device=reference.device)
		angle = position[None,:,None] * self.frequencies[None,None,:]
		encoded = torch.cat([angle.sin(), angle.cos()], dim=-1)
		encoded = encoded / math.sqrt(angle.size(-1)) # NOTE: L2-norm = 1.0
		return encoded
