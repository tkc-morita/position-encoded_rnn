# coding: utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEncoder(nn.Module):
	def __init__(self, embed_size):
		super().__init__()
		frequencies = torch.exp(torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size))
		self.register_buffer('frequencies', frequencies)

	def forward(self, reference):
		"""
		reference: batch_size x max_length_in_batch x embed_size
		"""
		position = torch.arange(reference.size(1), device=reference.device)
		angle = position[None,:,None] * self.frequencies[None,None,:]
		encoded = torch.cat([angle.sin(), angle.cos()], dim=-1)
		encoded = encoded / math.sqrt(angle.size(-1)) # NOTE: L2-norm = 1.0
		return encoded

class LearnablePositionEncoder(nn.Module):
	def __init__(self, embed_size, max_length):
		super().__init__()
		embeddings = torch.randn((max_length,embed_size))
		self.register_parameter('embeddings', nn.Parameter(embeddings, requires_grad=True))

	def forward(self, reference):
		"""
		reference: batch_size x max_length_in_batch x embed_size
		"""
		B,L,_ = reference.size()
		return self.embeddings[:L,:].unsqueeze(0).expand_as(reference)
	

class RandomPositionalEncoder(LearnablePositionEncoder):
	def __init__(self, embed_size, max_length):
		super(LearnablePositionEncoder, self).__init__()
		embeddings = torch.randn((max_length,embed_size))
		embeddings = F.normalize(embeddings, p=2.0, dim=-1)
		self.register_buffer('embeddings', embeddings)

class DummyPositionEncoder(nn.Module):
	"""
	Pseudo positional encoding duplicating the input embeddings.
	"""
	def forward(self, reference):
		"""
		reference: batch_size x max_length_in_batch x embed_size
		"""
		return reference