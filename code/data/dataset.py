# coding: utf-8

import torch
import torch.nn.functional as F

class RandomSequence(object):
	collate_fn = None
	def __init__(self, vocab_size, length, num_held_out=0):
		self.vocab_size = vocab_size
		self.length = length
		if num_held_out:
			possible_patterns = vocab_size**length
			assert possible_patterns>num_held_out, 'Cannot hold out {num_held_out} sequences from {possible_patterns} patterns.'.format(num_held_out=num_held_out, possible_patterns=possible_patterns)
			self.held_out = torch.randint(self.vocab_size, size=(1, length))
			while self.held_out.size(0)<num_held_out:
				candidate = torch.randint(self.vocab_size, size=(1, length))
				if (candidate!=self.held_out).any(dim=-1).all(dim=0).item(): # check duplication
					self.held_out = torch.cat([self.held_out,candidate], dim=0)
			# self.held_out = torch.randint(self.vocab_size, size=(num_held_out, self.length))
		else:
			self.held_out = None

	def __getitem__(self, ix):
		while True: # Rejection sampling
			sequence = torch.randint(self.vocab_size, size=(self.length,))
			if self.held_out is None or (sequence!=self.held_out).any(dim=-1).all(dim=0).item():
				break
		return sequence

	def __len__(self):
		return 512 # dummy length

class VariableLengthSequence(object):
	collate_fn = None
	def __init__(self, vocab_size, max_length, num_held_out=0, min_length=1):
		self.vocab_size = vocab_size
		self.max_length = max_length
		self.min_length = min_length
		if num_held_out:
			self.held_out = dict()
			for length in range(min_length,max_length+1):
				possible_patterns = vocab_size**length
				assert possible_patterns>num_held_out, 'Cannot hold out {num_held_out} sequences from {possible_patterns} patterns.'.format(num_held_out=num_held_out, possible_patterns=possible_patterns)
				self.held_out[length] = torch.randint(self.vocab_size, size=(1, length))
				while self.held_out[length].size(0)<num_held_out:
					candidate = torch.randint(self.vocab_size, size=(1, length))
					if (candidate!=self.held_out[length]).any(dim=-1).all(dim=0).item(): # check duplication
						self.held_out[length] = torch.cat([self.held_out[length],candidate], dim=0)
		else:
			self.held_out = None

	def __getitem__(self, ix):
		while True: # Rejection sampling
			length = torch.randint(self.min_length,self.max_length+1,size=(1,)).item()
			sequence = torch.randint(self.vocab_size, size=(length,))
			if self.held_out is None or \
				(not length in self.held_out) \
				or (sequence!=self.held_out[length]).any(dim=-1).all(dim=0).item():
				break
		return sequence

	def __len__(self):
		return 512 # dummy length
	
	@staticmethod
	def collate_fn(batch):
		return batch # NOTE: Return as a list
	

class TransitivityTest(object):
	"""
	Each sequence is sampled from either the first or last three quarter of the vocab.
	e.g., {0,...,7} -> {0,1,2,3,4,5} or {2,3,4,5,6,7}
	Then, the test sequence is sampled from the disjoint portion {0,1,6,7}:
	"""
	collate_fn = None
	def __init__(self, vocab_size, length, num_held_out=0):
		assert vocab_size>=4, 'vocab_size must be 4 or greater.'
		assert (vocab_size % 4)==0, 'vocab_size must be divisible by 4.'
		self.vocab_size = vocab_size
		self.length = length
		if num_held_out:
			possible_patterns = (vocab_size//2)**length
			assert possible_patterns>num_held_out, 'Cannot hold out {num_held_out} sequences from {possible_patterns} patterns.'.format(num_held_out=num_held_out, possible_patterns=possible_patterns)
			self.held_out = torch.tensor([], dtype=torch.long).view(0,length)
			while self.held_out.size(0)<num_held_out:
				candidate = torch.randint(vocab_size//4, size=(1,length))
				in_smaller_partition = torch.randint_like(candidate,2)
				candidate = candidate + (vocab_size*3//4)*in_smaller_partition # Random onset.
				in_smaller_partition = in_smaller_partition.bool()
				if ((candidate!=self.held_out).any(dim=-1).all(dim=0).item() # check duplication
					and in_smaller_partition.any().item() # Filter out all in partition=0
					and in_smaller_partition.logical_not().any().item()): # Filter out all in partition=1
					self.held_out = torch.cat([self.held_out,candidate], dim=0)
		else:
			self.held_out = None

	def __getitem__(self, ix):
		return torch.randint(self.vocab_size*3//4, size=(self.length,)) \
				+ (self.vocab_size//4)*torch.randint(2, size=(1,)) # vocab onset

	def __len__(self):
		return 512 # dummy length

class RandomSphere(object):
	collate_fn = None
	def __init__(self, dimensionality, length, num_test_seqs=0):
		self.dimensionality = dimensionality
		self.length = length
		if num_test_seqs:
			self.held_out = torch.randn(size=(num_test_seqs,length,dimensionality))
			self.held_out = F.normalize(self.held_out, p=2.0, dim=-1)
		else:
			self.held_out = None

	def __getitem__(self, ix):
		sequence = torch.randn(size=(self.length,self.dimensionality))
		sequence = F.normalize(sequence, p=2.0, dim=-1)
		return sequence

	def __len__(self):
		return 512 # dummy length