# coding: utf-8

import torch
import torch.nn.functional as F

class _Base(object):
	def __init__(self, dummy_datasize=512):
		self.dummy_datasize = dummy_datasize
	
	def __len__(self):
		return self.dummy_datasize

class RandomSequence(_Base):
	collate_fn = None
	def __init__(self, vocab_size, length, num_held_out=0, **kwargs):
		super().__init__(**kwargs)
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

class VariableLengthSequence(_Base):
	collate_fn = None
	def __init__(self, vocab_size, max_length, num_held_out=0, min_length=1, **kwargs):
		super().__init__(**kwargs)
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
	
	@staticmethod
	def collate_fn(batch):
		return batch # NOTE: Return as a list

class FreqentVSRare(_Base):
	collate_fn = None
	def __init__(self, vocab_size, length, num_held_out=0, rarity=1/8, # <- Changed the default rarity to match the experiments (10/01/2024)
					num_frequent=None, mixed_held_out=False, **kwargs):
		super().__init__(**kwargs)
		self.vocab_size = vocab_size
		self.length = length
		self.rarity = rarity
		if num_frequent is None:
			num_frequent = vocab_size//2
			self.num_rare = None # NOTE: Use this None for backward-compatibility w/ previous implementations.
		else:
			self.num_rare = vocab_size-num_frequent
		self.num_frequent = num_frequent
		if num_held_out:
			def _hold_out(sub_vocab_size, num_sub_held_out):
				held_out = torch.randint(sub_vocab_size, size=(1, length))
				while held_out.size(0)<num_sub_held_out:
					candidate = torch.randint(sub_vocab_size, size=(1, length))
					if (candidate!=held_out).any(dim=-1).all(dim=0).item(): # check duplication
						held_out = torch.cat([held_out,candidate], dim=0)
				return held_out
			if mixed_held_out:
				possible_patterns = vocab_size**length
				assert possible_patterns>num_held_out, 'Cannot hold out {num_held_out} sequences from {possible_patterns} patterns.'.format(num_held_out=num_held_out, possible_patterns=possible_patterns)
				self.held_out = _hold_out(vocab_size, num_held_out)
			else:
				possible_patterns = min(num_frequent**length, (vocab_size-num_frequent)**length)
				assert possible_patterns>num_held_out//4, 'Cannot hold out {num_held_out} sequences from {possible_patterns} patterns.'.format(num_held_out=num_held_out, possible_patterns=possible_patterns)
				assert (num_held_out % length)==0, 'num_held_out must be divisible by length.'
				held_out_frequent_frequent = _hold_out(num_frequent, num_held_out//4
												).view(-1,length,length)
				held_out_rare_rare = _hold_out(vocab_size-num_frequent, num_held_out//4
												).view(-1,length,length)+num_frequent
				to_be_evaluated = torch.eye(length, dtype=torch.bool) # L x L
				held_out_frequent_rare = held_out_frequent_frequent.where(
											to_be_evaluated, held_out_rare_rare)
				held_out_rare_frequent = held_out_rare_rare.where(
											to_be_evaluated, held_out_frequent_frequent)
				# held_out_frequent_rare = torch.cat([held_out_frequent_frequent[:,:length//2],
													# held_out_rare_rare[:,length//2:]], dim=-1)
				# held_out_rare_frequent = torch.cat([held_out_rare_rare[:,:length//2],
													# held_out_frequent_frequent[:,length//2:]], dim=-1)
				self.held_out = torch.stack([
									torch.stack([held_out_frequent_frequent,held_out_frequent_rare],dim=0),
									torch.stack([held_out_rare_frequent,held_out_rare_rare],dim=0),
								], dim=0) # prefix_frequency x suffix_frequency x N x seq_length x seq_length
		else:
			self.held_out = None

	def __getitem__(self, ix):
		while True: # Rejection sampling
			sequence = torch.randint(self.num_frequent, size=(self.length,))
			is_rare = torch.rand_like(sequence, dtype=torch.float)<self.rarity
			if self.num_rare is None: # = Even sized
				sequence = sequence+is_rare.long()*self.num_frequent
			else:
				rare_sequence = torch.randint(self.num_rare, size=(self.length,))+self.num_frequent
				sequence = torch.where(is_rare, rare_sequence, sequence)
			if self.held_out is None or (sequence!=self.held_out).any(dim=-1).all().item():
				break
		return sequence

class NonRepeatingRandomSequence(_Base):
	collate_fn = None
	def __init__(self, vocab_size, length, num_held_out=0, **kwargs):
		super().__init__(**kwargs)
		assert vocab_size>=length, 'vocab_size must be at least length.'
		self.vocab_size = vocab_size
		self.length = length
		if num_held_out:
			# NOTE: torch.prod more easily explodes than Python's builtin math.
			possible_patterns = torch.e**torch.arange(1,vocab_size+1)[-self.length:].log().sum().item()
			assert possible_patterns>torch.tensor(num_held_out).log().item(), 'Cannot hold out {num_held_out} sequences from {possible_patterns} patterns.'.format(num_held_out=num_held_out, possible_patterns=possible_patterns)
			self.held_out = torch.randperm(self.vocab_size)[None,:self.length]
			while self.held_out.size(0)<num_held_out:
				candidate = torch.randperm(self.vocab_size)[None,:self.length]
				if (candidate!=self.held_out).any(dim=-1).all(dim=0).item(): # check duplication
					self.held_out = torch.cat([self.held_out,candidate], dim=0)
			# self.held_out = torch.randint(self.vocab_size, size=(num_held_out, self.length))
		else:
			self.held_out = None

	def __getitem__(self, ix):
		while True: # Rejection sampling
			sequence = torch.randperm(self.vocab_size)[:self.length]
			if self.held_out is None or (sequence!=self.held_out).any(dim=-1).all(dim=0).item():
				break
		return sequence
