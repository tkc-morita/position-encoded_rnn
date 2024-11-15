# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from .time_encoding import SinusoidalPositionEncoder,LearnablePositionEncoder,RandomPositionalEncoder,DummyPositionEncoder

class RNN(nn.Module):
	def __init__(self, vocab_size, hidden_size, rnn_name,
					embed_size=None, time_encoding=None,
					time_encoding_form='sinusoidal', max_length=None,
					learnable_padding_token=False,
					no_padding_token=False, **rnn_kwargs):
		super().__init__()
		if embed_size is None:
			embed_size = hidden_size
		self.embedding = nn.Embedding(vocab_size+int(not no_padding_token), embed_size,
										padding_idx=None if (learnable_padding_token or no_padding_token)
													else vocab_size)
		self.time_encoding = time_encoding
		if not time_encoding is None:
			assert time_encoding in ['add', 'concat'], 'time_encoding must be either "add" or "concat"'
			if time_encoding_form=='sinusoidal':
				self.time_encoder = SinusoidalPositionEncoder(embed_size)
			elif time_encoding_form=='dummy':
				self.time_encoder = DummyPositionEncoder()
			else:
				assert not max_length is None, 'max_length must be specified.'
				if time_encoding_form=='learnable':
					self.time_encoder = LearnablePositionEncoder(embed_size, max_length)
				elif time_encoding_form=='random':
					self.time_encoder = RandomPositionalEncoder(embed_size, max_length)
				else:
					raise ValueError('time_encoding_form must be "sinusoidal", "learnable", or "random".')
		self.rnn = getattr(nn, rnn_name)(embed_size*2 if time_encoding=='concat' else embed_size,
										hidden_size, batch_first=True, bidirectional=False,
										**rnn_kwargs)
		self.to_logits = nn.Linear(hidden_size, vocab_size)
	
	def forward(self, input, time_mask=None, return_feature=False, target=None):
		"""
		input: batch_size x length (x embed_size if already embedded)
		time_mask: batch_size x length x (1)
		"""
		if not input.is_floating_point(): # NOTE: Skip embedding if aready float
			input = self.embedding(input)
		output = self._rnn(input, time_mask)
		if return_feature:
			return output
		output = self.to_logits(output)
		if target is None:
			return output
		# NOTE: Efficiently compute loss on multiple GPUs for big vocab.
		vocab_size = output.size(-1)
		mask = target>=vocab_size # Mask out-of-vocab items.
		target = target.masked_fill(mask, 0) # Dummy target
		loss = F.cross_entropy(output.view(-1,vocab_size), target.view(-1), reduction='none')
		loss = loss.masked_select(~mask.view(-1)).sum()
		normalizer = mask.logical_not().sum() # NOTE: Different # of non-unk tokens on different GPUs.
		return loss.view(1),normalizer.view(1) # NOTE: Work around PyTorch's warning on scalar output.
	
	def _rnn(self, input, time_mask):
		input = self._encode_time(input, time_mask)
		output,_ = self.rnn(input)
		return output
	
	def _encode_time(self, input, time_mask):
		if self.time_encoding is None:
			return input
		time_encoded = self.time_encoder(input)
		if not time_mask is None:
			if time_mask.ndim==2:
				time_mask = time_mask.unsqueeze(-1)
			time_encoded = time_encoded.masked_fill(time_mask, 0.0)
		if self.time_encoding=='add':
			input = input + time_encoded
		elif self.time_encoding=='concat':
			input = torch.cat([input, time_encoded.expand(input.size(0),-1,-1)], dim=-1)
		return input

	def get_padding_embedding(self):
		return self.embedding.weight[-1,:]

class DifferentOutputVocabulary(RNN):
	def __init__(self, vocab_size, output_size, *args, **kwargs):
		super().__init__(vocab_size, *args, **kwargs)
		self.to_logits = nn.Linear(self.to_logits.in_features, output_size)

# class DualInputRNN(RNN):
# 	def __init__(self, vocab_size, vocab_size2, *args, **kwargs):
# 		super().__init__(vocab_size, *args, **kwargs)
# 		embed_size = self.embedding.embedding_dim
# 		self.embedding2 = nn.Embedding(vocab_size2+(self.embedding.num_embeddings-vocab_size),
# 										embed_size, padding_idx=None)
# 		self.rnn = self.rnn.__class__(self.rnn.input_size+embed_size, self.rnn.hidden_size, batch_first=True, bidirectional=False,
# 										num_layers=self.rnn.num_layers, dropout=self.rnn.dropout)
# 		if self.time_encoding=='add':
# 			time_encoder_args = [embed_size*2]
# 			if hasattr(self.time_encoder, 'embeddings'):
# 				time_encoder_args.append(self.time_encoder.embeddings.size(0))
# 			self.time_encoder = self.time_encoder.__class__(*time_encoder_args)

# 	def forward(self, input1, input2, time_mask=None, return_feature=False):
# 		input1 = self.embedding(input1)
# 		input2 = self.embedding2(input2)
# 		output = self._rnn(torch.cat([input1,input2], dim=-1), time_mask)
# 		if return_feature:
# 			return output
# 		output = self.to_logits(output)
# 		return output

class RNN_w_MultiplePadding(RNN):
	def __init__(self, vocab_size, *args, num_paddings=1, **kwargs):
		super().__init__(vocab_size, *args, **kwargs)
		self.embedding = nn.Embedding(vocab_size+num_paddings, self.embedding.embedding_dim)

class WeightSharedRNN(RNN):
	def __init__(self, *args, embed_size=None, **kwargs):
		super().__init__(*args, embed_size=None, **kwargs) # NOTE: Force embed_size==hidden_size
		self.to_logits = WeightSharedVocabulary(self.to_logits.out_features,
												self.to_logits.in_features,
												learnable_padding_token=self.embedding.padding_idx is None)
		self.embedding = self.to_logits


class WeightSharedVocabulary(nn.Module):
	def __init__(self, vocab_size, hidden_size, learnable_padding_token=False):
		super().__init__()
		weight = torch.randn(vocab_size+1, hidden_size)
		self.register_parameter('weight_raw', nn.Parameter(weight, requires_grad=True))
		self.register_parameter('log_scale', nn.Parameter(torch.tensor(0.0), requires_grad=True))
		self.register_forward_pre_hook(self.normalize_weight)
		self.padding_idx = None if learnable_padding_token else vocab_size

	@staticmethod
	def normalize_weight(module, inputs):
		module.weight = F.normalize(module.weight_raw, p=2.0, dim=-1)

	def forward(self, input):
		if input.is_floating_point(): # -> hidden-to-category
			input = F.normalize(input, p=2.0, dim=-1)
			out = F.linear(input, self.weight[:-1,:]) * self.log_scale.exp()
		else:
			out = F.embedding(input, self.weight, padding_idx=self.padding_idx)
		return out
	

class RNN_w_ContinuousInput(RNN):
	def __init__(self, input_size, hidden_size, rnn_name,
					embed_size=None, time_encoding=None,
					time_encoding_form='sinusoidal', max_length=None,
					**rnn_kwargs):
		nn.Module.__init__(self)
		if embed_size is None:
			embed_size = hidden_size
		self.in_proj = nn.Linear(input_size, embed_size)
		self.time_encoding = time_encoding
		if not time_encoding is None:
			assert time_encoding in ['add', 'concat'], 'time_encoding must be either "add" or "concat"'
			if time_encoding_form=='learnable':
				self.time_encoder = LearnablePositionEncoder(embed_size, max_length)
			elif time_encoding_form=='random':
				self.time_encoder = RandomPositionalEncoder(embed_size, max_length)
			else:
				raise ValueError('time_encoding_form must be "sinusoidal", "learnable", or "random".')
		self.rnn = getattr(nn, rnn_name)(embed_size*2 if time_encoding=='concat' else embed_size,
										hidden_size, batch_first=True, bidirectional=False,
										**rnn_kwargs)
		self.out_proj = nn.Linear(hidden_size, input_size)
	
	def forward(self, input, time_mask=None):
		"""
		input: batch_size x length (x embed_size if already embedded)
		time_mask: batch_size x length x (1)
		"""
		input = self.in_proj(input)
		output = self._rnn(input, time_mask)
		output = self.out_proj(output)
		return output
	
class RNN_AR(RNN):
	def __init__(self, *args, **kwargs):
		kwargs['time_encoding'] = 'concat' # NOTE: Used for phase encoding.
		super().__init__(*args, **kwargs)
		del self.time_encoding
		self.phase_embedding = nn.Embedding(2, self.embedding.embedding_dim)

	def forward(self, input, ar_input=None):
		phase = torch.ones_like(input)
		if not ar_input is None: # i.e. training
			phase = torch.cat([phase, torch.zeros_like(ar_input)], dim=1)
			input = torch.cat([input, ar_input], dim=1)
		phase = self.phase_embedding(phase)
		input = self.embedding(input)
		input = torch.cat([input,phase], dim=-1)
		output,hidden = self.rnn(input)
		output = self.to_logits(output)
		if not ar_input is None:
			return output
		# Autoregression
		phase = torch.zeros((phase.size(0),1), dtype=int, device=phase.device)
		input_t = torch.full_like(phase, fill_value=self.to_logits.out_features)
		phase = self.phase_embedding(phase)
		output = list(output.unbind(dim=1))
		for t in range(input.size(1)):
			input_t = torch.cat([self.embedding(input_t),phase], dim=-1)
			output_t,hidden = self.rnn(input_t, hidden)
			output_t = self.to_logits(output_t)
			output.append(output_t.squeeze(dim=1))
			input_t = output_t.argmax(dim=-1)
		return torch.stack(output, dim=1)
	
class RNN_LM(RNN):
	"""
	Autoregressive language modeling.
	"""
	def forward(self, input, num_steps=None):
		if num_steps is None:
			return super().forward(input)
		# NOTE: input is assumed to be a prefix.
		input = self.embedding(input)
		input = self._encode_time(input, None)
		output,hidden = self.rnn(input)
		output = self.to_logits(output)
		input_t = output[:,-1,None,:].argmax(dim=-1)
		dummy_input = torch.zeros((input.size(0),input.size(1)+num_steps), device=input.device)
		time_encoded = None if self.time_encoding is None \
						else self.time_encoder(dummy_input)[:,input.size(1):,:] # Strip off the prefix
		output = list(output.unbind(dim=1))
		for t in range(num_steps):
			input_t = self.embedding(input_t)
			if self.time_encoding=='add':
				input_t = input_t + time_encoded[:,t,None,:]
			elif self.time_encoding=='concat':
				input_t = torch.cat([input_t,time_encoded[:,t,None,:].expand_as(input_t)], dim=-1)
			output_t,hidden = self.rnn(input_t, hidden)
			output_t = self.to_logits(output_t)
			output.append(output_t.squeeze(dim=1))
			input_t = output_t.argmax(dim=-1)
		return torch.stack(output, dim=1)
	