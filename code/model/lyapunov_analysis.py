# coding: utf-8

import torch
import torch.nn as nn

def register_grad_storing_hook(module: nn.Module):
	name2norm_list = dict()
	rnn_name = module.rnn.__class__.__name__
	for name,p in module.named_parameters():
		norm_storage = _StoreWeightNorm(name, rnn_name, name2norm_list)
		p.register_hook(norm_storage)
	def initialize_records(module, args):
		for name in name2norm_list.keys():
			if name2norm_list[name][-1]:
				name2norm_list[name].append([])
	module.register_forward_pre_hook(initialize_records)
	return name2norm_list

class _StoreWeightNorm(object):
	def __init__(self, name, rnn_name, name2norm_list):
		self.name = name
		self.name2norm_list = name2norm_list
		if 'rnn' in name and rnn_name!='RNN':
			if rnn_name=='LSTM':
				self.suffices = ['|input_gate','|forget_gate','|cell_gate','|output_gate']
			elif rnn_name=='GRU':
				self.suffices = ['|reset_gate','|update_gate','|new_gate']
		else:
			self.suffices = ['']
		for suffix in self.suffices:
			self.name2norm_list[name+suffix] = [[]]

	def __call__(self, grad: torch.Tensor):
		for suffix,g in zip(self.suffices, grad.chunk(chunks=len(self.suffices), dim=0)):
			norm = torch.linalg.norm(g.flatten(), ord=2.0, dim=0)
			self.name2norm_list[self.name+suffix][-1].append(norm.item())

	def _store_grad_norm(self, grad: torch.Tensor, name: str):
		norm = torch.linalg.norm(grad.flatten(), ord=2.0, dim=0)
		self.name2norm_list[name][-1].append(norm.item())
