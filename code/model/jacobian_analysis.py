# coding: utf-8

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch import _VF
from torch.func import vmap,jacrev
from torch.distributions import Normal
from .rnn import RNN

class JacobianAnalyzer(nn.Module):
	def __init__(self, model: RNN):
		super().__init__()
		self.model = model

	def forward(self, input, input_time=0, max_chunk_size=128):
		# NOTE: 2nd dim of input bundles the triplet (cat(x,s),cat(x,s'),cat(x',s)),
		# i.e., (base, initial-shared, suffix-shared).
		B,_,L = input.size()
		input = input.view(-1,L)
		with torch.no_grad():
			input = self._format_input(input)
		prefix = input[:,:input_time+1,:]
		suffix = input[:,input_time+1:,:]
		with torch.no_grad():
			_,hidden = self.model.rnn(prefix)
		if isinstance(hidden, tuple):
			num_layers,Bx3,D = hidden[0].size()
			hidden_chunks = len(hidden)
			hidden = torch.cat(hidden,dim=0) # 3*#layers x batch_size x hidden_dim
		else:
			num_layers,Bx3,D = hidden.size()
			hidden_chunks = None
		hidden = hidden.transpose(0,1).reshape(Bx3,-1) # B x others
		def run_on_suffix(suffix_unbatched,h_unbatched):
			"""
			suffix_unbatched: length x hidden_size
			h_unbatched: #layers x others
			"""
			h_unbatched = h_unbatched.view(1,-1,D).unbind(dim=-2) if hidden_chunks is None \
				else [h_l.unbind(dim=1) for h_l in h_unbatched.view(1,hidden_chunks,num_layers,D).unbind(dim=-2)]
			for x_t in suffix_unbatched.unsqueeze(0).unbind(dim=-2):
				new_h = list()
				for l,h_l in enumerate(h_unbatched):
					W_ih,W_hh,b_ih,b_hh = self._get_weight_and_bias(l)
					new_h_l = getattr(self, self.model.rnn.mode.lower()+'_cell')(
										x_t, h_l, W_ih, W_hh, b_ih, b_hh)
					x_t = new_h_l[0] if isinstance(new_h_l, tuple) else new_h_l
					new_h.append(new_h_l)
				h_unbatched = new_h
			x_t = x_t.squeeze(0)
			return x_t,x_t
		chunk_size = min(max_chunk_size,D)
		J,output = vmap(jacrev(run_on_suffix, argnums=1, chunk_size=chunk_size, has_aux=True))(suffix,hidden)
		output = output.detach().view(B,3,-1)
		J = J.detach().view(B,3,*J.size()[1:])
		# Get useful stats.
		# if hidden_chunks is None:
		stats = self._get_jacobian_stats(J)
		forward_dist = torch.linalg.norm(output[:,0,None,:]-output[:,1:,:], ord=2, dim=-1)
		output_normalized = F.normalize(output, p=2.0, dim=-1)
		forward_similarity = (output_normalized[:,0,None,:]*output_normalized[:,1:,:]).sum(dim=-1)
		stats['forward_dist_common_target'] = forward_dist[:,0]
		stats['forward_dist_common_suffix'] = forward_dist[:,1]
		stats['forward_similarity_common_target'] = forward_similarity[:,0]
		stats['forward_similarity_common_suffix'] = forward_similarity[:,1]
		# else:
			# stats = [self._get_jacobian_stats(J_h)
							# for J_h in J.view(B,num_layers*D,hidden_chunks,num_layers*D
											# ).unbind(dim=-2)]
		return stats
	
	def _get_jacobian_stats(self, J, hard_rank_atol=0.0, hard_rank_rtol=None, soft_rank_std=None):
		# NOTE: Non-pairwise stats are evaluated on the 1st (0th) entry.
		frobenius_norm = torch.linalg.norm(J[:,0,...], ord='fro', dim=(-2,-1))
		frobenius_dist = torch.linalg.norm(J[:,0,None,...]-J[:,1:,...], ord='fro', dim=(-2,-1))
		grad_norm = torch.linalg.norm(J, ord=2, dim=-1)
		J_norm = J / grad_norm.masked_fill(grad_norm==0.0, 1.0).unsqueeze(-1)
		# J_norm = F.normalize(J, p=2.0, dim=-1) # normalize over input dim.
		gradient_similarity = (J_norm[:,0,None,...]*J_norm[:,1:,...]).sum(dim=-1)
		min_gradient_similarity = gradient_similarity.amin(dim=-1)
		max_gradient_similarity = gradient_similarity.amax(dim=-1)
		mean_gradient_similarity = gradient_similarity.mean(dim=-1)
		weight = grad_norm[:,0,None,...]*grad_norm[:,1:,...]
		weight_normalizer = weight.sum(dim=-1, keepdim=True)
		weight = weight / weight_normalizer.masked_fill(weight_normalizer==0.0, 1.0)
		weighted_mean_gradient_similarity = (gradient_similarity*weight).sum(dim=-1)
		# slogdet = torch.linalg.slogdet(J)
		# U,singulars,V_trans = torch.linalg.svdvals(J, full_matrices=False,
								# driver='gesvd' if J.device.type=='cuda' else None)
		singulars = torch.linalg.svdvals(J, driver='gesvd' if J.device.type=='cuda' else None)
		if hard_rank_rtol is None:
			hard_rank_rtol = max(J.size(-2),J.size(-1))*torch.finfo(J.dtype).eps
		rank_thresh = max(hard_rank_atol, hard_rank_rtol)
		hard_rank_bundled = (singulars>rank_thresh).long().sum(dim=-1)
		if soft_rank_std is None:
			# NOTE: Set soft_rank_std s.t. singular values at rank_thresh
			#       yield the discount of 0.5
			soft_rank_std = (rank_thresh/math.sqrt(-2.0*math.log(0.5)))
		soft_rank = singulars.size(-1)-(-0.5*(singulars[:,0,...]/soft_rank_std).pow(2)).exp().sum(dim=-1) # NOTE: Smoothing by Gaussian PDF (Wang+18, IJMLC, https://link.springer.com/article/10.1007/s13042-017-0665-9)
		# Computes the similarity b/w the spans of the Jacobians.
		Q,R = torch.linalg.qr(J, mode='reduced') # NOTE: Q is of size J.size(-2) x min(J.size(-2),J.size(-1))
		Q = Q.masked_fill(hard_rank_bundled[...,None,None]
							<=torch.arange(Q.size(-1), device=Q.device).view(1,1,1,-1),
							0.0) # Leave only the basis.
		Q_1 = Q[:,0,None,:,:hard_rank_bundled[:,0].amax().clamp_min(1).item()]
		Q_2 = Q[:,1:,:,:hard_rank_bundled[:,1:].amax().clamp_min(1).item()]
		Q_1_T_x_Q_2 = Q_1.transpose(-2,-1)@Q_2
		try:
			subspace_similarity = torch.linalg.svdvals(Q_1_T_x_Q_2, driver='gesvd'
														if J.device.type=='cuda' else None)
			max_subspace_similarity = subspace_similarity.amax(dim=-1)
			mean_subspace_similarity = subspace_similarity.sum(dim=-1) \
									/ torch.minimum(hard_rank_bundled[:,0,None],hard_rank_bundled[:,1:]
										).clamp_min(1)
		except:
			# NOTE: Addition of small turbulence is reported to be effective. https://github.com/pytorch/pytorch/issues/28293
			# Q_1_T_x_Q_2 = Q_1_T_x_Q_2 + 1e-4*Q_1_T_x_Q_2.mean(dim=(-2,-1), keepdim=True)*torch.rand_like(Q_1_T_x_Q_2)
			# subspace_similarity = torch.linalg.svdvals(Q_1_T_x_Q_2, driver='gesvd'
														# if J.device.type=='cuda' else None)
			max_subspace_similarity = mean_subspace_similarity = torch.full_like(frobenius_dist, torch.nan)
		return dict(frobenius_norm=frobenius_norm,
					frobenius_dist_common_target=frobenius_dist[:,0],
					frobenius_dist_common_suffix=frobenius_dist[:,1],
					min_gradient_similarity_common_target=min_gradient_similarity[:,0],
					min_gradient_similarity_common_suffix=min_gradient_similarity[:,1],
					max_gradient_similarity_common_target=max_gradient_similarity[:,0],
					max_gradient_similarity_common_suffix=max_gradient_similarity[:,1],
					mean_gradient_similarity_common_target=mean_gradient_similarity[:,0],
					mean_gradient_similarity_common_suffix=mean_gradient_similarity[:,1],
					weighted_mean_gradient_similarity_common_target=weighted_mean_gradient_similarity[:,0],
					weighted_mean_gradient_similarity_common_suffix=weighted_mean_gradient_similarity[:,1],
					hard_rank=hard_rank_bundled[:,0],
					soft_rank=soft_rank,
					max_subspace_similarity_common_target=max_subspace_similarity[:,0],
					max_subspace_similarity_common_suffix=max_subspace_similarity[:,1],
					mean_subspace_similarity_common_target=mean_subspace_similarity[:,0],
					mean_subspace_similarity_common_suffix=mean_subspace_similarity[:,1],
					)
	
	def _format_input(self, input):
		input = self.model.embedding(input)
		input = self.model._encode_time(input, time_mask=None)
		return input

	def _get_weight_and_bias(self, layer_ix):
		W_ih = getattr(self.model.rnn, 'weight_ih_l{}'.format(layer_ix))
		W_hh = getattr(self.model.rnn, 'weight_hh_l{}'.format(layer_ix))
		b_ih = getattr(self.model.rnn, 'bias_ih_l{}'.format(layer_ix))
		b_hh = getattr(self.model.rnn, 'bias_hh_l{}'.format(layer_ix))
		return W_ih,W_hh,b_ih,b_hh
	

	def rnn_tanh_cell(self, *args, **kwargs):
		return self._rnn_cell(*args, nonlinearity='tanh', **kwargs)
	
	def rnn_relu_cell(self, *args, **kwargs):
		return self._rnn_cell(*args, nonlinearity='relu', **kwargs)

	@staticmethod
	def _rnn_cell(x_t, h_l, W_ih, W_hh, b_ih, b_hh, nonlinearity):
		# Forward
		y = F.linear(x_t, W_ih, b_ih) + F.linear(h_l, W_hh, b_hh)
		out = getattr(F, nonlinearity)(y)
		return out

	@staticmethod
	def gru_cell(x_t, h_l, W_ih, W_hh, b_ih, b_hh):
		# Forward
		y_ih = F.linear(x_t, W_ih, b_ih)
		y_hh = F.linear(h_l, W_hh, b_hh)
		y = y_ih + y_hh
		y_reset,y_update,_ = y.chunk(3, dim=-1)
		y_ih_new = y_ih.chunk(3,dim=-1)[-1]
		y_hh_new = y_hh.chunk(3,dim=-1)[-1]
		r = y_reset.sigmoid()
		n = (y_ih_new + r*y_hh_new).tanh()
		z = y_update.sigmoid()
		out = (1.0-z)*n + z*h_l
		return out

	@staticmethod
	def lstm_cell(x_t, h_l, W_ih, W_hh, b_ih, b_hh):
		h_l,c_l = h_l
		# Forward
		y = F.linear(x_t, W_ih, b_ih) + F.linear(h_l, W_hh, b_hh)
		y_in,y_forget,y_cell,y_out = y.chunk(4, dim=-1)
		f = y_forget.sigmoid()
		g = y_cell.tanh()
		i = y_in.sigmoid()
		c = f*c_l+i*g
		o = y_out.sigmoid()
		out = o*c.tanh()
		return out,c