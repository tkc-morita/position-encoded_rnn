# coding: utf-8

import os,argparse
import pandas as pd
import torch
import torch.nn as nn
from utils.logging import get_logger
from model.jacobian_analysis import JacobianAnalyzerSSM
from test_jacobian_palindrome import Tester as _Tester

class Tester(_Tester):
	def __call__(self, save_path, num_seq_triplets, prefix_length, batch_size=None):
		self.logger.info('Prefix length: {}'.format(prefix_length))
		vocab_size = self.checkpoint['modules']['ssm']['init_args']['vocab_size']
		held_out_data = self.checkpoint['held_out_data']
		seq_length = held_out_data.size(-1)

		# NOTE: Build pairs of sequences headed by the same initial token.
		if held_out_data.ndim>2: # Whether or not vocab is split into frequent vs. rare.
			if self.uneven_vocab_split:
				# NOTE: held_out_data is of size 2 x 2 x #held_out_per_type x L x L
				assert (held_out_data.size(0)==2) \
						and (held_out_data.size(1)==2) \
						and (held_out_data.size(-2)==held_out_data.size(-1)), \
						'held_out_data in invalid size of {}'.format(held_out_data.size())
				assert prefix_length==0, 'non-zero prefix is not supported for unevenly split vocab.'
				# Extract frequent/rare tokens from the disturbants in the test data.
				is_disturbant = torch.eye(held_out_data.size(-1), dtype=torch.bool).logical_not(
							)[None,None,:,:]
				frequent_tokens = held_out_data[:,0,...].masked_select(is_disturbant)
				rare_tokens = held_out_data[:,1,...].masked_select(is_disturbant)
				# held_out_data is of size 2 x 2 x #held_out x length
				# frequent_tokens = torch.cat([held_out_data[0,0,:,:].reshape(-1),
				# 							held_out_data[0,1,:,:seq_length//2].reshape(-1),
				# 							held_out_data[1,0,:,seq_length//2:].reshape(-1)], dim=0)
				# rare_tokens = torch.cat([held_out_data[1,1,:,:].reshape(-1),
				# 							held_out_data[1,0,:,:seq_length//2].reshape(-1),
				# 							held_out_data[0,1,:,seq_length//2:].reshape(-1)], dim=0)
				target_token_ixs = torch.randint(frequent_tokens.nelement(), size=(2,num_seq_triplets//2,2,1))
				target_token = torch.cat([frequent_tokens.take(target_token_ixs[0,...]),
											rare_tokens.take(target_token_ixs[1,...])],
											dim=0)
				suffix_tokens_ixs = torch.randint(frequent_tokens.nelement(),
													size=(2,2,num_seq_triplets//4,2,seq_length-1))
				suffix = torch.cat([frequent_tokens.take(suffix_tokens_ixs[0,0,...]),
											rare_tokens.take(suffix_tokens_ixs[0,1,...]),
											frequent_tokens.take(suffix_tokens_ixs[1,0,...]),
											rare_tokens.take(suffix_tokens_ixs[1,1,...])], dim=0)
			else:
				# NOTE: Assumes even split of the frequent vs. rare tokens
				target_token = torch.randint(vocab_size//2, size=(2,num_seq_triplets//2,2,1))
				target_token[1,...] += vocab_size//2
				target_token = target_token.view(num_seq_triplets,2,1)
				suffix = torch.randint(vocab_size//2, size=(2,2,num_seq_triplets//4,2,seq_length-1-prefix_length))
				suffix[:,1,...] += vocab_size//2
				suffix = suffix.view(num_seq_triplets,2,seq_length-1-prefix_length)
				if prefix_length:
					prefix = torch.randint(vocab_size//2, size=(2,2,num_seq_triplets//4,2,prefix_length))
					prefix[:,1,...] += vocab_size//2
					prefix = prefix.view(num_seq_triplets,2,prefix_length)
			target_token_type = ['frequent']*(num_seq_triplets//2)+['rare']*(num_seq_triplets//2)
			suffix_tokens_type = ['frequent']*(num_seq_triplets//4)\
									+['rare']*(num_seq_triplets//4)\
									+['frequent']*(num_seq_triplets//4)\
									+['rare']*(num_seq_triplets//4)
		else:
			target_token = torch.randint(vocab_size, size=(num_seq_triplets,2,1))
			suffix = torch.randint(vocab_size, size=(num_seq_triplets,2,seq_length-1))
			target_token_type = None
			suffix_tokens_type = None
		target_token = torch.cat([target_token[...,0,None,:],target_token], dim=-2) # (x,x,x')
		suffix = torch.cat([suffix,suffix[...,0,None,:]], dim=-2) # (s,s',s)
		if prefix_length:
			# NOTE: Prefix is shared when target is shared.
			prefix = torch.cat([prefix[...,0,None,:],prefix], dim=-2) # (p,p,p')
			full_data = torch.cat([prefix,target_token,suffix], dim=-1).to(self.device)
			dummy_input = torch.full_like(full_data[...,:-prefix_length], vocab_size) # NOTE: prefix need not be reconstructed.
		else:
			full_data = torch.cat([target_token,suffix], dim=-1).to(self.device)
			dummy_input = torch.full_like(full_data, vocab_size) # NOTE: Mapped to a fixed zero vector or learnable filler.
		full_input = torch.cat([full_data, dummy_input], dim=-1)

		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		dfs_full = []
		dfs_real = []
		dfs_imag = []
		for checkpoint_path in self.checkpoint_paths:
			self.retrieve_model(checkpoint_path=checkpoint_path)
			iteration = self.checkpoint.get('iteration')+1 # NOTE: 1-start counting
			self.logger.info('Test {}th iteration.'.format(iteration))

			if batch_size is None:
				batch_size = num_seq_triplets

			ssm = self.ssm.module if isinstance(self.ssm, nn.DataParallel) else self.rnn
			ssm.eval() # NOTE: Turn off dropout (but not used so far...)
			analyzer = nn.DataParallel(JacobianAnalyzerSSM(ssm).to(self.device))

			# with torch.no_grad():

			for batch_ix,input in enumerate(full_input.split(batch_size, dim=0)):
				stats_per_type = analyzer(input, input_time=prefix_length)
				for stats,dfs in zip(stats_per_type,[dfs_full,dfs_real,dfs_imag]):
					sub_df = pd.DataFrame({key:value.cpu().numpy() for key,value in stats.items()})
					sub_df['iteration'] = iteration
					if not target_token_type is None:
						onset = batch_size*batch_ix
						offset = batch_size*(batch_ix+1)
						sub_df['target_token_type'] = target_token_type[onset:offset]
						sub_df['disturbant_token_type'] = suffix_tokens_type[onset:offset]
					dfs.append(sub_df)
		for dfs,type in zip([dfs_full,dfs_real,dfs_imag],['','_real','_imag']):
			pd.concat(dfs, axis=0).to_csv(save_path.format(type), index=False)

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('model_dir', type=str, help='Path to the directory where model checkpoints are stored.')
	parser.add_argument('--prefix_length', type=int, default=0, help='Length of prefix to the target token.')
	parser.add_argument('--num_seq_triplets', type=int, default=1024, help='# of sequence triplets (base, shared initial token, shared suffix).')
	parser.add_argument('--device', type=str, default='cpu', help='Device.')
	parser.add_argument('--batch_size', type=int, default=None, help='Batch size to chunk test data (for VRAM management).')
	args = parser.parse_args()

	logger = get_logger()#args.log_path,filename='bag-of-words.log')

	tester = Tester(logger, args.model_dir, device=args.device)
	save_fn = 'jacobian_stats_prefix-{}'.format(args.prefix_length) \
				if args.prefix_length>0 else 'jacobian_stats'
	save_fn = save_fn+'{}.csv'
	save_path = os.path.join(args.model_dir, save_fn)
	tester(save_path, args.num_seq_triplets, args.prefix_length, batch_size=args.batch_size)