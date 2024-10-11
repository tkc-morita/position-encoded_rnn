# coding: utf-8

import torch

class IterationBasedBatchSampler(torch.utils.data.BatchSampler):
	"""
	Wraps a BatchSampler, resampling from it until
	a specified number of iterations have been sampled.
	Partially Copied from maskedrcnn-benchmark.
	https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/samplers/iteration_based_batch_sampler.py
	"""

	def __init__(self, batch_sampler, num_iterations, start_iter=0):
		self.batch_sampler = batch_sampler
		self.num_iterations = num_iterations
		self.start_iter = start_iter
		if hasattr(self.batch_sampler.sampler, 'set_start_ix'):
			start_ix = (self.start_iter % len(self.batch_sampler)) * self.batch_sampler.batch_size
			self.batch_sampler.sampler.set_start_ix(start_ix)

	def __iter__(self):
		iteration = self.start_iter
		epoch = iteration // len(self.batch_sampler)
		while iteration <= self.num_iterations:
			if hasattr(self.batch_sampler.sampler, 'set_epoch'):
				self.batch_sampler.sampler.set_epoch(epoch)
			for batch in self.batch_sampler:
				iteration += 1
				if iteration > self.num_iterations:
					break
				yield batch
			epoch += 1

	def __len__(self):
		return self.num_iterations

class _CustomSamplerBase(object):
	def set_epoch(self, epoch):
		self.epoch = epoch

	def set_start_ix(self, start_ix):
		self.start_ix = start_ix

class RandomSampler(torch.utils.data.RandomSampler, _CustomSamplerBase):
	"""
	Custom random sampler for iteration-based learning.
	"""
	def __init__(self, *args, seed=111, **kwargs):
		super(RandomSampler, self).__init__(*args, **kwargs)
		self.epoch = 0
		self.start_ix = 0
		self.seed = seed

	def __iter__(self):
		g = torch.Generator()
		g.manual_seed(self.epoch+self.seed)
		start_ix = self.start_ix
		self.start_ix = 0
		n = len(self.data_source)
		max_chunk_size = 300000
		if self.replacement:
			return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64, generator=g)[start_ix:].tolist())
		if n<=max_chunk_size:
			return iter(torch.randperm(n, generator=g)[start_ix:].tolist())
		return self._chunky_generator(n,start_ix,max_chunk_size,g)
	
	@staticmethod
	def _chunky_generator(dataset_size,start_ix,max_chunk_size,generator):
		init_chunk,start_ix = divmod(start_ix, max_chunk_size)
		init_chunk *= max_chunk_size
		for chunk_start in range(init_chunk,dataset_size,max_chunk_size):
			chunk_size = min(max_chunk_size,dataset_size-chunk_start)
			for ix in (torch.randperm(chunk_size, generator=generator)+chunk_start)[start_ix:].tolist():
				yield ix
			start_ix = 0

class VariableOnsetSequentialSampler(torch.utils.data.SequentialSampler, _CustomSamplerBase):
	def __init__(self, *args, batch_size, seed=111, **kwargs):
		super().__init__(*args, **kwargs)
		self.epoch = 0
		self.start_ix = 0
		self.seed = seed
		self.batch_size = batch_size

	def __iter__(self):
		g = torch.Generator()
		g.manual_seed(self.epoch+self.seed)
		shift = torch.randint(high=self.batch_size, size=(1,), dtype=torch.int64, generator=g).item()
		start_ix = self.start_ix
		self.start_ix = 0
		return iter(range(start_ix+shift, len(self.data_source)))


def get_data_loader(dataset, batch_size=1, shuffle=False, num_iterations=None, start_iter=0, num_workers=1, collate_fn=None, random_seed=111):
	if shuffle:
		sampler = RandomSampler(dataset, replacement=False, seed=random_seed)
		drop_last = True
	elif not num_iterations is None:
		sampler = VariableOnsetSequentialSampler(dataset, batch_size=batch_size, seed=random_seed)
		drop_last = True
	else:
		sampler = torch.utils.data.SequentialSampler(dataset)
		drop_last = False
	batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=drop_last)
	if not num_iterations is None:
		batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iterations, start_iter=start_iter)

	data_loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_sampler=batch_sampler, collate_fn=getattr(dataset,'collate_fn',collate_fn))
	return data_loader
