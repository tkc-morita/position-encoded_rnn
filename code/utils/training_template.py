# coding: utf-8

import os.path,copy,itertools
import torch
import torch.nn as nn
from timm.scheduler import CosineLRScheduler
import model as M
from data.dataloader import get_data_loader

class Learner(object):
	def __init__(self, logger, save_dir, model_configs, optim_config, scheduler_config, device='cpu', seed=111):
		self.logger = logger
		self.retrieval = os.path.isfile(os.path.join(save_dir, 'checkpoint.pt'))
		self.device = torch.device(device)
		if self.retrieval:
			self.logger.info("CONTINUE PREVIOUS LEARNING.")
		self.logger.info("PyTorch ver.: {ver}".format(ver=torch.__version__))
		self.logger.info('Device: {device}'.format(device=device))
		if torch.cuda.is_available():
			if device.startswith('cuda'):
				self.logger.info('CUDA Version: {version}'.format(version=torch.version.cuda))
				if torch.backends.cudnn.enabled:
					self.logger.info('cuDNN Version: {version}'.format(version=torch.backends.cudnn.version()))
				for device_idx in range(torch.cuda.device_count()):
					self.logger.info('CUDA Device #{device_idx}: {device_name}'.format(device_idx=device_idx, device_name=torch.cuda.get_device_name(device_idx)))
			else:
				print('CUDA is available. Restart with option -C or --cuda to activate it.')

		self.save_dir = save_dir

		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

		if self.retrieval:
			self.retrieve_model()
			self.logger.info('Model retrieved.')
		else:
			self.seed = seed
			torch.manual_seed(seed)
			torch.cuda.manual_seed_all(seed) # According to the docs, "It’s safe to call this function if CUDA is not available; in that case, it is silently ignored."
			self.logger.info('Random seed: {seed}'.format(seed = seed))
			self.checkpoint = dict(modules=dict())
			for attr_name,kwargs in model_configs.items():
				self.set_module(attr_name, **kwargs)
				self.checkpoint['modules'][attr_name] = kwargs
			self.optimizer = torch.optim.Adam(self.get_parameters(), **optim_config)
			self.logger.info('Modules are trained by Adam optimizer w/ following parameters: {}'.format(str(optim_config)))
			self.checkpoint['optimizer'] = dict(init_args=optim_config)
			self.scheduler = CosineLRScheduler(self.optimizer, **scheduler_config)
			self.logger.info('Learning rate is scheduled by CosineLRScheduler w/ following parameters: {}'.format(str(scheduler_config)))
			self.checkpoint['scheduler'] = dict(init_args=scheduler_config)
			self.distinct_optimizers = dict()
			self.checkpoint['distinct_optimizers'] = dict()
			self.distinct_schedulers = dict()
			self.checkpoint['distinct_schedulers'] = dict()

	def update_records(self, records, name, value):
		records[name] = records.get(name, 0.0) + value

	def set_module(self, attr_name, module_name, init_args, state_dict=None):
		self.logger.info('{module_name} instantiated as "{attr_name}" w/ following parameters: {init_args}'.format(
							module_name=module_name, attr_name=attr_name, init_args=str(init_args)))
		init_args = copy.deepcopy(init_args) # NOTE: Some module pops dictionary arguments at instantiation, which synchronyously modifies the init_args in checkpoint.
		module = getattr(M, module_name)(**init_args)
		if not state_dict is None:
			module.load_state_dict(state_dict, strict=True)
			self.logger.info('state_dict loaded on {}.'.format(module_name))
		setattr(self, attr_name, nn.DataParallel(module.to(self.device)))


	def get_parameters(self):
		return itertools.chain.from_iterable(
				[getattr(self, attr_name).parameters()
				for attr_name in self.checkpoint['modules'].keys()])

	def train_per_iteration(self, batch, records, iteration):
		self.optimizer.zero_grad()
		# NOTE: Fill out the training procedure here.
		return records

	def log_training_stats(self, records, saving_interval):
		# NOTE: Fill out logging operations below
		pass

	def test(self, batch):
		# NOTE: Post-training test operations here.
		pass

	def train(self, dataloader, num_iterations, saving_interval, start_iter=0):
		[getattr(self, attr_name).train() for attr_name in self.checkpoint['modules'].keys()]

		records = dict()
		for iteration,batch in enumerate(dataloader, start_iter):
			iteration += 1 # Original starts with 0.

			torch.manual_seed(iteration+self.seed)
			torch.cuda.manual_seed_all(iteration+self.seed)

			records = self.train_per_iteration(batch, records, iteration)

			if iteration % saving_interval == 0:
				self.logger.info('{iteration}/{num_iterations} iterations complete.'.format(iteration=iteration, num_iterations=num_iterations))
				self.log_training_stats(records, saving_interval)
				self.save_model(iteration-1) # Back to the original numbering starting with 0.
				records = dict()
		self.save_model(iteration-1)

	def __call__(self, dataset, num_iterations, batch_size, saving_interval, num_workers=1):
		if self.retrieval:
			start_iter = self.last_iteration + 1
			self.logger.info('To be restarted from the beginning of iteration #: {iteration}'.format(iteration=start_iter+1))
		else:
			self.logger.info("START LEARNING.")
			self.logger.info("max # of iterations: {ep}".format(ep=num_iterations))
			self.logger.info("batch size for training data: {size}".format(size=batch_size))
			self.checkpoint['held_out_data'] = dataset.held_out
			patterns_held_out = 0 if dataset.held_out is None \
								else dataset.held_out.size(0)
			self.logger.info('{} patterns are held out for test.'.format(patterns_held_out))
			start_iter = 0
		dataloader = get_data_loader(
								dataset,
								batch_size=batch_size,
								start_iter=start_iter,
								num_iterations=num_iterations,
								shuffle=True,
								num_workers=num_workers,
								random_seed=self.seed)
		self.train(dataloader, num_iterations, saving_interval, start_iter=start_iter)
		self.logger.info('END OF TRAINING')
		if not dataset.held_out is None:
			self.logger.info('START OF TEST ON HELD-OUT DATA')
			with torch.no_grad():
				self.test(dataset.held_out)
			self.logger.info('END OF TEST ON HELD-OUT DATA')

	def save_model(self, iteration):
		"""
		Save model config.
		"""
		for attr_name,info in self.checkpoint['modules'].items():
			module = getattr(self, attr_name)
			if isinstance(module, nn.DataParallel):
				module = module.module
			info['state_dict'] = module.state_dict()
		self.checkpoint['optimizer']['state_dict'] = self.optimizer.state_dict()
		self.checkpoint['scheduler']['state_dict'] = self.scheduler.state_dict()
		self.checkpoint['random_seed'] = self.seed
		self.checkpoint['iteration'] = iteration
		torch.save(self.checkpoint, os.path.join(self.save_dir, 'checkpoint_after-{iteration}-iters.pt'.format(iteration=iteration+1)))
		torch.save(self.checkpoint, os.path.join(self.save_dir, 'checkpoint.pt'))
		self.logger.info('Config successfully saved.')

	def retrieve_model(self, checkpoint_path = None):
		if checkpoint_path is None:
			checkpoint_path = os.path.join(self.save_dir, 'checkpoint.pt')
		self.checkpoint = torch.load(checkpoint_path, map_location='cpu') # Random state needs to be loaded to CPU first even when cuda is available.
		
		for attr_name,info in self.checkpoint['modules'].items():
			self.set_module(attr_name, **info)

		self.optimizer = torch.optim.Adam(self.get_parameters(),**self.checkpoint['optimizer']['init_args'])
		self.optimizer.load_state_dict(self.checkpoint['optimizer']['state_dict'])
		self.scheduler = CosineLRScheduler(self.optimizer, **self.checkpoint['scheduler']['init_args'])
		self.scheduler.load_state_dict(self.checkpoint['scheduler']['state_dict'])
		
		self.seed = self.checkpoint['random_seed']
		self.last_iteration = self.checkpoint['iteration']


class NLPLearner(Learner):
	def __call__(self, train_dataset, valid_dataset, vocabs, num_iterations,
					batch_size, saving_interval, num_workers=1,
					padding_token='<pad>', unk_token='<unk>', tokenizers=None,
					max_length=None, specials=None):
		if self.retrieval:
			start_iter = self.last_iteration + 1
			self.logger.info('To be restarted from the beginning of iteration #: {iteration}'.format(iteration=start_iter+1))
			from torchtext.vocab import Vocab
			from torchtext._torchtext import Vocab as VocabPybind
			# from collections import OrderedDict
			vocabs = {name:Vocab(VocabPybind(itos,None))
						for name,itos in self.checkpoint['vocabs'].items()}
			for vocab in vocabs.values():
				if unk_token in vocab:
					vocab.set_default_index(vocab[unk_token])
		else:
			self.logger.info("START LEARNING.")
			self.logger.info("max # of iterations: {ep}".format(ep=num_iterations))
			self.logger.info("batch size for training data: {size}".format(size=batch_size))
			start_iter = 0
			self.checkpoint['vocabs'] = {name:vocab.get_itos()
											for name,vocab in vocabs.items()}
		self.vocabs = vocabs
		if specials is None:
			specials = list(set([padding_token,unk_token]))
		self.specials = {name:torch.tensor([vocab[s] for s in specials], device=self.device)
							for name,vocab in vocabs.items()}
		from torchtext.transforms import VocabTransform,ToTensor,Truncate,Sequential
		self.transforms = {name:Sequential(
								VocabTransform(vocab),
								ToTensor(padding_value=vocab[padding_token]
											if padding_token in vocab else None)
								)
							for name,vocab in vocabs.items()}
		if not tokenizers is None:
			for name,tokenizer in tokenizers.items():
				self.transforms[name].insert(0,tokenizer)
		if not max_length is None:
			for name,transforms in self.transforms.items():
				transforms.insert(-2,Truncate(max_seq_len=max_length)) # NOTE: Truncate before ToTensor
		from torch.utils.data import DataLoader
		from torch.utils.data.backward_compatibility import worker_init_fn
		torch.manual_seed(self.seed)
		torch.cuda.manual_seed_all(self.seed)
		dataloader = DataLoader(
								train_dataset,
								batch_size=batch_size,
								shuffle=True,
								num_workers=num_workers,
								collate_fn=self.collate_fn,
								worker_init_fn=worker_init_fn)
		self.train(dataloader, num_iterations, saving_interval, start_iter=start_iter)
		self.logger.info('END OF TRAINING')

		self.logger.info('START OF VALIDATION')
		torch.manual_seed(self.seed)
		torch.cuda.manual_seed_all(self.seed)
		dataloader = DataLoader(
								valid_dataset,
								batch_size=batch_size,
								shuffle=False,
								num_workers=num_workers,
								collate_fn=self.collate_fn,
								worker_init_fn=worker_init_fn)
		self.test(dataloader)
		self.logger.info('END OF VALIDATION')

	def train(self, dataloader, num_iterations, saving_interval, start_iter=0):
		[getattr(self, attr_name).train() for attr_name in self.checkpoint['modules'].keys()]

		iteration = 0
		records = dict()
		while iteration<num_iterations:
			for iteration,batch in enumerate(dataloader, iteration):
				if iteration>=num_iterations:
					break # NOTE: Stop training
				if iteration<start_iter:
					continue # NOTE: Skip previous training.
				iteration += 1 # Original starts with 0.

				torch.manual_seed(iteration+self.seed)
				torch.cuda.manual_seed_all(iteration+self.seed)

				records = self.train_per_iteration(batch, records, iteration)

				if iteration % saving_interval == 0:
					self.logger.info('{iteration}/{num_iterations} iterations complete.'.format(iteration=iteration, num_iterations=num_iterations))
					self.log_training_stats(records, saving_interval)
					self.save_model(iteration-1) # Back to the original numbering starting with 0.
					records = dict()
		self.save_model(iteration-1)

	def collate_fn(self, batch):
		# NOTE: Specific to each task/dataset.
		pass