# coding: utf-8

import os,argparse,math
import torch
# import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
# from torchtext.datasets import WikiText103
# from torchtext.data.functional import load_sp_model,sentencepiece_tokenizer
# from torchtext.datasets.udpos import NUM_LINES
from torchtext.vocab import vocab as ordered2vocab #build_vocab_from_iterator
from collections import OrderedDict
from torchtext.transforms import SentencePieceTokenizer
from torchdata.datapipes.iter import FileOpener,LineReader
# from torch.utils.data import DataLoader
# from torch.utils.data.backward_compatibility import worker_init_fn
from utils.training_template import NLPLearner
from utils.logging import get_logger

class Learner(NLPLearner):
	def train_per_iteration(self, text, records, iteration):
		self.optimizer.zero_grad()
		text = text.to(self.device)

		# vocab_size = self.checkpoint['modules']['rnn']['init_args']['vocab_size']
		input = text[:,:-1]

		loss,normalizer = self.rnn(input, target=text[:,1:])
		loss = loss.sum() / normalizer.sum() # NOTE: Different # of non-unk tokens on different GPUs.
		self.update_records(records, 'loss', loss.item())

		loss.backward()
		clip_grad_norm_(self.get_parameters(), 1.0)
		self.optimizer.step()
		self.scheduler.step(iteration)
		return records

	def log_training_stats(self, records, saving_interval):
		self.logger.info('Cross entropy loss (in perplexity): {:0.6f}'.format(math.exp(records['loss']/saving_interval)))

	def test(self, dataloader):
		# vocab_size = self.checkpoint['modules']['rnn']['init_args']['vocab_size']
		total_loss = 0.0
		num_tokens = 0.0
		for text in dataloader:
			text = text.to(self.device)

			input = text[:,:-1]
			loss,normalizer = self.rnn(input, target=text[:,1:])

			total_loss += loss.sum().item()
			num_tokens += normalizer.sum().item()
		total_loss /= num_tokens

		self.logger.info('Test perplexity: {}'.format(math.exp(total_loss)))
		self.logger.info('# non-<unk> tokens: {}'.format(num_tokens))

	def collate_fn(self, text):
		text = self.transforms['text'](list(text))
		return text


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('data_dir', type=str, help='Path to the directory where data are stored.')
	parser.add_argument('tokenizer_prefix', type=str, help='Prefix to the config & vocab files for a pretrained tokenizer.')
	# parser.add_argument('vocab_size', type=int, help='Vocabulary size for SentencePiece.')
	# parser.add_argument('seq_length', type=int, help='Sequence length.')
	parser.add_argument('save_dir', type=str, help='Path to the directory where results are saved.')

	parser.add_argument('--max_length', type=int, default=None, help='Maximum input length.')

	# parser.add_argument('--min_freq', type=int, default=3, help='Minimum training frequency of tokens to be included in the vocab.')

	# parser.add_argument('--num_held_out', type=int, default=0, help='# of random sequences to be held out for testing.')

	parser.add_argument('--rnn_name', type=str, required=True, choices=['RNN','GRU','LSTM'], help='Type of RNN.')
	parser.add_argument('--hidden_size', type=int, default=512, help='Dimensionality of hidden layer(s) in RNN.')
	parser.add_argument('--embed_size', type=int, default=None, help='Dimensionality of input (& time) embeddings. Equals to hidden_size if not specified.')
	parser.add_argument('--time_encoding', type=str, default=None, choices=['add','concat'], help='Specifies whether time encoding is added to or concatenated with the input embeddings. Time encoding is not used if this option is left unspecified.')
	parser.add_argument('--time_encoding_form', type=str, default='sinusoidal', choices=['sinusoidal','learnable','random'], help='Implementation of time encoding.')
	parser.add_argument('--num_layers', type=int, default=1, help='# of layers in RNN.')
	parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate in RNN.')
	parser.add_argument('--learnable_padding_token', action='store_true', help='Use a learnable embedding for the dummy token in the output phase. Otherwise, the dummy token is represented by the zero vector.')

	parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate of Adam optimizer.')
	parser.add_argument('--num_iterations', type=int, default=10000, help='# of training iterations.')
	parser.add_argument('--warmup_iters', type=int, default=0, help='# of warm-up iterations.')
	parser.add_argument('--saving_interval', type=int, default=1, help='Intervals of logging of the learning progress.')
	parser.add_argument('--batch_size', type=int, default=512, help='Batch size.')
	parser.add_argument('--num_workers', type=int, default=1, help='# of dataloading workers.')

	parser.add_argument('--device', type=str, default='cpu', help='Device.')
	parser.add_argument('--seed', type=int, default=111, help='Random seed.')
	args = parser.parse_args()

	os.makedirs(args.save_dir, exist_ok=True)
	logger = get_logger(args.save_dir)

	logger.info('Learns language-modeling on the WikiText103 dataset.')
	def build_dataset(path):
		datapipe = FileOpener([path], mode='rt')
		datapipe = LineReader(datapipe, return_path=False)
		return datapipe
	train_dataset = build_dataset(os.path.join(args.data_dir, 'wiki.train.raw'))
	tokenizer = SentencePieceTokenizer(args.tokenizer_prefix+'.model')
	unk_or_pad = '<unk>' # NOTE: Abuse <unk> for padding.
	specials = ['<s>','</s>',unk_or_pad] # NOTE: <s> and </s> represent beginning- and end-of-sentence respectively.
	vocab_as_dict = OrderedDict()
	vocab = list()
	with open(args.tokenizer_prefix+'.vocab', 'r') as f:
		for line in f.readlines():
			token,prob = line.rstrip('\n').split('\t')
			vocab.append((token,float(prob)))
	vocabs = dict(text=ordered2vocab(OrderedDict(vocab),
									specials=specials,
									special_first=False,
									min_freq=-torch.inf))
	vocabs['text'].set_default_index(vocabs['text'][unk_or_pad]) # Default to <unk|pad>
	# max_length = max(map(len, train_dataset))
	valid_dataset = build_dataset(os.path.join(args.data_dir, 'wiki.valid.raw'))
	# valid_dataset = LengthSetter(valid_dataset, NUM_LINES['valid'])

	vocab_size = len(vocabs['text'])-1 # NOTE: -1 for <unk|pad>
	logger.info('Vocabulary size: {size}'.format(size=vocab_size))

	model_configs = dict()
	model_configs['rnn'] = dict(module_name='RNN',
								init_args=dict(
									vocab_size=vocab_size,
									hidden_size=args.hidden_size,
									rnn_name=args.rnn_name,
									embed_size=args.embed_size,
									time_encoding=args.time_encoding,
									time_encoding_form=args.time_encoding_form,
									max_length=args.max_length,
									num_layers=args.num_layers,
									dropout=args.dropout,
									learnable_padding_token=args.learnable_padding_token,
								))
	optim_config = dict(lr=args.learning_rate, weight_decay=0.0, betas=(0.9,0.98), eps=1e-09)
	scheduler_config = dict(t_initial=args.num_iterations,
								warmup_t=args.warmup_iters,
								warmup_prefix=True, lr_min=0.0)
	learner = Learner(logger, args.save_dir, model_configs, optim_config, scheduler_config,
						device=args.device, seed=args.seed)
	learner(train_dataset, valid_dataset, vocabs, args.num_iterations, args.batch_size, args.saving_interval,
			args.num_workers, padding_token=unk_or_pad, unk_token=unk_or_pad,
			tokenizers=dict(text=tokenizer), specials=specials,
			max_length=None if args.max_length is None else args.max_length+1) # NOTE: +1 s.t. length of input=seq[:-1] is args.max_length.