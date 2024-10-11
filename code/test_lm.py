# coding: utf-8

import os,argparse
import pandas as pd
import torch
import torch.nn.functional as F
from torchtext.vocab import vocab as ordered2vocab #build_vocab_from_iterator
from collections import OrderedDict
from torchtext.transforms import SentencePieceTokenizer
from torchdata.datapipes.iter import FileOpener,LineReader
from utils.logging import get_logger
from train_lm import Learner

class Tester(Learner):
	def __init__(self, logger, checkpoint_path, device='cpu'):
		self.logger = logger
		self.device = torch.device(device)
		self.retrieve_model(checkpoint_path)

	def test(self, dataloader, save_path):
		# vocab_size = self.checkpoint['modules']['rnn']['init_args']['vocab_size']
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		total_loss = 0.0
		num_tokens = 0.0
		with torch.no_grad():
			for text in dataloader:
				text = text.to(self.device)

				input = text[:,:-1]
				target=text[:,1:]
				logits = self.rnn(input)

				vocab_size = logits.size(-1)
				mask = target>=vocab_size # Mask out-of-vocab items.
				masked_target = target.masked_fill(mask, 0) # Dummy target
				loss = F.cross_entropy(logits.view(-1,vocab_size), masked_target.view(-1), reduction='none')
				unmask_float = mask.logical_not().float()
				loss = (loss.view_as(target)*unmask_float).sum(dim=0) # sum over batch
				counts = unmask_float.sum(dim=0)

				if isinstance(total_loss, torch.Tensor):
					pad_length = total_loss.nelement()-loss.nelement()
					if pad_length>0:
						loss = F.pad(loss, (0,pad_length), value=0.0)
						counts = F.pad(counts, (0,pad_length), value=0)
					elif pad_length<0:
						total_loss = F.pad(total_loss, (0,-pad_length), value=0.0)
						num_tokens = F.pad(num_tokens, (0,-pad_length), value=0)
				total_loss += loss
				num_tokens += counts
		df = pd.DataFrame()
		df['perplexity'] = (total_loss/num_tokens).exp().cpu().numpy()
		df['num_tokens'] = num_tokens.cpu().numpy()
		df['time'] = df.index+1
		df.to_csv(save_path, index=False)

	def __call__(self, dataset, vocabs, save_path,
					batch_size=1, num_workers=1,
					padding_token='<pad>', unk_token='<unk>', tokenizers=None,
					max_length=None, specials=None):
		from torchtext.vocab import Vocab
		from torchtext._torchtext import Vocab as VocabPybind
		# from collections import OrderedDict
		vocabs = {name:Vocab(VocabPybind(itos,None))
					for name,itos in self.checkpoint['vocabs'].items()}
		for vocab in vocabs.values():
			if unk_token in vocab:
				vocab.set_default_index(vocab[unk_token])
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
								dataset,
								batch_size=batch_size,
								shuffle=False,
								num_workers=num_workers,
								collate_fn=self.collate_fn,
								worker_init_fn=worker_init_fn)
		self.test(dataloader, save_path)

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('checkpoint', type=str, help='Path to the checkpoint of the trained model.')
	parser.add_argument('data_path', type=str, help='Path to the text data.')
	parser.add_argument('tokenizer_prefix', type=str, help='Prefix to the config & vocab files for a pretrained tokenizer.')
	parser.add_argument('save_path', type=str, help='Path to the csv file where results are saved.')

	parser.add_argument('--max_length', type=int, default=None, help='Maximum input length.')

	parser.add_argument('--batch_size', type=int, default=512, help='Batch size.')
	parser.add_argument('--num_workers', type=int, default=1, help='# of dataloading workers.')

	# parser.add_argument('--log_path', type=str, default=None, help='Path to the .log file where results are logged.')
	parser.add_argument('--device', type=str, default='cpu', help='Device.')
	args = parser.parse_args()

	# if not args.log_path is None:
		# os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
	logger = get_logger()#args.log_path)

	def build_dataset(path):
		datapipe = FileOpener([path], mode='rt')
		datapipe = LineReader(datapipe, return_path=False)
		return datapipe
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
	test_dataset = build_dataset(args.data_path)

	tester = Tester(logger, args.checkpoint, device=args.device)
	tester(test_dataset, vocabs, args.save_path, args.batch_size,
			args.num_workers, padding_token=unk_or_pad, unk_token=unk_or_pad,
			tokenizers=dict(text=tokenizer), specials=specials,
			max_length=None if args.max_length is None else args.max_length+1)