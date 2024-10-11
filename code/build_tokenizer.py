# coding: utf-8

import os,argparse
from torchtext.data.functional import generate_sp_model

def main(data_path, vocab_size, save_dir):
	os.makedirs(save_dir, exist_ok=True)
	generate_sp_model(data_path, vocab_size,
				model_prefix=os.path.join(save_dir,'vocabsize-{}'.format(vocab_size)))

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('data_path', type=str, help='Path to the dataset file.')
	parser.add_argument('vocab_size', type=int, help='Size of the vocabulary to be built by SentencePiece.')
	parser.add_argument('save_dir', type=str, help='Path to the directory where the trained tokenizer is saved.')
	args = parser.parse_args()
	main(**vars(args))