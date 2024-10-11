# coding: utf-8

import argparse,os

def main(data_path, save_path):
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	print('Preprocess {}'.format(data_path))
	count = 0
	with open(data_path, 'r') as f_orig, open(save_path, 'w') as f_new:
		for line in f_orig.readlines():
			if not (is_empty_line(line) or is_heading(line)):
				f_new.write(line)
				count+=1
	print('# of lines: {}'.format(count))

def is_empty_line(line):
	return line.strip()==''

def is_heading(line):
	return line.startswith(' =')

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('data_path', type=str, help='Path to the original data file.')
	parser.add_argument('save_path', type=str, help='Path to the formatted data file.')
	args = parser.parse_args()
	main(**vars(args))