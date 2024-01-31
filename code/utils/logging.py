# coding: utf-8

from logging import getLogger,FileHandler,StreamHandler,DEBUG,Formatter
import os.path,sys

def get_logger(file_dir=None):
	logger = getLogger("__main__")
	current_handlers=logger.handlers[:]
	for h in current_handlers:
		logger.removeHandler(h)
	if file_dir is None:
		handler = StreamHandler(sys.stdout)
	else:
		log_file_path = os.path.join(file_dir,'history.log')
		handler = FileHandler(filename=log_file_path)	#Define the handler.
	handler.setLevel(DEBUG)
	formatter = Formatter('{asctime} - {levelname} - {message}', style='{')	#Define the log format.
	handler.setFormatter(formatter)
	logger.setLevel(DEBUG)
	logger.addHandler(handler)	#Register the handler for the logger.
	logger.info("Logger set up.")
	return logger