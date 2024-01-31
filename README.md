# Experiments on Position Encoding Coupled with RNNs

Source code used in the study entitled "Positional Encoding Helps Recurrent Neural Networks Handle a Large Vocabulary".

## Environment

The conda environment used in the study can be recovered from [conda_env.yml](conda_env.yml).

## Description of the Execution Files

- Reverse-Ordering Task:
  - [code/train_palindrome.py](code/train_palindrome.py): Train & test an RNN on the reverse-ordering task.
  - [code/train_palindrome_autoregression.py](code/train_palindrome_autoregression.py): Train & test an autoregressive RNN on the reverse-ordering task.
  - [code/train_palindrome_variable-length.py](code/train_palindrome_variable-length.py): Train & test an RNN on the reverse-ordering task w/ variable input length.
  - [code/train_palindrome_continuous.py](code/train_palindrome_continuous.py): Train & test an RNN on the reverse-ordering task w/ continuous random samples from the unit hypersphere (i.e., using an "infinite" vocabulary).
  - [code/extend-vocab_palindrome.py](code/extend-vocab_palindrome.py): Train extra weights of input-embedding & output-projection layers for an extended vocaburary.
- Sorting Task:
  - [code/train_sort.py](code/train_sort.py): Train & test an RNN on the sorting task.
  - [code/train_sort_autoregression.py](code/train_sort_autoregression.py): Train & test an autoregressive RNN on the sorting task.
  - [code/test_sort_pairwise-order.py](code/test_sort_pairwise-order.py): Evaluate the sorting of outputs of a trained RNN ignoring their accuracy against the ground-truth target sequence.
- Copying Memory Task:
  - [code/train_copy-memory.py](code/train_copy-memory.py): Train & test an RNN on the copying-memory task.

The study used the following random seeds: 111, 222, 333, 444, and 555.
Refer to Appendix A of the paper for information about the other hyperparameters.