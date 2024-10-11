# Experiments on Position Encoding Coupled with RNNs

Source code used in the study entitled "Positional Encoding Helps Recurrent Neural Networks Handle a Large Vocabulary".

## Environment

The conda environment used in the study can be recovered from [conda_env.yml](conda_env.yml).

## Description of the Execution Files

- Reverse-Ordering Task:
  - [code/train_palindrome.py](code/train_palindrome.py): Train & test an RNN on the reverse-ordering task.
  - [code/train_palindrome_ssm.py](code/train_palindrome_ssm.py): Train & test the S4D on the reverse-ordering task.
  - [code/train_palindrome_dual-freq.py](code/train_palindrome_dual-freq.py): Train & test an RNN on the reverse-ordering task with a dual-frequency vocabulary consisting of frequent and rare tokens.
    - [code/test_jacobian_palindrome.py](code/test_jacobian_palindrome.py): Analyze the gradients of the RNN trained on the dual-frequency vocabulary.
  - [code/train_palindrome_dual-freq_ssm.py](code/train_palindrome_dual-freq_ssm.py): Train & test the S4D on the reverse-ordering task with a dual-frequency vocabulary consisting of frequent and rare tokens.
    - [code/test_jacobian_palindrome_ssm.py](code/test_jacobian_palindrome_ssm.py): Analyze the gradients of the S4D trained on the dual-frequency vocabulary.
  - [code/train_palindrome_variable-length.py](code/train_palindrome_variable-length.py): Train & test an RNN on the reverse-ordering task w/ variable input length (Appendix B).
- Reverse-Ordering + Delayed Sum:
  - [code/train_delayed-sum.py](code/train_delayed-sum.py): Train & test an RNN on the reverse-ordering + delayed-sum task (Appendix A1).
- Sorting Task:
  - [code/train_sort.py](code/train_sort.py): Train & test an RNN on the sorting task (Appendix A2).
- Predecessor-Query Task:
  - [code/train_predecessor_query.py](code/train_predecessor_query.py): Train & test an RNN on the predecessor-query task (Appendix A3).

[replication.sh](replication.sh) provides more detailed information about the programs as well as the hyperparameters used in the study.