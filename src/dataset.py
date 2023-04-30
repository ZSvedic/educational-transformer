''' 
    Very simple dataset for sorting integers in range [0, vocab_size-1]. Although simple, 
    it requires transformer to pay attention to a number at eact position, and reproduce
    that number at the sorted position in the output.

    Based on Karpathy's SortDataset: https://github.com/karpathy/minGPT/blob/master/demo.ipynb

    E.g. for problem length 6 and vocab_size 3:
    nums: 0 0 2 1 0 1 -> sorted: 0 0 0 1 1 2
    Transformer input is nums concatenated with first 5 elements of sorted (6th is being predicted):
    input:   0  0  2  1  0  1  0  0  0  1  1
    Correct output of transformer input shifted by 1 to the left, and last element predicted:
    target: -1 -1 -1 -1 -1  0  0  0  1  1  2
'''

import numpy as np
from torch.utils.data import Dataset

####################################################################################################

class SortDataset(Dataset):
    def __init__(self, n_samples, n_input_tokens=6, n_vocab=3):
        # Save parameters.
        self.n_samples = n_samples
        self.n_in_tokens = n_input_tokens
        self.num_digits = n_vocab
        # Generate a NumPy array of random digits.
        self.nums = np.random.randint(n_vocab, size=(n_samples, n_input_tokens))
        # Sort each row in nums.
        self.sorted = np.sort(self.nums, axis=1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        x = np.hstack((self.nums[index,:], self.sorted[index,:-1]))
        # y = np.hstack((self.nums[index,-self.num_tokens+1:], self.sorted[index,:])) # Without masking
        y = np.hstack((np.full((5), -1), self.sorted[index,:]))
        return x, y

####################################################################################################
