''' 
    Collection of common functions and classes used in PyTorch projects.
'''

import copy, math, os, random, time

import numpy as np
import torch, torch.nn as nn

####################################################################################################

def seed_all(seed=1337):
    ''' Sets seeds for various libraries to ensure determinism. 
        From https://github.com/pytorch/pytorch/issues/11278 '''
    # Python seed.
    random.seed(seed)
    # Numpy seed.
    np.random.seed(seed)
    # Torch seeds and determinism.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # For deterministic hashes of str and bytes objects: 
    # https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED
    os.environ['PYTHONHASHSEED'] = str(seed)

####################################################################################################

class ResidualBlock(nn.Module):
    ''' Residual connection around a layer or sequence of layers. '''

    def __init__(self, *layers):
        super(ResidualBlock, self).__init__()

        if len(layers) == 0:
            raise ValueError("At least one layer must be provided.")
        elif len(layers) == 1:
            self.layer = layers[0]
        else:
            self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.layer(x)
    
####################################################################################################

def count_parameters(model, log=lambda *_: None): 
    ''' Count trainable parameters. Optionaly print layers by passing a print function. '''
    log('Model layers:')
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = param.numel()
            if param.dim() > 1:
                log('- '+name+':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            else:
                log('- '+name+':', num_param)
            total_param += num_param
    log(f'TOTAL trainable parameters: {total_param:,}')
    return total_param

####################################################################################################

def duration(name, func, log=lambda *_: None):
    ''' Measures the duration of a function call. '''
    log(f"Starting {name}...")
    start = time.perf_counter()
    ret_val = func()
    elapsed = time.perf_counter() - start
    log(f"Finished {name} in {elapsed:.2f} seconds.")
    return ret_val, elapsed

####################################################################################################

def search_parameters(trainer, *, n_samples=10, n_retries=2, log=lambda *_: None,
                      scoring=lambda tr: tr.loss,
                      modifiers=[('lr', 0.0001, 0.005), ('batch_size', 16, 128)]):
    ''' Finds hyperparameters that minimize a scoring function, given a list of trainer modifiers. 
        Each modifier is a tuple (attr_name, low, high).
        Returns a tuple (sol_loss, sol_dur, sol_dict). '''

    # Step 1: For each sample, generate modifier touples that are going to be used later.
    # They are going to be reused in each retry and to generate the final solution.
    def modifier_pairs():
        for attr_name, low, high in modifiers:
            # Create a random value between low and high, logarithmic distribution.
            val = math.exp( random.uniform(math.log(low), math.log(high)) )
            # If given integer range, sample integers.
            if type(low) == int and type(high) == int: 
                val = int(val)
            yield attr_name, val
    sample_modifiers = [dict(modifier_pairs()) for _ in range(n_samples)]

    # Step 2: Create a numpy array to store the results.
    SCORE, LOSS, DURATION, n_col = 0, 1, 2, 3 # Poor man's enum :)
    results = np.empty((n_samples, n_retries, n_col))

    # Step 3: Populate the array with training results.
    for si in range(n_samples):
        for ri in range(n_retries):
            # Each sample retry uses a different trainer.
            trainer_copy = copy.deepcopy(trainer)
            # Modify the trainer with the random values from the modifier tuples.
            for attr_name, val in sample_modifiers[si].items():
                setattr(trainer_copy, attr_name, val)
            # Train the modified trainer.
            trainer_copy.train()
            # Store the results.
            results[si, ri, SCORE] = scoring(trainer_copy)
            results[si, ri, LOSS] = trainer_copy.loss
            results[si, ri, DURATION] = trainer_copy.duration

    # Step 4: Average over n_retries.
    avg_results = np.mean(results, axis=1)

    # Step 5: Find the sample with the smallest average score.
    min_idx = np.argmin(avg_results[:, SCORE])
    sol_loss = avg_results[min_idx, LOSS]
    sol_dur = avg_results[min_idx, DURATION]
    sol_dict = sample_modifiers[min_idx]

    # Step 6: Log and return the best solution.
    log(f'Best parameters: {sol_dict} have loss={sol_loss:.5} and duration={sol_dur:.2f} sec')
    return sol_loss, sol_dur, sol_dict

####################################################################################################
