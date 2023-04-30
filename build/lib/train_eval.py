'''
    Trainer and evaluator.
'''

import torch
from torch.utils.data import DataLoader

from src.pytorch_utils import duration

####################################################################################################

class Trainer:
    ''' Class that holds together variables needed for NN training. '''

    def __init__(self, model, dataset, device, lr=0.001, batch_size=16, n_epochs=10, log=lambda *_: None):
        # Required params.
        self.model, self.dataset, self.device = model, dataset, device
        # Optional params.
        self.lr, self.batch_size, self.n_epochs, self.log = lr, batch_size, n_epochs, log
        # Training results.
        self.loss = float('inf')
        self.duration = 0

    def train(self):
        # Set model to training mode.
        self.model.train()
        # Time the training loop.
        _, self.duration = duration('Training', self._train, self.log)

    def _train(self):
        # Variables.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        total_batches = 0
        # criterion = torch.nn.CrossEntropyLoss()
        # Training loop.    
        for epoch in range(1, self.n_epochs+1):
            for b, (x, y) in enumerate(DataLoader(self.dataset, batch_size=self.batch_size)):
                # Move tensors to the device.
                x, y = x.to(self.device), y.to(self.device)
                # Forward pass.
                _, loss = self.model(x, y)
                # my_loss = criterion(logits, y)
                # assert torch.allclose(loss, my_loss)
                # Backward pass.
                optimizer.zero_grad()
                loss.backward()
                # Update the model parameters.
                optimizer.step()
            # Print the loss for every epoch.
            total_batches += b+1
            self.loss = loss.item()
            self.log(f'- epoch #{epoch}, total batches: {total_batches}:, loss: {self.loss}')

    def set_lr(self, lr):
        self.lr = lr

    def set_n_epochs(self, n_epochs):
        self.n_epochs = n_epochs

####################################################################################################

# Disable gradient computation in evaluation.    
@torch.no_grad()
def eval_model(model, data_loader, device, n):
    # Set model to evaluation mode.
    model.eval()
    # List of all wrong predictions.
    mistakes = []
    
    # Loop over batches of data.
    for _, (x, y) in enumerate(data_loader):
        # Move tensors to the device.
        x, y = x.to(device), y.to(device)
        # Isolate the input numbers and sorted numbers.
        nums = x[:, :n]
        sorted = y[:, -n:]
        # Generate the output from the model using greedy argmax and get the solution part.
        solution = model.generate(nums, n, do_sample=False)[:, n:]
        # Compare the predicted sequence to the actual sequence.
        wrong = (sorted != solution).any(1).cpu()
        # Add the wrong predictions to the mistakes list.
        mistakes.extend( zip(nums[wrong], sorted[wrong], solution[wrong]) )
    
    # Print the first 5 mistakes.
    for m_nums, m_sorted, m_solution in mistakes[:5]:
        print(f'- model claims {m_nums} sorted is {m_solution}, not {m_sorted}.')
    
    # Print test results.
    n_samples = len(data_loader.dataset)
    n_corr = n_samples - len(mistakes) 
    print(f'Test score: {n_corr} of {n_samples} = {100*n_corr/n_samples:.2f}% correct')

####################################################################################################
