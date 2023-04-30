# %% 
# Init:
import torch
from torch.utils.data import DataLoader

from src.dataset import SortDataset
from src.model import GPT
from src.pytorch_utils import search_parameters, seed_all
from src.train_eval import Trainer, eval_model

seed_all()
train, test = SortDataset(6400), SortDataset(1600)
device = torch.device("cpu") # Tiny models actually train faster on "cpu" than "mps" or "cuda".
config = GPT.Config.NANO
config.vocab_size = 3
config.block_size = 11
model = GPT(config).to(device)

# %% 
# Find the best hyperparameters:
trainer = Trainer(model, train, device, batch_size=32, n_epochs=1, log=print)
_, _, sol_dict = search_parameters(
    trainer, n_samples=5, n_retries=2, log=print,
    # Criterion finds both low loss and short training time.
    scoring=lambda tr: 30*tr.loss + tr.duration, 
    # Modify learning rate and batch size.
    modifiers=[('lr', 0.0015, 0.0085)] ) # , ('batch_size', 20, 60)

# %%
# Train and evaluate the model with best hyperparameters:
trainer.lr = sol_dict['lr']
trainer.train()
eval_model(model, DataLoader(test, batch_size=64), device, test.n_in_tokens)
