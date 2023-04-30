# %% 
# Imports:
import torch
from torch.utils.data import DataLoader

from src.dataset import SortDataset
from src.model import GPT
from src.pytorch_utils import seed_all
from src.train_eval import Trainer, eval_model

# %% 
# Seed and DataSet:
seed_all()
train, test = SortDataset(6400), SortDataset(1600)
print(f'{len(train)} training samples, {len(test)} test samples')
for si in range(3):
    print(f'Train #{si+1}:', *(train[si]))

# %% 
# Device and config:
device = torch.device("cpu") # Tiny models actually train faster on "cpu" than "mps" or "cuda".
config = GPT.Config.NANO
config.vocab_size = 3
config.block_size = 11
print(f'Device: {device}\nConfig: {config}')

# %% 
# Create and print model:
model = GPT(config, print).to(device)

# %%
# Test untrained model:
result, _ = model( torch.tensor([[1,2,0,0,0,0,0,0,0,0,0]]).to(device) )
print(f'Result shape: {result.shape},\nresult: {result}')

# %% 
# Train model and print training progress:
trainer = Trainer(model, train, device, lr=0.004, batch_size=64, n_epochs=1, log=print)
trainer.train()

# %%
# Evaluation:
eval_model(model, DataLoader(test, batch_size=64), device, test.n_in_tokens)

# %%
# Test on one random example:
inp = torch.randint(low=0, high=config.vocab_size, size=(1, train.n_in_tokens), dtype=torch.long)
result = model.generate(inp.to(device), train.n_in_tokens, do_sample=False)
inp, result = inp.cpu(), result.cpu()
sol = torch.sort(inp[0])[0]
sol_candidate = result[:, train.n_in_tokens:]
print('Input sequence:  ', inp.tolist())
print('Predicted sorted:', sol_candidate.tolist())
print('Actual sort:     ', sol.tolist())
print('Matches:         ', bool((sol == sol_candidate).all()))
