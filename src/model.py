''' 
    Zel's recreation of a toy GPT model, for educational purposes. 

    Reference GPT-2 diagram: ./Wiki-Full-GPT-architecture.png
    Number of parameters for each layer: ./param-calc.xlsx
    Based on minGPT: https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
'''

import math
from types import SimpleNamespace

import torch, torch.nn as nn, torch.nn.functional as F

import src.pytorch_utils as utils

####################################################################################################

class GPT(nn.Module):
    ''' Top class for GPT model. '''

    class Config:
        PICO = SimpleNamespace(n_embd=16, n_layer=2, n_head=2, dropout=0.1, vocab_size=None, block_size=None)
        NANO = SimpleNamespace(n_embd=48, n_layer=3, n_head=3, dropout=0.1, vocab_size=None, block_size=None)
        GPT2 = SimpleNamespace(n_embd=768, n_layer=12, n_head=12, dropout=0.1, vocab_size=None, block_size=None)
    
    def __init__(self, config, log=lambda *_: None):
        super().__init__()
        # Check that vocab_size and block_size were set.
        assert config.vocab_size is not None and config.block_size is not None
        # Save config.
        self.conf = config
        # Create input and position embeddings.
        self.input_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        # Create other layers: dropout, transformer blocks, layer norm, linear.
        self.sequential = nn.Sequential(
            nn.Dropout(config.dropout),
            *[TransformerBlock(config) for _ in range(config.n_layer)],
            nn.LayerNorm(config.n_embd),
            nn.Linear(config.n_embd, config.vocab_size, bias=False)
        )
        # Count and print parameters, layer by layer.
        utils.count_parameters(self, log)

    def forward(self, idx, targets=None):
        # Get batch size and sequence length.
        B, T = idx.shape
        # Convert from vocabulary to embeddings.
        x = self.input_embedding(idx); assert x.shape == (B, T, self.conf.n_embd)
        # Convert each possible position to positional embedding.
        positions = self.position_embedding( 
            torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0) )
        assert positions.shape == (1, T, self.conf.n_embd)
        # Add embeddings, positions will be broadcasted over batches.
        x = x + positions
        # Pass through other layers.
        logits = self.sequential(x); assert logits.shape == (B, T, self.conf.vocab_size)
        # If targets are provided, compute loss.
        loss = None if targets is None else F.cross_entropy(
            # Use exact dimensions and ignore targets padding (-1).
            logits.view(B*T, self.conf.vocab_size), targets.view(B*T).long(), ignore_index=-1 )
            # logits.view(B*T, self.conf.vocab_size), targets.view(B*T) )
        # Return logits and loss.
        return logits, loss

    def configure_optimizers(self, train_config):
        ''' Returns plain Adam optimizer with specified learning rate. MinGPT uses weight decay, 
            but that is not needed for small models, and it complicates the implementation:
            https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L215 '''
        return torch.optim.Adam(self.parameters(), lr=train_config.learning_rate)

    # Disable gradient computation as generation is only used for evaluation.    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        ''' Generates max_new_tokens given input tokens idx of shape (B,T). It calls model 
            again for each new token. Returns a tensor of shape (B,T+max_new_tokens). '''
        # This method should only be called in eval mode.
        assert not self.training
                
        # Generate tokens one by one.
        for _ in range(max_new_tokens):
            # Crop if sequence became longer than block size.
            idx_cond = idx if idx.size(1) <= self.conf.block_size else idx[:, -self.conf.block_size:]
            # Get number of batches and sequence length.
            B, T = idx_cond.shape
            # Run the model to get logits, loss is ignored.
            logits, _ = self(idx_cond); assert logits.shape == (B, T, self.conf.vocab_size)
            # We only need the logits for the last token.
            logits = logits[:, -1, :] / temperature; assert logits.shape == (B, self.conf.vocab_size)
            # Crop logits to top_k options if specified, increasing focus on
            # higher probability tokens and potentially improving output coherence.
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                minimums = v[:, [-1]]; assert minimums.shape == (B, 1)
                logits[logits < minimums] = -float('Inf')
            # Get probabilities from logits. 
            probs = F.softmax(logits, dim=-1)
            # Either sample from the distribution or take the most likely token.
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            assert idx_next.shape == (B, 1)
            # Append new token to the sequence.
            idx = torch.cat((idx, idx_next), dim=1)

        # Return the last sequence.
        return idx

####################################################################################################

class TransformerBlock(nn.Module):
    ''' One transformer block. '''

    def __init__(self, config):
        super().__init__()
        # Create all components.
        self.layers = nn.Sequential(
            utils.ResidualBlock( 
                nn.LayerNorm(config.n_embd),
                # nn.MultiheadAttention(config.n_embd, config.n_head, dropout=config.dropout), # PyTorch's implementation.
                CasualSelfAttention(config) # Manual implementation.
            ),
            nn.LayerNorm(config.n_embd),
            utils.ResidualBlock(
                nn.Linear(config.n_embd, config.n_embd*4),
                nn.GELU(),
                nn.Linear(config.n_embd*4, config.n_embd),
                nn.Dropout(config.dropout)
            )            
        )

    def forward(self, x):
        return self.layers(x)

####################################################################################################

class CasualSelfAttention(nn.Module):
    ''' Manual implementation of self-attention. 
        Not using PyTorch's nn.MultiheadAttention, so we can see what's going on. '''

    def __init__(self, config):
        super().__init__()
        # Check that embedding size is divisible by number of heads.
        assert config.n_embd % config.n_head == 0
        # Save config.
        self.n_head = config.n_head
        self.head_embd = config.n_embd // config.n_head
        # Create layers.
        self.in_linear = nn.Linear(config.n_embd, config.n_embd*3)
        self.att_dropout = nn.Dropout(config.dropout)
        self.out_linear = nn.Linear(config.n_embd, config.n_embd)
        self.out_dropout = nn.Dropout(config.dropout)
        # Causal mask to ensure that attention is only applied to the left in the input sequence.
        # Created as buffer because it is not updated during backprop.
        self.register_buffer(
            "mask", torch.tril(torch.ones(1, 1, config.block_size, config.block_size)) == 0 )

    def forward(self, x):
        # Get batch size, sequence length, and embedding size.
        B, T, C = x.shape
        # Expand embeddings 3X, because each head will need query, key, and value.
        x = self.in_linear(x)
        # Split heads (B, T, n_head, 3*C//n_head), transpose to (B, n_head, T, 3*C//n_head).
        x = x.reshape(B, T, self.n_head, -1).transpose(1, 2)
        # Split each head (last dimension) into query, key, and value.
        q, k, v = x.chunk(3, dim=-1); assert q.shape == (B, self.n_head, T, self.head_embd)
        # Self-attend: (B, n_head, T, head_embd) x (B, n_head, head_embd, T) -> (B, n_head, T, T)
        att = q @ k.transpose(-2, -1) / math.sqrt(self.head_embd)
        # Apply causal mask to ignore future tokens.
        att = att.masked_fill(self.mask[:,:,:T,:T], float('-inf'))
        # Apply softmax and dropout, both for normalization.
        att = self.att_dropout( F.softmax(att, dim=-1) )
        # Output is attention applied to values.
        x = att @ v
        # Re-assemble heads.
        x = x.transpose(1, 2).reshape(B, T, C) 
        # Compress back to embedding size, apply dropout, and return.
        return self.out_dropout(self.out_linear(x))

####################################################################################################
