import math
import time
import os
import wget

import numpy as np
#import tiktoken

import torch 
import torch.nn as nn 
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

from dataclasses import dataclass
from datasets import load_dataset

######################################
#  Helper Function
######################################

# -------------------------------------
# Load data and convert it in Pytorch
# -------------------------------------

def load_tokens(filename):
    "Load and Transform data into a torch tensor"
    npt = np.load(filename)
    npt = npt.astype(np.int32)                 # Vocabulary size is within int32 range
    ptt = torch.tensor(npt, dtype=torch.long)  # Pytorch requires torch.long for indexing
    return ptt

# -------------------------------------
# Shards data and process it in batches
# -------------------------------------

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank    # ID current process (0 to process_rank - 1) 
        self.num_processes = num_processes  # Total number GPUs running in parallel
        assert split in {'train', 'val'}

        # get the shard filenames
        os.makedirs("edu_fineweb10B", exist_ok = True)
        data_root = "edu_fineweb10B"                            # Root dir
        shards = os.listdir(data_root)                          # List dir's files
        shards = [s for s in shards if split in s]              # Filter files with split
        shards = sorted(shards)                                 # Order shards
        shards = [os.path.join(data_root, s) for s in shards]   # ["train_000.npy"] -> ["edu_fineweb10B/train_000.npy"]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # State, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        # Browse file's token by chunk B*T*Process_rank
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # Inputs
        y = (buf[1:]).view(B, T)  # Targets
        # Advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y
    
# -------------------------------------
# Optimization: Learning Rate
# -------------------------------------

def get_lr(it, warmup_steps, max_steps, max_lr, min_lr):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# -------------------------------------
# Helper function: HellaSwag
# -------------------------------------

def get_most_likely_row(tokens, mask, logits):
    # Evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    # Flatten tensors for loss 
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))                       # (B * (T-1), vocab_size)
    flat_shift_tokens = shift_tokens.view(-1)                                              # (B * (T-1))
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none') # (B * (T-1))
    shift_losses = shift_losses.view(tokens.size(0), -1)                                   # (B, (T-1))
    # Now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # Shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # Sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # Now we have a loss for each of the 4 completions
    # The one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


######################################
#  GPT2 Language Model Architecture
######################################

# -------------------------------------
# Config GPT2
# -------------------------------------

@dataclass
class GPT2Config:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int    = 12    # number of layers
    n_head: int     = 12    # number of heads
    n_embd: int     = 768   # embedding dimension
    
# -------------------------------------
# Attention Block
# -------------------------------------
   
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh: "number of heads", hs: "head size", and C: "number of channels" (nh * hs)
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)     # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)                # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

# -------------------------------------
# MLP
# -------------------------------------

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x) 
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

# -------------------------------------
# Transformer Block
# -------------------------------------

class Block(nn.Module):
    """ 
    Transformer block: 
    Applies LayerNorm → CausalSelfAttention → residual add, 
    then LayerNorm → MLP → residual add.
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# -------------------------------------
# General GPT2 
# -------------------------------------

class GPT2(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),              # Word-Token Embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),              # Word-Position Embedding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # Multi-Head Attention
            ln_f = nn.LayerNorm(config.n_embd),                                # Layer Normalization
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # Finish with a Linear-Head

        # Weight sharing scheme: inking references (reduces the number of parameters by 30%) 
        self.transformer.wte.weight = self.lm_head.weight

        # Init params (apply _init_weights to every submodule)
        self.apply(self._init_weights)


    def _init_weights(self, module):
        """Initialize parameters of the model: cautious variance of residual connections!"""
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # Stabilize variance (sqrt(2* number_residual_paths))
                # Per block there is 2 (Attention-Head + MLP)
                std *= (2 * self.config.n_layer) ** -0.5 
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """Classical forward layer"""
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # Forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod 
    def from_pretrained(cls,model_type): 
        """Loads pretrained from HF"""
        assert model_type in {"gpt2","gpt2-medium","gpt2-large","gpt2-xl"}
        from transformers import GPT2LMHeadModel
        print(f"Loading weights from the pretrained gpt2Model:{model_type}")

        config_args = {
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]

        config_args['vocab_size'] = 50257 # Config GPT2
        config_args['block_size'] = 1024  # Config GPT2

        # Initialize a from-scratch minGPT model
        config = GPT2Config(**config_args) # Unpack model configuration
        model = GPT2(config)               # Create model instance

        # Extract state dictionary (learned parameters)
        sd = model.state_dict()
        sd_keys = [k for k in sd if not k.endswith('.attn.bias')]  # Exclude attention bias buffers (not learnable params)

        # Init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore (buffer)
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]        # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        # The openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        """
        Specialize optimizer treating decay and no decay parameters seperatly
        """
        # Dict all params with requires_grad
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # Distinguish params according to their dimension
        decay_params   = [p for n,p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n,p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_day": 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Check use: Special CUDA-optimized  fused kernel implementation
        
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        used_fused = fused_available and device_type == "cuda"
        if master_process: 
            print(f"Using fused AdamW: {used_fused}")

        # Create AdamW optimizer with hyperparameters
        optimizer = torch.optim.AdamW(optim_groups, lr = learning_rate, betas = (0.9,0.95), eps = 1e-8, fused = used_fused)

        return optimizer
