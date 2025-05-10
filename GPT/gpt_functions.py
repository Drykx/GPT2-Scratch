import torch 
import torch.nn as nn 
from torch.nn import functional as F
from dataclasses import dataclass


######################################
###   General Functions
######################################

# -------------------------------------
# Function retrieve batches of a dataset
# -------------------------------------

def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    """ Randomly select the number of block_size in batches of size block_size """

    max_start = len(data) - block_size - 1
    ix = torch.randint(0, max_start + 1, (batch_size,)) # Tensor random integer of size (batch_size,)

    x = torch.empty((batch_size,block_size), dtype = data.dtype, device = device)
    y = torch.empty((batch_size,block_size), dtype = data.dtype, device = device)

    for i, start in enumerate(ix): 
        x[i] = data[start : start + block_size]
        y[i] = data[start + 1 : start + block_size + 1]

    return x,y

# -------------------------------------
# Formatting function
# -------------------------------------

def poem_formatting(text: str):

    # Remove unwanted prefixes
    unwanted_pref = ("Poésie","Poète", "Recueil")
    cleaned_lines = [
        line for line in text.splitlines()            # split text into a list of lines
        if not line.strip().startswith(unwanted_pref) # condition that they do not start with the prefixes
        ]
    title = cleaned_lines[0].split(":",1)[1].strip()  # Keep the title row but remove that its the title
    cleaned_lines = cleaned_lines[1:]                 # Remove the title
    poem = "\n".join(cleaned_lines).strip()           # Format back into text
    return([title,poem])

######################################
#   GPT Language Model Architecture
######################################

# -------------------------------------
# Hyperparameters Configuration GPT Model
# -------------------------------------

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_embd: int
    n_head: int
    n_layer: int
    batch_size: int
    max_steps: int
    eval_iter: int
    eval_interval: int
    learning_rate: float
    dropout: float
    device: str = "cuda"


# -------------------------------------
# Single Attention Head
# -------------------------------------

class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size, config: GPTConfig):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)

        # Causal mask to ensure each position only attends to the left
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        # Compute attention scores
        attn_scores = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        attn_scores = attn_scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        v = self.value(x)
        out = attn_weights @ v  # (B, T, head_size)

        return out

# -------------------------------------
# Multi-Head Attention
# -------------------------------------

class MultiHeadAttention(nn.Module):
    """Multiple attention heads in parallel"""

    def __init__(self, num_heads, head_size, config: GPTConfig):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, config) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# -------------------------------------
# Feed-Forward Network
# -------------------------------------

class FeedForward(nn.Module):
    """Simple MLP block"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------------------
# Transformer Block
# -------------------------------------

class Block(nn.Module):
    """Transformer block: attention + feedforward"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        head_size = config.n_embd // config.n_head

        self.attn = MultiHeadAttention(config.n_head, head_size, config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# -------------------------------------
# Full GPT Language Model
# -------------------------------------

class GPTLanguageModel(nn.Module):
    """GPT-like language model"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding(idx)             # (B, T, n_embd)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = token_emb + pos_emb                           # (B, T, n_embd)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                          # (B, T, vocab_size)

        loss = None
        if targets is not None:
            logits_flat = logits.view(-1, logits.size(-1))   # (B*T, vocab_size)
            targets_flat = targets.view(-1)                  # (B*T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                # take logits at last time step
            probs = F.softmax(logits, dim=-1)        # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)  # (B, T+1)
        return idx