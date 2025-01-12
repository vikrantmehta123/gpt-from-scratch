from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tiktoken
import inspect
import os
import time
import numpy as np
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

ddp = int(os.environ.get('RANK', -1)) != -1  # if you have multiple GPUs then, this condition will be true & you want to execute script in parallel

if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE']) # the total number of GPUs across all nodes

    device = f'cuda:{ddp_local_rank}' # Device name within that node. All nodes index GPUs as cuda:0, cuda:1, etc. Thus, we use local rank
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # if the zeroth GPU, then this will be true. For checkpointint, logging, etc.

else:
    # non ddp run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# region Models and Classes

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class CausalSelfAttention(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # 4D shape now
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)) 

    def forward(self, x):
        B, T, C = x.size() # Batch Size, Sequence Length, Embedding Dim

        qkv = self.c_attn(x)

        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Flash Attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)


        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)

        return y

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config:GPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # text embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # positional embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), 
            ln_final = nn.LayerNorm(config.n_embd)
        )) 

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()

        assert T <= self.config.block_size

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = pos_emb + tok_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_final(x)

        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizer(self, weight_decay, lr, device):
        # Get all params that require gradient i.e. will be updated by optimizer
        param_dict = {pn:p for pn, p in self.named_parameters()}
        param_dict = {pn:p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        non_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params' : decay_params, 'weight_decay': weight_decay}, 
            {'params': non_decay_params, 'weight_decay': 0.0}
        ]

        # In later versions of PyTorch have this fused, not earlier.
        # If this fused parameter is present, then it's again kernel fusion (all params are updated in one kernel) and thus runs faster.
        # By default, it is not used.
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        if master_process:
            print(f"using fused AdamW: {use_fused}")
            
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused )
        return optimizer
    
# endregion

# region Data Loader

def load_tokens(filename):
    # Read the raw binary file
    with open(filename, 'rb') as f:
        data = f.read()
    # Convert binary data back to a NumPy array
    npt = np.frombuffer(data, dtype=np.uint16).reshape(-1)
    # Convert the NumPy array to a PyTorch tensor
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        data_root = '/kaggle/temp/edu_fineweb10B'
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards

        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T+1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)

        self.current_position += B * T * self.num_processes

        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# endregion

# region Learning Rate Scheduler, Optimizer
max_steps = 15 # maximum optimization "steps"
max_lr = 6e-4
min_lr = max_lr * 0.1 # Min LR is 10% of max lr
warmup_steps = 5

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps # (it + 1) to ensure we don't start at zero i.e. when it=0
    if it > max_steps:
        return min_lr
    
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# endregion

# Instantiate the model
total_batch_size = 524288 # 2^19 tokens in one batch
B = 8
T = 1024

model = GPT(GPTConfig(vocab_size=50304))
model.to(device) # This is not going to work on laptop
# model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
optimizer = raw_model.configure_optimizer(weight_decay=0.1, lr=6e-4, device=device)

# region training loop

grad_accum_steps = (total_batch_size) // (B * T * ddp_world_size) # Since there are multiple GPUs, each will have B * T micro batch running

if master_process:
    print(f"Total Desired Batch Size: {total_batch_size}")
    print(f"Number of Gradient Accumulation Steps: {grad_accum_steps}")

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0.0
    
    # Accumulate gradients for some time before you update
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y) 

        loss = loss / grad_accum_steps
        accumulated_loss += loss.detach() # To keep track of how much loss we accumulated over micro batches for printing
        
        # Synchronize only when the total batch is processed
        loss.backward() # accumulate gradients

    if ddp:
        dist.all_reduce(accumulated_loss, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    torch.cuda.synchronize()

    t1 = time.time() 

    dt = t1 - t0
    # tokens processed is also higher now
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt

    if master_process:
        print(f"Step {step:4d} | Loss: {loss.item():.6f} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec} | lr: {lr:4e}")


if ddp:
    destroy_process_group()

model.to('cpu')
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, '/kaggle/working/')

# endregion

# region Sampling Text

model.eval()
num_return_sequences = 5 # number of sequences to generate.
max_length = 30 # The size of the generated text
# Convert the prompt into tokens using the tokenizer used for GPT-2
enc = tiktoken.get_encoding('gpt-2')
tokens = enc.encode("Hello, I'm a language model, ")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = torch.unsqueeze(0).repeat(num_return_sequences, 1) # We want to generate five sequences. So we repeat same prompt, five times
x = tokens.to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.size(1) <= max_length:
    with torch.no_grad():
        logits = model(x)
        logits = logits[:, -1, :]

        probs = F.softmax(logits, dim=-1)

        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # Use Top K sampling method to select 50 tokens with highest probabilities

        ix = torch.multinomial(topk_probs, 1) # Sample one token from the top 50 tokens for each batch-> (B, 1)

        # From the topk token indices, we want to select the only the ones that we sampled for each batch. And we want to collect them in one tensor
        xcol = torch.gather(topk_indices, -1, ix)

        # Append the new tokens to the sequence
        x = torch.cat((x, xcol), dim=1)

# Decode and print
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist() # Get the tokens for the ith sequence
    decoded = enc.decode(tokens)
    print("> ", decoded)

# endregion