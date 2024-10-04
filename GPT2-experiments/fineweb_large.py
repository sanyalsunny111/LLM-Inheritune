import os
import time
import math
import pickle
from contextlib import nullcontext
import inspect
import numpy as np
import torch
import torch.nn.functional as F
import tiktoken
import gc
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'rough_path'

eval_interval = 1000
log_interval = 1
eval_iters = 20
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = 'resume'  # 'scratch' or 'resume' or 'gpt2*'
# iter_init = 50000
# wandb logging
wandb_log = False  # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2'  # 'run' + str(time.time())
# data
dataset = 'edu_fineweb100B'
gradient_accumulation_steps = 10  # used to simulate larger batch sizes
batch_size = 2  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# updated large model
n_layer = 32
n_head = 20
n_embd = 1280
# medium model
# n_layer = 24
# n_head = 16
# n_embd = 1024
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
# optimizer
optimizer_name = 'adamw'
learning_rate = 2e-4  # max learning rate
max_iters = 100000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.999
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
rho = 0.1
interval = 10
variant = 4
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 100000  # should be ~= max_iters per Chinchilla
min_lr = 1e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl'  # 'nccl', 'gloo', etc.
# system
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16'  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False  # use PyTorch 2.0 to compile the model to be faster
scale_attn_by_inverse_layer_idx = True
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    # init_process_group(backend='nccl')
    # ddp_rank = int(os.environ['RANK'])
    # ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    # device = f'cuda:{ddp_local_rank}'
    # torch.cuda.set_device(device)
    # master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # if not ddp, we are running on a single gpu, and one process
    # master_process = True
    seed_offset = 0
    # gradient_accumulation_steps *= 8  # simulate 8 gpus
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(5000 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.autocast(device_type=device_type, dtype=ptdtype)

# torch.manual_seed(1337)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)  # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


# dataloader
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "/data/edu_fineweb100B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y


# poor man's data loader
# data_dir = os.path.join('data', dataset)
# train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
# val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
# ddp_rank =0
# ddp_world_size = 1
train_loader = DataLoaderLite(B=batch_size, T=block_size, process_rank=ddp_rank, num_processes=ddp_world_size,
                              split="train")
val_loader = DataLoaderLite(B=batch_size, T=block_size, process_rank=ddp_rank, num_processes=ddp_world_size,
                            split="val")

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
billion_tokens_processed = 0
# attempt to derive vocab_size from the dataset
# meta_path = os.path.join(data_dir, 'meta.pkl')
# meta_vocab_size = None
# if os.path.exists(meta_path):
#     with open(meta_path, 'rb') as f:
#         meta = pickle.load(f)
#     meta_vocab_size = meta['vocab_size']
#     print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout,
                  scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx)  # start with model_args from command line

# Initializing a new model from saved 50K checkpoint with head freezing based on ranks
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    model_args['vocab_size'] = 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt_full.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size  # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(optimizer_name, weight_decay, learning_rate, (beta1, beta2), rho, device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
    del state_dict
    del checkpoint
# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# helps estimate an arbitrarily accurate loss over either split using many batches
# @torch.no_grad()
# def estimate_loss():
#     out = {}
#     model.eval()
#     for split in ['train', 'val']:
#         losses = torch.zeros(eval_iters)
#         for k in range(eval_iters):
#             X, Y = get_batch(split)
#             with ctx:
#                 logits, loss = model(X, Y)
#             losses[k] = loss.item()
#         out[split] = losses.mean()
#     model.train()
#     return out
# Modified estimate_loss function using DataLoaderLite
@torch.no_grad()
def estimate_loss():
    """Estimate the loss over the validation set using the new DataLoaderLite."""
    out = {}
    model.eval()  # Set model to evaluation mode
    for split in ['train', 'val']:
        loader = train_loader if split == 'train' else val_loader
        losses = torch.zeros(eval_iters)
        loader.reset()  # Reset the data loader for each split
        for k in range(eval_iters):
            X, Y = loader.next_batch()  # Use DataLoaderLite to fetch batches
            X, Y = X.to(device), Y.to(device)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # Set model back to training mode
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# logging
if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
# X, Y = get_batch('train')  # fetch the very first batch
X, Y = train_loader.next_batch()
X, Y = X.to(device), Y.to(device)
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0
clip_time = 0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:

        # val_loader.reset()
        losses = estimate_loss()
        # print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        log_text2 = f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, B_tok: {billion_tokens_processed:.6f}"
        with open("/logs/large/val_full.txt", "a") as log_file:
            log_file.write(log_text2 + "\n")
        print(log_text2)
        # checkpoint = {
        #         'model': raw_model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'model_args': model_args,
        #         'iter_num': iter_num,
        #         'best_val_loss': best_val_loss,
        #         'config': config,
        #     }
        # print(f"saving checkpoint to {out_dir}")
        # torch.save(checkpoint, os.path.join(out_dir, 'ckpt_' + str(iter_num) + '.pt'))
        # print(log_text)
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu * 100,  # convert to percentage
            }, step=iter_num)

        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt_full.pt'))
        # if iter_num % (eval_interval * 10) == 0:
        #     checkpoint = {
        #         'model': raw_model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'model_args': model_args,
        #         'iter_num': iter_num,
        #         'best_val_loss': best_val_loss,
        #         'config': config,
        #     }
        #     print(f"saving checkpoint to {out_dir}")
        #     torch.save(checkpoint, os.path.join(out_dir, 'ckpt_' + str(iter_num) + '.pt'))
    if iter_num % eval_interval == 0 and master_process:
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Details of Glycemic Index (GI) The GI Scale The glycemic index")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen)  # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :]  # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")
    model.train()

    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        # X, Y = get_batch('train')
        X, Y = train_loader.next_batch()
        X, Y = X.to(device), Y.to(device)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if total_norm.item() > grad_clip:
            clip_time += 1
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item()  # loss as float. note: this is a CPU-GPU sync point
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

        # Calculate tokens processed
        tokens_processed = train_loader.B * train_loader.T * gradient_accumulation_steps * (
            torch.distributed.get_world_size() if ddp else 1)
        tokens_per_sec = tokens_processed / dt
        billion_tokens_processed = tokens_processed / 1e9  # Convert to billion tokens

        # print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%, tok/sec: {tokens_per_sec:.2f}, B_tok: {billion_tokens_processed:.6f}")
        log_text1 = f"step {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%, tok/sec: {tokens_per_sec:.2f}"

        with open("/logs/large/train_full.txt", "a") as log_file:
            log_file.write(log_text1 + "\n")
        print(log_text1)

        # print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")
        # log_text = f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        #
        # with open("logs/small/small_log.txt", "a") as log_file:
        #     log_file.write(log_text + "\n")

        # with open("logs/small/train3-log/training_val.txt", "a") as log_file:
        #     log_file.write(log_text + "\n")
        # print(log_text)
        params = []
        for (name, p) in model.named_parameters():
            params.append(p)
        total_param_norm = 0
        for p in params:
            param_norm = p.data.norm(2)
            total_param_norm += param_norm.item() ** 2
        total_param_norm = total_param_norm ** 0.5
        momentum_norm = 0
        LL = len(optimizer.state_dict()['state'])
        for jj in range(LL):
            momentum_norm += (optimizer.state_dict()['state'][jj]['exp_avg'].detach().norm(2)) ** 2
        momentum_norm = torch.sqrt(momentum_norm).item()


        def generate_log_message(iter_num, lossf, lr, total_param_norm, momentum_norm, clip_time):
            log_message = (
                f"iter: {iter_num}, "
                f"train/loss: {lossf}, "
                f"val/loss: {losses['val']},"
                f"lr: {lr}, "
                f"param_norm: {total_param_norm}, "
                f"momentum_norm: {momentum_norm}, "
                f"train/clip_rate: {clip_time / (iter_num + 1)}"
            )
            return log_message


        # During training:
        log_message = generate_log_message(iter_num, lossf, lr, total_param_norm, momentum_norm, clip_time)

        # Print the log message to console
        # print(log_message)
        # append the log message to the log file
        # with open("logs/small/small_log.txt", "a") as log_file:
        #     log_file.write(log_message + "\n")

        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": lossf,
                "lr": lr,
                "param_norm": total_param_norm,
                "momentum_norm": momentum_norm,
                "train/clip_rate": clip_time / (iter_num + 1)
            }, step=iter_num)
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()