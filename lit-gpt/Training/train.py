import os
import sys
import time
import glob
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Optional, List, Dict, Tuple

import lightning as L
import torch
from lightning.fabric.strategies import XLAStrategy, FSDPStrategy
from lightning.fabric.loggers import CSVLogger

# from lightning.fabric.utilities.load import _lazy_load as lazy_load

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
print(wd)
sys.path.append(str(wd))

# from generate.base import generate
from lit_gpt.model import GPT, Config, Block
from lit_gpt.tokenizer import Tokenizer
# from lit_gpt.utils import lazy_load, check_valid_checkpoint_dir, step_csv_logger, chunked_cross_entropy
# from lit_gpt.utils2 import lazy_load
# from lit_gpt.speed_monitor import SpeedMonitorFabric as SpeedMonitor, measure_flops, estimate_flops
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    chunked_cross_entropy,
    get_default_supported_precision,
    load_checkpoint,
    num_parameters,

)

eval_interval = 1000
eval_iters = 1000
log_interval = 1
devices = 1
layer = 13
# change this value to force a maximum sequence length
override_max_seq_length = None

# Hyperparameters
learning_rate = 3e-4
batch_size = 8

micro_batch_size = 1
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
max_iters = 191000
weight_decay = 0.01
warmup_steps = 1000

# Data proportions from https://arxiv.org/pdf/2302.13971.pdf Table 1
data_config = [
    ("arxiv", 2.5),
    ("book", 4.5),
    ("c4", 15.0),
    ("cc", 67.0),
    ("github", 4.5),
    ("stackexchange", 2.0),
    ("wikipedia", 4.5),
]

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}


def setup(
        train_data_dir: Path = Path("sample/data/lit-redpajama"),
        val_data_dir: Optional[Path] = None,
        checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
        out_dir: Path = Path("sample/out"),
        precision: Optional[str] = None,
        tpu: bool = False,
):
    if precision is None:
        precision = "32-true" if tpu else "bf16-mixed"
    fabric_devices = devices
    if fabric_devices > 1:
        if tpu:
            # For multi-host TPU training, the device count for Fabric is limited to the count on a single host.
            fabric_devices = "auto"
            strategy = XLAStrategy(sync_module_states=False)
        else:
            precision = "bf16-true"
            strategy = FSDPStrategy(
                auto_wrap_policy={Block},
                activation_checkpointing_policy={Block},
                state_dict_type="full",
                limit_all_gathers=True,
            )
    else:
        strategy = "auto"

    logger = CSVLogger(out_dir.parent, out_dir.name, flush_logs_every_n_steps=log_interval)
    fabric = L.Fabric(devices="auto", strategy=strategy, precision=precision, loggers=logger)
    fabric.launch(main, train_data_dir, val_data_dir, checkpoint_dir, out_dir)


def main(fabric: L.Fabric, train_data_dir: Path, val_data_dir: Path, checkpoint_dir: Path, out_dir: Path):
    fabric.print(hparams)
    check_valid_checkpoint_dir(checkpoint_dir)

    fabric.seed_everything(1337)  # same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    config = Config.from_name(name=checkpoint_dir.name)

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=config.block_size,
        fabric=fabric,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        seed=(1337 + fabric.global_rank),
    )
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    checkpoint_path = checkpoint_dir / "lit_model.pth"
    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
    with fabric.init_module(empty_init=False):
        model = GPT(config)
        load_checkpoint(fabric, model, checkpoint_path, strict=True)

    student_config = model.config
    student_config.n_layer = layer
    print(student_config)
    student = GPT(student_config)
    teacher_layers = model.transformer.h
    student_layers = student.transformer.h
    for i in range(student_config.n_layer):
        student_layers[i].load_state_dict(teacher_layers[i].state_dict())

    # copy the lm_head
    student.lm_head.load_state_dict(model.lm_head.state_dict())
    # After adaptation, the student model is ready
    del model
    torch.cuda.empty_cache()
    model = student
    model.to(fabric.device)
    print('-----Student Model-----')
    print(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in trainable_params)
    fabric.print(f"Number of trainable parameters: {num_params:,}")
    num_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    fabric.print(f"Number of non trainable parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    model, optimizer = fabric.setup(model, optimizer)

    fabric.seed_everything(1337 + fabric.global_rank)

    train_time = time.time()
    train(fabric, model, optimizer, train_dataloader, val_dataloader, checkpoint_dir, out_dir)
    fabric.print(f"Training time: {(time.time() - train_time):.2f}s")

    # Save the final LoRA checkpoint at the end of training
    save_path = out_dir / "lit_model.pth"
    save_checkpoint(fabric, model, save_path)


def train(
        fabric: L.Fabric,
        model: GPT,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        checkpoint_dir: Path,
        out_dir: Path,

) -> None:
    tokenizer = Tokenizer(checkpoint_dir)
    with torch.device("meta"):
        meta_model = GPT(model.config)

    step_count = 0
    total_lengths = 0
    iter_num = 0
    total_t0 = time.time()

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()
    for iter_num, train_data in enumerate(train_dataloader, iter_num):

        # Check if we've reached the maximum number of iterations
        if iter_num >= max_iters:
            # breakpoint()  # Not sure why you need a breakpoint here, usually used for debugging
            break  # Exit the loop

        if step_count <= warmup_steps:
            # linear warmup
            lr = learning_rate * step_count / warmup_steps
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        iter_t0 = time.time()

        input_ids = train_data[:, 0: model.max_seq_length].contiguous()
        targets = train_data[:, 1: model.max_seq_length + 1].contiguous()

        is_accumulating = (iter_num + 1) % gradient_accumulation_iters != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = chunked_cross_entropy(logits, targets, chunk_size=0)
            fabric.backward(loss / gradient_accumulation_iters)

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
        elif fabric.device.type == "xla":
            xm.mark_step()

        t1 = time.time()
        total_lengths += input_ids.size(1)

        if iter_num % log_interval == 0:
            fabric.print(
                f"iter {iter_num} step {step_count}: loss {loss.item():.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
            )


@torch.no_grad()
def validate(fabric: L.Fabric, model: GPT, val_dataloader: DataLoader) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k, val_data in enumerate(val_dataloader):
        input_ids = val_data[:, 0: model.max_seq_length].contiguous()
        targets = val_data[:, 1: model.max_seq_length + 1].contiguous()
        logits = model(input_ids)
        losses[k] = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:], chunk_size=0)
    val_loss = losses.mean()

    model.train()
    return val_loss


def get_max_seq_length(data: List[Dict]) -> Tuple[int, int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    lengths = [len(d["input_ids"]) for d in data]
    max_seq_length = max(lengths)
    longest_seq_ix = lengths.index(max_seq_length)
    # support easy override at the top of the file
    return (
        override_max_seq_length if isinstance(override_max_seq_length, int) else max_seq_length,
        max_seq_length,
        longest_seq_ix,
    )


def save_checkpoint(fabric, model, file_path: Path):
    fabric.print(f"Saving weights to {str(file_path)!r}")
    fabric.save(file_path, {"model": model})


def create_dataloader(
        batch_size: int, block_size: int, data_dir: Path, fabric: L.Fabric, shuffle: bool = True, seed: int = 12345
) -> DataLoader:
    datasets = []
    for prefix, _ in data_config:
        filenames = glob.glob(str(data_dir / f"{prefix}*"))
        dataset = PackedDataset(
            filenames,
            n_chunks=4,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_dataloaders(
        batch_size: int,
        block_size: int,
        fabric: L.Fabric,
        train_data_dir: Path = Path("data/redpajama_sample"),
        val_data_dir: Optional[Path] = None,
        seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
    )
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
