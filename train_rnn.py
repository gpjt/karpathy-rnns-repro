import json
from datetime import datetime
from pathlib import Path

import click
from tqdm import tqdm

from safetensors.torch import save_file

import torch
import torch.nn.functional as F

from generate_sample_text import generate_sample_text
from karpathy_lstm import KarpathyLSTM
from next_byte_dataset import NextByteDataset, batchify, read_corpus_bytes


def save_checkpoint(checkpoints_dir, descriptor, model, epoch, train_loss, val_loss, is_best_epoch):
    save_dir = checkpoints_dir / f"{datetime.utcnow():%Y%m%dZ%H%M%S}-{descriptor}"
    save_dir_tmp = save_dir.with_suffix(".tmp")
    save_dir_tmp.mkdir()
    meta = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    (save_dir_tmp / "meta.json").write_text(json.dumps(meta) + "\n")

    save_file(model.state_dict(), save_dir_tmp / "model.safetensors")

    save_dir_tmp.rename(save_dir)

    symlink_target = Path(".") / save_dir.name
    if is_best_epoch:
        best_path = checkpoints_dir / "best"
        best_path.unlink(missing_ok=True)
        best_path.symlink_to(symlink_target, target_is_directory=True)

    latest_path = checkpoints_dir / "latest"
    latest_path.unlink(missing_ok=True)
    latest_path.symlink_to(symlink_target, target_is_directory=True)



def calculate_loss(y_logits, target_y_ids):
    return F.cross_entropy(y_logits.flatten(0, 1), target_y_ids.flatten())


def train(model, tokenizer, train_batches, val_batches, train_data, checkpoints_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer_class = getattr(torch.optim, train_data["optimizer"], None)
    if optimizer_class is None:
        raise Exception(f"Could not find optimizer {train_data['optimizer']}")
    optimizer = optimizer_class(
        model.parameters(),
        lr=train_data["lr"],
        weight_decay=train_data["weight_decay"],
    )

    best_val_loss = None
    for epoch in tqdm(range(train_data["epochs"]), desc="Run"):
        print(f"Starting epoch {epoch}")
        print("Sample text at epoch start:")
        print(repr(generate_sample_text(model, tokenizer, 100, temperature=1)))
        model.train()
        hidden_state = None
        total_train_loss = 0
        total_train_tokens = 0
        for x_ids, target_y_ids, xs, ys in tqdm(train_batches, desc=f"Epoch {epoch} train"):
            x_ids = x_ids.to(device)
            target_y_ids = target_y_ids.to(device)
            if hidden_state is not None:
                h_n, c_n = hidden_state
                hidden_state = (h_n.detach(), c_n.detach())
                y_logits, hidden_state = model(x_ids, hidden_state)
            else:
                y_logits, hidden_state = model(x_ids)
            train_loss = calculate_loss(y_logits, target_y_ids)
            train_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            num_tokens = x_ids.numel()
            total_train_tokens += num_tokens
            total_train_loss += train_loss.item() * num_tokens

        train_per_token_loss = total_train_loss / total_train_tokens
        print(f"Epoch {epoch}, average train loss per-token is {train_per_token_loss}")

        with torch.no_grad():
            total_val_loss = 0
            total_val_tokens = 0
            model.eval()
            hidden_state = None
            for x_ids, target_y_ids, xs, ys in tqdm(val_batches, desc=f"Epoch {epoch} validation"):
                x_ids = x_ids.to(device)
                target_y_ids = target_y_ids.to(device)
                y_logits, hidden_state = model(x_ids, hidden_state)

                val_loss = calculate_loss(y_logits, target_y_ids)
                num_tokens = x_ids.numel()
                total_val_tokens += num_tokens
                total_val_loss += val_loss.item() * num_tokens

            val_per_token_loss = total_val_loss / total_val_tokens
            print(f"Epoch {epoch}, validation loss is {val_per_token_loss}")

        is_best_epoch = True
        if best_val_loss is None:
            best_val_loss = val_per_token_loss
        elif val_per_token_loss < best_val_loss:
            best_val_loss = val_per_token_loss
        else:
            is_best_epoch = False

        save_checkpoint(
            checkpoints_dir, f"epoch-{epoch}", model,
            epoch, train_per_token_loss, val_per_token_loss, is_best_epoch
        )

    print("Sample text at training end:")
    print(repr(generate_sample_text(model, tokenizer, 100, temperature=1)))



@click.command()
@click.argument("directory")
@click.argument("run_name")
def main(directory, run_name):
    root_dir = Path(directory)
    if not root_dir.is_dir():
        raise Exception(f"Could not find directory {root_dir}")

    run_dir = root_dir / "runs" / run_name
    if not run_dir.is_dir():
        raise Exception(f"No runs directory {run_dir}")

    train_data = json.loads((run_dir / "train.json").read_text())
    model_data = json.loads((run_dir / "model.json").read_text())

    data_dir = root_dir / "data"
    if not data_dir.is_dir():
        raise Exception(f"No data directory {data_dir}")

    dataset = NextByteDataset(read_corpus_bytes(data_dir), train_data["seq_length"])
    batches = batchify(dataset, train_data["batch_size"])

    val_batch_count = int(len(batches) * (train_data["val_batch_percent"] / 100))
    if val_batch_count == 0:
        val_batch_count = 1
    train_batch_count = len(batches) - val_batch_count
    assert train_batch_count > 0, "Not enough data for training and validation"

    train_batches = batches[0:train_batch_count]
    val_batches = batches[train_batch_count:]
    print(f"We have {len(train_batches)} training batches and {len(val_batches)} validation batches")

    model = KarpathyLSTM(vocab_size=dataset.tokenizer.vocab_size, **model_data)

    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        checkpoints_dir.mkdir()

    train(model, dataset.tokenizer, train_batches, val_batches, train_data, checkpoints_dir)



if __name__ == "__main__":
    main()
