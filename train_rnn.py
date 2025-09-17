import random
import time

import click
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from next_byte_dataset import NextByteDataset, batchify, read_corpus_bytes


class KarpathyLSTM(nn.Module):

    def __init__(self, vocab_size, hidden_size, num_layers, dropout):
        super().__init__()

        self.vocab_size = vocab_size

        self.lstm = nn.LSTM(
            input_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.final_dropout = nn.Dropout(dropout)

        self.decoder = nn.Linear(hidden_size, vocab_size)



    def forward(self, x_ids, state=None):
        one_hot = F.one_hot(x_ids, num_classes=self.vocab_size).float()

        if state is not None:
            outputs, new_state = self.lstm(one_hot, state)
        else:
            outputs, new_state = self.lstm(one_hot)

        outputs = self.final_dropout(outputs)

        logits = self.decoder(outputs)

        return logits, new_state


def generate_sample_text(model, vocab, id_to_byte, byte_to_id, length, temperature=0):
    with torch.no_grad():
        model.eval()
        hidden_state = None
        primer = torch.tensor([[byte_to_id[random.choice(vocab)]]], dtype=torch.long)
        primer = primer.to(next(model.parameters()).device)

        y_logits, hidden_state = model(primer)
        next_id = torch.argmax(y_logits, dim=-1, keepdim=True).squeeze(-1)
        output = []
        for ii in range(length):
            y_logits, hidden_state = model(next_id, hidden_state)
            logits_last = y_logits[:, -1, :]
            if temperature == 0:
                next_id = torch.argmax(logits_last, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits_last / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            next_byte = id_to_byte[next_id.item()]
            output.append(next_byte)

    return bytes(output)



def calculate_loss(y_logits, target_y_ids):
    return F.cross_entropy(y_logits.flatten(0, 1), target_y_ids.flatten())


def train(model, vocab, id_to_byte, byte_to_id, train_batches, val_batches, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.0
    )

    for epoch in range(epochs):
        print(f"Starting epoch {epoch}")
        print("Sample text at epoch start:")
        print(repr(generate_sample_text(model, vocab, id_to_byte, byte_to_id, 100, temperature=1)))
        print("Training...")
        model.train()
        hidden_state = None
        total_train_loss = 0
        total_train_tokens = 0
        for x_ids, target_y_ids, xs, ys in tqdm(train_batches):
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
            print("Validation")
            model.eval()
            hidden_state = None
            for x_ids, target_y_ids, xs, ys in tqdm(val_batches):
                x_ids = x_ids.to(device)
                target_y_ids = target_y_ids.to(device)
                y_logits, hidden_state = model(x_ids, hidden_state)

                val_loss = calculate_loss(y_logits, target_y_ids)
                num_tokens = x_ids.numel()
                total_val_tokens += num_tokens
                total_val_loss += val_loss.item() * num_tokens

            val_per_token_loss = total_val_loss / total_val_tokens
            print(f"Epoch {epoch}, validation loss is {val_per_token_loss}")

    print("Sample text at training end:")
    print(repr(generate_sample_text(model, vocab, id_to_byte, byte_to_id, 100, temperature=1)))



VAL_BATCH_PERCENT = 5


@click.command()
@click.argument("directory")
@click.argument("seq_length", type=int)
@click.argument("batch_size", type=int)
@click.argument("epochs", type=int)
def main(directory, seq_length, batch_size, epochs):
    start = time.time()
    dataset = NextByteDataset(read_corpus_bytes(directory), seq_length)
    print(time.time() - start)

    print(len(dataset))
    print(dataset[0])
    print(dataset[1])

    batches = batchify(dataset, batch_size)

    val_batch_count = int(len(batches) * (VAL_BATCH_PERCENT / 100))
    if val_batch_count == 0:
        val_batch_count = 1
    train_batch_count = len(batches) - val_batch_count
    assert train_batch_count > 0, "Not enough data for training and validation"

    train_batches = batches[0:train_batch_count]
    val_batches = batches[train_batch_count:]

    print(f"We have {len(train_batches)} training batches and {len(val_batches)} validation batches")

    model = KarpathyLSTM(vocab_size=len(dataset.vocab), hidden_size=512, num_layers=3, dropout=0.5)

    train(model, dataset.vocab, dataset.id_to_byte, dataset.byte_to_id, train_batches, val_batches, epochs)



if __name__ == "__main__":
    main()
