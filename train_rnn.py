import time
from pathlib import Path

import click

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class NextByteDataset(Dataset):

    def __init__(self, full_data, seq_length):
        super().__init__()

        assert seq_length > 0, "Sequence length must be > 0"
        self.seq_length = seq_length

        self.num_sequences = (len(full_data) - 1) // self.seq_length
        assert self.num_sequences > 0, "Not enough data for any sequences"

        self._data = full_data[:(self.num_sequences * self.seq_length) + 1]

        self.vocab = sorted(set(self._data))
        self.id_to_byte = {ii: byte for (ii, byte) in enumerate(self.vocab)}
        self.byte_to_id = {
            byte: ii
            for (ii, byte) in self.id_to_byte.items()
        }

        self._data_as_ids = torch.tensor(
            [self.byte_to_id[b] for b in self._data],
            dtype=torch.long
        )


    def __len__(self):
        return self.num_sequences


    def __getitem__(self, ix):
        start = ix * self.seq_length
        end = start + self.seq_length

        xs = self._data[start:end]
        x_ids = self._data_as_ids[start:end]

        ys = self._data[start + 1:end + 1]
        y_ids = self._data_as_ids[start + 1:end + 1]


        return x_ids, y_ids, xs, ys


def read_corpus_bytes(directory):
    path = Path(directory) / "input.txt"
    return path.read_bytes()


def batchify(dataset, batch_size):
    num_batches = len(dataset) // batch_size

    batches = []
    for batch_num in range(num_batches):
        batch_x_ids_list = []
        batch_xs_list = []
        batch_y_ids_list = []
        batch_ys_list = []
        for batch_position in range(batch_size):
            item = dataset[batch_num + batch_position * num_batches]
            x_ids, y_ids, xs, ys = item
            batch_x_ids_list.append(x_ids)
            batch_xs_list.append(xs)
            batch_y_ids_list.append(y_ids)
            batch_ys_list.append(ys)
        batches.append((
            torch.stack(batch_x_ids_list),
            torch.stack(batch_y_ids_list),
            batch_xs_list,
            batch_ys_list,
        ))
    return batches



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
        one_hot = nn.F.one_hot(x_ids, num_classes=self.vocab_size).float()

        if state is not None:
            outputs, new_state = self.lstm(one_hot, state)
        else:
            outputs, new_state = self.lstm(one_hot)

        outputs = self.final_dropout(outputs)

        logits = self.decoder(outputs)

        return logits, new_state




@click.command()
@click.argument("directory")
@click.argument("seq_length", type=int)
def main(directory, seq_length):
    start = time.time()
    dataset = NextByteDataset(read_corpus_bytes(directory), seq_length)
    print(time.time() - start)

    print(len(dataset))
    print(dataset[0])
    print(dataset[1])

    lstm = KarpathyLSTM(vocab_size=len(dataset.vocab), hidden_size=512, num_layers=3, dropout=0.5)



if __name__ == "__main__":
    main()
