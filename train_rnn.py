from pathlib import Path

import torch
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

