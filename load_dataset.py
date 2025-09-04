import os
import time

import click
import torch

from torch.utils.data import Dataset


class NextByteDataset(Dataset):

    def __init__(self, full_data, seq_length):
        super().__init__()

        assert seq_length > 0, "Sequence length must be > 0"
        self.seq_length = seq_length

        self.num_sequences = (len(full_data) - 1) // self.seq_length
        assert self.num_sequences > 0, "Not enough data for any sequences"

        self.data = full_data[:(self.num_sequences * self.seq_length) + 1]

        values_used = set(self.data)
        self.id_to_byte = {ii: byte for (ii, byte) in enumerate(sorted(values_used))}
        self.byte_to_id = {
            byte: ii
            for (ii, byte) in self.id_to_byte.items()
        }
        self.data_as_ids = torch.tensor(
            [self.byte_to_id[b] for b in self.data],
            dtype=torch.long
        )


    def __len__(self):
        return self.num_sequences


    def __getitem__(self, ix):
        start = ix * self.seq_length
        end = start + self.seq_length

        xs = self.data[start:end]
        x_tensors = self.data_as_ids[start:end]

        ys = self.data[start + 1:end + 1]
        y_tensors = self.data_as_ids[start + 1:end + 1]


        return (xs, x_tensors, ys, y_tensors)



def load_dataset(directory, seq_length):
    filename = os.path.join(directory, "input.txt")
    with open(filename, "rb") as f:
        data = f.read()

    return NextByteDataset(data, seq_length)


@click.command()
@click.argument("directory")
@click.argument("seq_length", type=int)
def main(directory, seq_length):
    start = time.time()
    dataset = load_dataset(directory, seq_length)
    print(time.time() - start)

    print(len(dataset))
    print(dataset[0])


if __name__ == "__main__":
    main()
