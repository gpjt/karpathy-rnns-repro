import os
import time

import click
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset


class NextByteDataset(Dataset):

    def __init__(self, data, seq_length):
        super().__init__()

        num_sequences = len(data) // seq_length
        truncated_data = data[:(num_sequences * seq_length) + 1]

        values_used = set(truncated_data)
        self.byte_to_id = {
            byte: ii
            for (ii, byte) in enumerate(sorted(values_used))
        }
        data_as_ids = [self.byte_to_id[b] for b in truncated_data]

        self.xs = []
        self.x_tensors = []
        self.ys = []
        self.y_tensors = []
        for sequence_num in range(num_sequences):
            start = sequence_num * seq_length
            x_sequence = data_as_ids[start:start + seq_length]
            y_sequence = data_as_ids[start + 1: start + seq_length + 1]

            self.xs.append(x_sequence)
            self.x_tensors.append(F.one_hot(torch.tensor(x_sequence, dtype=torch.long)))

            self.ys.append(y_sequence)
            self.y_tensors.append(F.one_hot(torch.tensor(y_sequence, dtype=torch.long)))


    def __len__(self):
        return len(self.xs)


    def __getitem__(self, ix):
        return (
            self.xs[ix], self.x_tensors[ix],
            self.ys[ix], self.y_tensors[ix],
        )



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
