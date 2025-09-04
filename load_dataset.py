import os
import time

import click
import torch


def load_dataset(directory):
    filename = os.path.join(directory, "input.txt")
    with open(filename, "rb") as f:
        data = f.read()

    characters = set(data)
    byte_to_id = {byte: ii for (ii, byte) in enumerate(sorted(characters))}
    data = torch.tensor([byte_to_id[byte] for byte in data], dtype=torch.long)


@click.command()
@click.argument("directory")
def main(directory):
    start = time.time()
    load_dataset(directory)
    print(time.time() - start)


if __name__ == "__main__":
    main()
