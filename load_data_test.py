import time

import click
import torch


@click.command()
@click.argument("filename")
def main(filename):
    start = time.time()
    with open(filename, "rb") as f:
        data = f.read()

    characters = set(data)
    byte_to_id = {byte: ii for (ii, byte) in enumerate(sorted(characters))}
    data = torch.tensor([byte_to_id[byte] for byte in data], dtype=torch.long)
    print(time.time() - start)



if __name__ == "__main__":
    main()

