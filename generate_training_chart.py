import json
import shutil
from pathlib import Path

import click

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from persistence import RunData, meta_file


def get_training_data(run):
    train_losses = []
    val_losses = []
    for item in run.checkpoints_dir.iterdir():
        if item.name in ("latest", "best"):
            continue
        meta = json.loads(meta_file(item).read_text())
        train_losses.append((meta["epoch"], meta["train_loss"]))
        val_losses.append((meta["epoch"], meta["val_loss"]))

    train_losses.sort(key=lambda x: x[0])
    val_losses.sort(key=lambda x: x[0])

    return train_losses, val_losses


def generate_training_chart(run):
    train_points, val_points = get_training_data(run)

    plt.title("TRAINING RUN LOSS")
    plt.xkcd()
    plt.rcParams['font.family'] = "xkcd"

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    train_epochs, train_losses = zip(*train_points)
    val_epochs, val_losses = zip(*val_points)
    ax.plot(train_epochs, train_losses, label="TRAINING LOSS", marker="o")
    ax.plot(val_epochs, val_losses, label="VALIDATION LOSS", marker="s")

    ax.set_title("TRAINING RUN LOSS")
    ax.set_xlabel("EPOCH")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("lOSS")
    ax.legend()

    fig.tight_layout()
    image_file = run.run_dir / "training_run.png"
    fig.savefig(image_file, bbox_inches="tight")
    plt.close(fig)

    this_dir = Path(__file__).resolve().parent
    html_source = this_dir / "templates" / "training_run.html"
    html_dest = run.run_dir / "training_run.html"
    shutil.copyfile(html_source, html_dest)


@click.command()
@click.argument("directory")
@click.argument("run_name")
def main(directory, run_name):
    run = RunData(directory, run_name)

    generate_training_chart(run)


if __name__ == "__main__":
    main()
