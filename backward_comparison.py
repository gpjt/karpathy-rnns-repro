import click
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from karpathy_model import KarpathyModel
from next_byte_dataset import NextByteDataset, batchify, read_corpus_bytes
from persistence import RunData


def measure_total_gradients(
    model, input_sequence, target_sequence, truncate_depth
):
    model.zero_grad(set_to_none=True)
    _, seq_length = input_sequence.shape

    h = None
    all_ys = []
    for step in range(seq_length):
        x_step = input_sequence[:, step:step + 1]
        if h is None:
            y, h = model(x_step)
        else:
            y, h = model(x_step, h)
        all_ys.append(y.squeeze())

        steps_left = seq_length - (step + 1)
        if steps_left == truncate_depth:
            if isinstance(h, tuple):
                h_n, c_n = h
                h = (h_n.detach(), c_n.detach())
            else:
                h = h.detach()

    all_ys_tensor = torch.stack(all_ys, dim=1)
    all_ys_tensor = all_ys_tensor.squeeze()

    within_truncation_window_ys = all_ys_tensor[:, -truncate_depth:, :]
    within_truncation_window_targets = target_sequence[:, -truncate_depth:]

    loss = F.cross_entropy(
        within_truncation_window_ys.flatten(0, 1),
        within_truncation_window_targets.flatten()
    )
    loss.backward()

    gradients = 0
    for name, param in model.internal_model.named_parameters():
        if "weight" in name:
            gradients += param.grad.abs().norm().item()
    return gradients


def plot_backward_comparison(
    depth_vs_gradients_lstm,
    depth_vs_gradients_rnn,
):
    # Unpack (depth, value) pairs
    d_lstm, g_lstm = zip(*depth_vs_gradients_lstm)
    d_rnn, g_rnn = zip(*depth_vs_gradients_rnn)

    d_lstm = np.array(d_lstm)
    d_rnn = np.array(d_rnn)
    g_lstm = np.array(g_lstm, dtype=float)
    g_rnn = np.array(g_rnn, dtype=float)

    # XKCD style + font
    plt.xkcd()
    plt.rcParams['font.family'] = "xkcd"

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax2 = ax.twinx()  # second y-axis (right) for LSTM

    # Left axis: RNN
    rnn_line, = ax.plot(
        d_rnn, g_rnn, label="RNN", marker="s", markersize=3
    )
    ax.set_yscale("log")
    ax.set_xlabel("BACKPROP DEPTH (TIME STEPS)")
    ax.set_ylabel("GRADIENT METRIC — RNN (LOG SCALE)")
    ax.set_title("GRADIENT VS DEPTH — RNN VS LSTM (LOG SCALE)")
    ax.yaxis.grid(True, which="both", alpha=0.3)

    # Right axis: LSTM
    lstm_line, = ax2.plot(
        d_lstm, g_lstm, label="LSTM", marker="o", markersize=3
    )
    ax2.set_yscale("log")
    ax2.set_ylabel("GRADIENT METRIC — LSTM (LOG)")

    # One legend with both lines
    lines = [rnn_line, lstm_line]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="best")

    fig.tight_layout()
    fig.savefig("backward-comparison.png", bbox_inches="tight")
    plt.close(fig)


@click.command()
@click.argument("directory")
@click.argument("run_name")
def main(directory, run_name):
    run = RunData(directory, run_name)

    dataset = NextByteDataset(
        read_corpus_bytes(run.data_dir),
        run.train_data["seq_length"]
    )
    batches = batchify(dataset, run.train_data["batch_size"])

    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lstm = KarpathyModel(
        dataset.tokenizer.vocab_size, run.model_data["hidden_size"],
        run.model_data["num_layers"], dropout=0,
        internal_model_class=torch.nn.LSTM
    )
    lstm.to(device)

    rnn = KarpathyModel(
        dataset.tokenizer.vocab_size, run.model_data["hidden_size"],
        run.model_data["num_layers"], dropout=0,
        internal_model_class=torch.nn.RNN
    )
    rnn.to(device)

    x_ids, y_ids, _, __ = batches[0]
    x_ids = x_ids.to(device)
    y_ids = y_ids.to(device)

    depth_vs_gradients_lstm = []
    depth_vs_gradients_rnn = []
    for truncate_depth in tqdm(range(1, run.train_data["seq_length"])):
        lstm_gradients = measure_total_gradients(
            lstm, x_ids, y_ids, truncate_depth
        )
        depth_vs_gradients_lstm.append((truncate_depth, lstm_gradients))

        rnn_gradients = measure_total_gradients(
            rnn, x_ids, y_ids, truncate_depth
        )
        depth_vs_gradients_rnn.append((truncate_depth, rnn_gradients))

    plot_backward_comparison(depth_vs_gradients_lstm, depth_vs_gradients_rnn)


if __name__ == "__main__":
    main()
