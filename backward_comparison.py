import matplotlib.pyplot as plt
import numpy as np
import torch


seq_length = 30
input_size = 3
hidden_size = 10
batch_size = 3


def measure_total_gradients(model, input_sequence, truncate_depth):
    model.zero_grad(set_to_none=True)

    h = None
    for step in range(seq_length):
        x_step = input_sequence[:, step:step + 1, :]
        if h is None:
            y, h = model(x_step)
        else:
            y, h = model(x_step, h)

        steps_left = seq_length - (step + 1)
        if steps_left == truncate_depth:
            if isinstance(h, tuple):
                h_n, c_n = h
                h = (h_n.detach(), c_n.detach())
            else:
                h = h.detach()

    loss = 1 - y.mean()
    loss.backward()

    gradients = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            gradients += param.grad.abs().mean().item()
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


def main():
    torch.manual_seed(42)

    lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
    lstm.eval()

    rnn = torch.nn.RNN(input_size, hidden_size, batch_first=True)
    rnn.eval()

    input_sequence = torch.rand(
        batch_size, seq_length, input_size, dtype=torch.float
    )

    depth_vs_gradients_lstm = []
    depth_vs_gradients_rnn = []
    for truncate_depth in range(1, seq_length):
        lstm_gradients = measure_total_gradients(
            lstm, input_sequence, truncate_depth
        )
        depth_vs_gradients_lstm.append((truncate_depth, lstm_gradients))

        rnn_gradients = measure_total_gradients(
            rnn, input_sequence, truncate_depth
        )
        depth_vs_gradients_rnn.append((truncate_depth, rnn_gradients))

    plot_backward_comparison(depth_vs_gradients_lstm, depth_vs_gradients_rnn)


if __name__ == "__main__":
    main()
