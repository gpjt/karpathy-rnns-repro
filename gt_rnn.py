import torch


class GTRNNCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_hh = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.W_xh = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.W_hy = torch.nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, x, hs=None):
        # x B,input_size
        # hs B,hidden_size
        if hs is None:
            hs = x.new_zeros(x.shape[0], self.hidden_size)
        hs = torch.tanh(self.W_hh(hs) + self.W_xh(x))
        y = self.W_hy(hs)
        return y, hs


class GTRNN(torch.nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, bias=True,
        dropout=0.0
    ):
        super().__init__()

        self.input_layer = GTRNNCell(input_size, hidden_size, bias=bias)
        self.hidden_layers = []
        for ii in range(num_layers - 1):
            self.hidden_layers.append((
                torch.nn.modules.dropout.Dropout(dropout),
                GTRNNCell(hidden_size, hidden_size, bias=bias)
            ))

    def forward(self, xs, hs=None):
        # xs B,seq_len,input_size
        # hs B,hidden_size
        batch_size, seq_length, input_size = xs.shape
        outputs = []
        for x_ix in range(seq_length):
            x = xs[:, x_ix, :]
            y, hs = self.input_layer(x)
            for (dropout, cell) in self.hidden_layers:
                y = dropout(y)
                y, hs = cell(y, hs)
            outputs.append(y)
        return torch.stack(outputs), hs


if __name__ == "__main__":
    cell = GTRNNCell(12, 23, False)
    ys, hs = cell(
        torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            ],
            dtype=torch.float
        )
    )
    print(ys)
    print(hs)

    rnn = GTRNN(2, 23, 4, True, 0.5)
    xs = torch.tensor(
        [
            [[1, 2], [3, 4], [5, 6]],
            [[6, 5], [4, 3], [2, 1]],
        ],
        dtype=torch.float
    )
    ys, hs = rnn(xs)
    print(ys, hs)
