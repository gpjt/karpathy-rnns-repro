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

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_layer = GTRNNCell(
            self.input_size, self.hidden_size, bias=bias
        )
        self.dropouts = torch.nn.ModuleList()
        self.hidden_layers = torch.nn.ModuleList()
        for ii in range(num_layers - 1):
            self.dropouts.append(
                torch.nn.Dropout(dropout)
            )
            self.hidden_layers.append(
                GTRNNCell(
                    self.hidden_size,
                    self.hidden_size,
                    bias=bias
                )
            )

    def _forward_one_step(self, x, hs):
        new_hs = hs.new_empty(hs.shape)
        y, layer_hs = self.input_layer(x, hs[:, 0, :])
        new_hs[:, 0, :] = layer_hs
        for layer_ix, (dropout, cell) in enumerate(zip(self.dropouts, self.hidden_layers), start=1):
            y = dropout(y)
            y, layer_hs = cell(y, hs[:, layer_ix, :])
            new_hs[:, layer_ix, :] = layer_hs
        return y, new_hs

    def forward(self, xs, hs=None):
        x_batch_size, seq_length, input_size = xs.shape
        assert input_size == self.input_size

        if hs is None:
            hs = xs.new_zeros((
                x_batch_size, self.num_layers, self.hidden_size
            ))
        else:
            # NB even with batch_first, PyTorch has the hidden
            # state as (L, B, H) -- we're keeping it consistent
            # with the batch-first inputs and outputs instead.
            # Doesn't matter for our use case
            hs_batch_size, hs_layers, hs_hidden_size = hs.shape
            assert hs_batch_size == x_batch_size
            assert hs_layers == self.num_layers
            assert hs_hidden_size == self.hidden_size

        outputs = xs.new_empty(x_batch_size, seq_length, self.hidden_size)

        for x_ix in range(seq_length):
            y, hs = self._forward_one_step(xs[:, x_ix, :], hs)
            outputs[:, x_ix, :] = y
        return outputs, hs


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
