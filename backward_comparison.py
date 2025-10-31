import torch


seq_length = 400
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


def main():
    torch.manual_seed(42)

    lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
    lstm.eval()

    input_sequence = torch.rand(
        batch_size, seq_length, input_size, dtype=torch.float
    )

    for truncate_depth in range(1, seq_length):
        gradients = measure_total_gradients(
            lstm, input_sequence, truncate_depth
        )
        print(truncate_depth, gradients)


if __name__ == "__main__":
    main()
