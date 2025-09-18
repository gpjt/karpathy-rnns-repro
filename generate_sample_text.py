import click

import torch

from persistence import RunData, load_checkpoint



def sample(logits, temperature):
    logits_last = logits[:, -1, :]
    if temperature == 0.0:
        next_id = torch.argmax(logits_last, dim=-1, keepdim=True)
    else:
        assert temperature > 0
        probs = torch.softmax(logits_last / temperature, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

    return next_id


def generate_sample_text(model, tokenizer, length, temperature=0):
    assert length >= 2

    with torch.no_grad():
        model.eval()
        hidden_state = None
        primer = tokenizer.encode([tokenizer.random_vocab_byte()]).unsqueeze(0)
        primer = primer.to(next(model.parameters()).device)

        y_logits, hidden_state = model(primer)
        next_id = sample(y_logits, temperature)
        output_ids = [primer.item(), next_id.item()]
        for ii in range(length - 2):
            y_logits, hidden_state = model(next_id, hidden_state)
            next_id = sample(y_logits, temperature)
            output_ids.append(next_id.item())

    return tokenizer.decode(output_ids)


@click.command()
@click.argument("directory")
@click.argument("run_name")
@click.argument("checkpoint")
@click.option(
    "--length", "-n", type=int,
    default=100, show_default=True,
    help="Bytes to generate"
)
@click.option(
    "--temperature", "-t", type=click.FloatRange(min=0.0),
    default=0.0, show_default=True,
    help="Sampling temperature (0 = greedy)"
)
def main(directory, run_name, checkpoint, length, temperature):
    run = RunData(directory, run_name)

    model, tokenizer = load_checkpoint(run, checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    text = generate_sample_text(
        model, tokenizer,
        length=length, temperature=temperature
    )
    print(text.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
