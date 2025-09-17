import random

import torch


def generate_sample_text(model, vocab, id_to_byte, byte_to_id, length, temperature=0):
    with torch.no_grad():
        model.eval()
        hidden_state = None
        primer = torch.tensor([[byte_to_id[random.choice(vocab)]]], dtype=torch.long)
        primer = primer.to(next(model.parameters()).device)

        y_logits, hidden_state = model(primer)
        next_id = torch.argmax(y_logits, dim=-1, keepdim=True).squeeze(-1)
        output = []
        for ii in range(length):
            y_logits, hidden_state = model(next_id, hidden_state)
            logits_last = y_logits[:, -1, :]
            if temperature == 0:
                next_id = torch.argmax(logits_last, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits_last / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            next_byte = id_to_byte[next_id.item()]
            output.append(next_byte)

    return bytes(output)
