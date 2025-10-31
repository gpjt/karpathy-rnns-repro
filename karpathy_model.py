import torch.nn as nn
import torch.nn.functional as F


class KarpathyModel(nn.Module):

    def __init__(self, vocab_size, hidden_size, num_layers, dropout):
        super().__init__()

        self.vocab_size = vocab_size

        self.lstm = nn.LSTM(
            input_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.final_dropout = nn.Dropout(dropout)

        self.decoder = nn.Linear(hidden_size, vocab_size)



    def forward(self, x_ids, state=None):
        one_hot = F.one_hot(x_ids, num_classes=self.vocab_size).float()

        if state is not None:
            outputs, new_state = self.lstm(one_hot, state)
        else:
            outputs, new_state = self.lstm(one_hot)

        outputs = self.final_dropout(outputs)

        logits = self.decoder(outputs)

        return logits, new_state

