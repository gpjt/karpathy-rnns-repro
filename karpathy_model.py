import torch.nn as nn
import torch.nn.functional as F


class KarpathyModel(nn.Module):

    def __init__(
        self, vocab_size, hidden_size, num_layers, dropout,
        internal_model_class=nn.LSTM
    ):
        super().__init__()

        self.vocab_size = vocab_size

        self.internal_model = internal_model_class(
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
            outputs, new_state = self.internal_model(one_hot, state)
        else:
            outputs, new_state = self.internal_model(one_hot)

        outputs = self.final_dropout(outputs)

        logits = self.decoder(outputs)

        return logits, new_state

