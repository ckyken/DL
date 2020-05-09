import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, rnn_module: nn.Module):
        super(Classifier, self).__init__()
        self.rnn_module = rnn_module
        self.rnn = rnn_module(input_size, hidden_size, bias=True)
        self.dense = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, x):
        """
        x: (seq_len, batch_size, 1)
        """

        if self.rnn_module == nn.LSTM:
            _, (hn, _) = self.rnn(x)
        else:
            _, hn = self.rnn(x)
        return torch.sigmoid(self.dense(hn).reshape(-1))
