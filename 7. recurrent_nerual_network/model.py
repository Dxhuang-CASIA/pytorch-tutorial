import torch
import torch.nn as nn

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bidirectional):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True, bidirectional = bidirectional)
        if self.bidirectional:
            self.fc = nn.Linear(2 * hidden_size, num_classes)
        else:
            self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if self.bidirectional:
            h0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_size).to(device)
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])

        return out