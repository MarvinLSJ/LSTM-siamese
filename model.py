from torch import nn
from torch.autograd import Variable
import torch



class LSTMEncoder(nn.Module):
    def __init__(self, config):
        super(LSTMEncoder, self).__init__()
        self.embed_size = config['model']['embed_size']
        self.batch_size = config['model']['batch_size']
        self.hidden_size = config['model']['encoder']['hidden_size']
        self.num_layers = config['model']['encoder']['num_layers']
        self.bidir = config['model']['encoder']['bidirectional']
        self.dropout = config['model']['encoder']['dropout']

        self.embedding = config['embedding_matrix']
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, dropout=self.dropout,
                            num_layers=self.num_layers, bidirectional=self.bidir)

    def initHiddenCell(self):
        rand_hidden = Variable(torch.randn(1, self.batch_size, self.hidden_size))
        rand_cell = Variable(torch.randn(1, self.batch_size, self.hidden_size))
        return rand_hidden, rand_cell

    def forward(self, input, hidden, cell):
        input = self.embedding(input).view(1, 1, -1)
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        return output, hidden, cell


class Siamese_lstm(nn.Module):
    def __init__(self, config):
        super(Siamese_lstm, self).__init__()

        self.encoder = LSTMEncoder(config)
        self.fc_dim = config['model']['fc_dim']

        self.input_dim = 4 * self.encoder.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.fc_dim),
            nn.Linear(self.fc_dim, 2)
        )

    def forward(self, s1, s2):

        # init hidden, cell
        h1, c1 = self.encoder.initHiddenCell()
        h2, c2 = self.encoder.initHiddenCell()

        # input one by one

        for i in range(len(s1)):
            v1, h1, c1 = self.encoder(s1[i], h1, c1)

        for j in range(len(s2)):
            v2, h2, c2 = self.encoder(s2[j], h2, c2)

        # utilize these two encoded vectors
        features = torch.cat((v1, v2, torch.abs(v1 - v2), v1 * v2), 2)

        output = self.classifier(features)

        return output

