import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.view(input.shape[0], -1)


class Permute(nn.Module):
    def __init__(self, *order):
        super.__init__()
        self.order = order

    def forward(self, input):
        return input.permute(*self.order)


class Reshape(nn.Module):
    def __init__(self, target_shape):
        super.__init__()
        self.target_shape = tuple(target_shape)

    def forward(self, input):
        return input.view(*self.target_shape)


class Encoder(nn.Module):

    def __init__(self,
                 voc_size,
                 hidden_size,
                 num_layers):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=voc_size,
            embedding_dim=hidden_size
        )
        self.rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )

    def forward(self, input, states):
        '''
        input shape: $$(timesteps, batch)$$
        states shape: $$(num_layers, batch, hidden_size)$$
        output shape: $$(timesteps, batch, hidden_size)$$
        '''

        return self.rnn(F.relu(self.embedding(input)), states)


class AttnDecoder(nn.Module):

    def __init__(self,
                 voc_size,
                 hidden_size,
                 timesteps,
                 num_layers):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=voc_size,
                                      embedding_dim=hidden_size)
        self.attn = nn.Linear(in_features=(num_layers+1)*hidden_size,
                              out_features=timesteps)
        self.combine = nn.Linear(in_features=2*hidden_size,
                                 out_features=hidden_size)
        self.rnn = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers)

        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, input, states, gru_out):
        '''
        input shape: $$(batch,)$$
        states shape: $$(num_layers*(num_directions=1), batch, hidden_size)$$
        gru_out shape: $$(timesteps, batch, (num_directions=1)*hidden_size)$$
        '''

        embedded = F.relu(self.embedding(input.view(-1, 1)))
        # shape: batch, timestep=1, hidden_size

        info = torch.cat(
            (embedded, states.view(-1, 1, self.num_layers*self.hidden_size)), dim=-1)
        # shape: batch, timestep, (num_layers+1)*hidden_size

        attn = F.relu(self.attn(info))
        # shape: batch, timestep, timesteps

        applied = torch.bmm(attn, gru_out.permute(1, 0, 2))
        # shape: batch, timestep, hidden_size

        combined = F.relu(self.combine(
            torch.cat((embedded, applied), dim=-1)))
        # shape: batch, timestep, hidden_size

        rnn_out, states = self.rnn(combined.permute(1, 0, 2), states)
        # shape: timestep, batch, hidden_size

        return rnn_out, states


class AC_Generator(nn.Module):
    def __init__(self, voc_size):
        super().__init__()

    def forward(self, input):
        pass


class Q_Generator(nn.Module):
    '''
    Techniques: dueling dqn, double dqn, experience
    '''

    def __init__(self,
                 voc_size,
                 hidden_size,
                 time_steps,
                 device,
                 num_layers=2):
        super().__init__()
        self.encoder = Encoder(voc_size,
                               hidden_size,
                               num_layers)
        self.decoder = AttnDecoder(voc_size,
                                   hidden_size,
                                   time_steps,
                                   num_layers)

    def forward(self, input):
        pass


class Reconstructor(nn.Module):
    def __init__(self, voc_size):
        super().__init__()

    def forward(self, input):
        pass


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        pass


if __name__ == "__main__":
    voc_size = 11
    hidden_size = 13
    embedding_dim = 17
    timesteps = 31
    batch = 19
    size = (timesteps, batch)
    num_layers = 3

    input_0 = torch.randint(voc_size, size)
    input_1 = torch.randint(voc_size, (1, batch))
    states = torch.zeros((num_layers, batch, hidden_size))
    enc_out = torch.randn((timesteps, batch, hidden_size))

    Enc = Encoder

    enc = Enc(voc_size, hidden_size, num_layers)

    enc_out_1, states = enc(input_0, states)

    Dec = AttnDecoder

    dec = Dec(voc_size, hidden_size, timesteps, num_layers=num_layers)
    print(dec)

    output = dec(input_1, states, enc_out)
    for _ in output:
        print(_.shape)
