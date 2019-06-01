import numpy as np
import torch
from numpy import random
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F


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


class OneHotEncoder(nn.Module):
    '''
    input is supposed to be a one-hot vector,
    or something similar like a softmax input
    '''

    def __init__(self,
                 voc_size,
                 hidden_size,
                 num_layers):
        super().__init__()
        self.embedding = nn.Linear(in_features=voc_size,
                                   out_features=hidden_size)
        self.rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )

    def forward(self, input, states):
        '''
        input shape: $$(timesteps, batch, voc_size)$$
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
        self.attn = nn.Linear(in_features=(num_layers + 1) * hidden_size,
                              out_features=timesteps)
        self.combine = nn.Linear(in_features=2 * hidden_size,
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
            (embedded, states.view(-1, 1, self.num_layers * self.hidden_size)), dim=-1)
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


class Q_Generator(nn.Module):
    '''
    Techniques: dueling dqn, epsilon-greedy
    '''

    def __init__(self,
                 voc_size,
                 hidden_size,
                 time_steps,
                 sos_value,
                 max_len,
                 device,
                 epsilon,
                 num_layers=2):
        super().__init__()
        self.encoder = Encoder(voc_size,
                               hidden_size,
                               num_layers)
        self.decoder = AttnDecoder(voc_size,
                                   hidden_size,
                                   time_steps,
                                   num_layers)
        self.average = nn.Linear(in_features=hidden_size,
                                 out_features=1,
                                 bias=False)
        self.advantage = nn.Linear(in_features=hidden_size,
                                   out_features=voc_size,
                                   bias=False)

        self.voc_size = voc_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.epsilon = epsilon
        self.sos_value = sos_value
        self.max_len = max_len

    def forward(self, input):
        '''
        Arguments:
            input {tensor} -- [long sentence]

        '''

        batch_size = input.shape[1]

        states = torch.zeros([self.num_layers, batch_size,
                              self.hidden_size], device=self.device, requires_grad=True)
        gru_out, states = self.encoder(input, states)

        current_word = torch.tensor(
            [self.sos_value] * batch_size, device=self.device)

        batched_sentences = []
        batched_Q = []

        for _ in range(self.max_len):

            rnn_out, states = self.decoder(current_word, states, gru_out)
            values = (self.average(rnn_out) +
                      self.advantage(rnn_out)).squeeze(0)

            current_word = values.argmax(-1)
            # m = Categorical(current_word)
            # current_word = m.sample()

            random_indices = (random.uniform(low=0., high=1., size=[
                              batch_size]) < self.epsilon).astype('long')
            random_indices = np.stack([random_indices] * 2, axis=-1)

            random_word_output = torch.randint(low=0, high=self.voc_size, size=[
                                               batch_size], device=self.device)

            word_selection = torch.stack(
                [current_word, random_word_output], dim=-1)

            current_word = word_selection.gather(
                dim=-1, index=torch.tensor(random_indices, device=self.device))[:, 0]

            batched_sentences.append(rnn_out)

            v_list = []
            for i, word in zip(range(len(values)), current_word):
                v_list.append(values[i, word])
            batched_Q.append(torch.stack(v_list, dim=0))

        return torch.stack(batched_sentences), torch.stack(batched_Q)


class Reconstructor(nn.Module):

    def __init__(self,
                 voc_size,
                 hidden_size,
                 time_steps,
                 sos_value,
                 max_len,
                 device,
                 num_layers=2):
        super().__init__()
        self.encoder = OneHotEncoder(voc_size,
                                     hidden_size,
                                     num_layers)
        self.decoder = AttnDecoder(voc_size,
                                   hidden_size,
                                   time_steps,
                                   num_layers)
        self.output = nn.Linear(in_features=hidden_size,
                                out_features=voc_size)

        self.voc_size = voc_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.sos_value = sos_value
        self.max_len = max_len

    def forward(self, input):
        '''
        Arguments:
            input {tensor} -- [long sentence probability]
            sos_value {int} -- [sos]
            termination {list} -- [max_len]
        '''

        batch_size = input.shape[1]

        states = torch.zeros([self.num_layers, batch_size,
                              self.hidden_size], device=self.device, requires_grad=True)
        gru_out, states = self.encoder(input, states)

        current_word = torch.tensor(
            [self.sos_value] * batch_size, device=self.device)

        batched_distribution = []

        for _ in range(self.max_len):

            rnn_out, states = self.decoder(current_word, states, gru_out)

            rnn_out = self.output(rnn_out)

            current_word = rnn_out.argmax(-1).squeeze_(0)

            batched_distribution.append(rnn_out.squeeze_(0))

        return torch.stack(batched_distribution, dim=0)


class Discriminator(nn.Module):
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
        self.gru = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers)
        self.grucell_list = [nn.GRUCell(
            input_size=hidden_size, hidden_size=hidden_size).to(device) for i in range(num_layers)]
        self.score = nn.Linear(in_features=hidden_size * num_layers,
                               out_features=1)

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

    def forward(self, input):
        '''
        input shape: $$(timesteps, batch)$$
        '''

        batch_size = input.shape[1]
        states = torch.zeros([self.num_layers, batch_size,
                              self.hidden_size], device=self.device, requires_grad=True)
        gru_out, states = self.encoder(F.relu(input), states)
        gru_out, states = self.gru(F.relu(gru_out), states)
        output_states = [[s] for s in states]
        for t_i, timestep_output in enumerate(gru_out):
            for g_i, grucell in enumerate(self.grucell_list):
                state = grucell(timestep_output, output_states[g_i][t_i])
                output_states[g_i].append(state)

        output_states = [s[1:] for s in output_states]
        output_states = [torch.stack(s, dim=0) for s in output_states]
        states = torch.cat(output_states, dim=-1)

        score = F.sigmoid(self.score(states).squeeze_())
        return score


if __name__ == "__main__":
    voc_size = 161
    hidden_size = 103
    embedding_dim = 17
    timesteps = 37
    batch = 109
    size = (timesteps, batch)
    num_layers = 3

    dis = Discriminator(voc_size, hidden_size, timesteps, 'cpu')
    input = torch.randint(voc_size, (timesteps, batch))
    print(dis(input).shape)

    # input_0 = torch.randint(voc_size, size)
    # input_1 = torch.randint(voc_size, (1, batch))
    # states = torch.zeros((num_layers, batch, hidden_size))
    # enc_out = torch.randn((timesteps, batch, hidden_size))

    # Enc = Encoder

    # enc = Enc(voc_size, hidden_size, num_layers)

    # enc_out_1, states = enc(input_0, states)

    # Dec = AttnDecoder

    # dec = Dec(voc_size, hidden_size, timesteps, num_layers=num_layers)
    # print(dec)

    # output = dec(input_1, states, enc_out)
    # for _ in output:
    #     print(_.shape)
