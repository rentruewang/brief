'''
Techniques to use:
Dualing DQN
Double DQN
Replay Buffer
Multi Steps

In the module, D, G, R's submodules are to be used directly.
`forward` is implemented for prediction use.
'''
from argparse import ArgumentParser
from os import makedirs
from os.path import join

import numpy as np
import torch
from numpy import exp2, random
from torch import cuda, nn, optim
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from basic_models import AttnDecoder as Decoder
from basic_models import Encoder
from datasets import AmazonReviewDataset as ARDataset
from datasets import AmazonSentenceDataset as ASDataset


def short_reward(sentence, pad, sReward, on):
    '''
    sentence: a batch of sentences -> shape: timesteps, batch
    '''
    batch = sentence.shape[1]
    reward = torch.zeros([batch], device=on)
    pads = torch.tensor([pad] * batch, device=on)

    for i in range(int(sentence.shape[0])):
        reward += ((sentence[i] == pads).float())

    reward *= sReward
    return reward


def select_value(QOutput, indexList):

    assert int(QOutput.shape[0]) == len(indexList)

    output = []
    for batch, index in enumerate(range(len(indexList))):
        output.append(QOutput[batch, index])
    return torch.stack(output, dim=0)


def take_action(QFunc, states, epsilon, categorical=False):

    output, states = QFunc(*states)

    if random.uniform(low=0., high=1.) < epsilon():
        index = torch.randint(low=0, high=output.shape[-1],
                              size=[output.shape[0]])

    elif categorical:
        index = Categorical(F.softmax(output, dim=-1)).sample()
    else:
        index = output.argmax(-1)
    return index, states


class Storage:
    def __init__(self, decay_value):
        self.decay_value = decay_value
        self.data = []

    def save(self, cState, nState, reward):
        self.data.append([cState, nState, reward])

    def refresh(self):
        to_discard = []
        for index in range(len(self.data)):
            if random.uniform(low=0., high=1.) < self.decay_value:
                to_discard.append(index)
        for index in reversed(to_discard):
            self.data.pop(index)

    def optimize(self, QFunc, Qoptim, QEval, QlossFunc, on):
        loss = torch.tensor(0., device=on)
        for (cState, nState, reward) in self.data:
            cQValue, _ = QFunc(*cState)
            nQValue, _ = QEval(*nState)
            loss += QlossFunc(cQValue.max(-1)[0], reward + nQValue.max(-1)[0])


class Dual(nn.Module):

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.advantage = nn.Linear(in_features=in_features,
                                   out_features=out_features, bias=bias)
        self.average = nn.Linear(in_features=in_features,
                                 out_features=1, bias=bias)

    def forward(self, input):
        advantage = self.advantage(input)
        advantage -= advantage.mean()
        average = self.average(input)
        return advantage + average


class FullDecoder(nn.Module):

    def __init__(self, voc_size, decoder):
        super().__init__()
        self.decoder = decoder
        self.out_layer = nn.Linear(decoder.hidden_size, voc_size)

    def forward(self, *args, **kwargs):
        output, states = self.decoder(*args, **kwargs)
        output = self.out_layer(F.relu(output))
        return output, states


class Q(nn.Module):

    def __init__(self, dec, dual):
        super().__init__()
        self.dec = dec
        self.dual = dual

    def forward(self, input, states, gru_out):
        output, states = self.dec(input, states, gru_out)
        values = self.dual(output)
        return values, states


class G(nn.Module):
    '''
    takes an unfinished sentence, evaluate the best choice {first char: SOS}
    1 trained independently using generator as reward
    2 trained with R as a VAE module
    '''

    def __init__(self, voc_size, hidden_size, timesteps, on, sos_pad, num_layers=3):
        super().__init__()
        self.encoder = Encoder(voc_size, hidden_size, num_layers)
        self.decoder = Decoder(voc_size, hidden_size, timesteps, num_layers)
        self.dual = Dual(hidden_size, voc_size)
        self.QFunc = Q(self.decoder, self.dual)

        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.num_layers = num_layers
        self.on = on
        self.sos, self.pad = sos_pad

    def forward(self, sentence):
        '''
        sentence: a batch of sentences -> shape: timesteps, batch
        '''
        batch = sentence.shape[1]

        states = torch.zeros((self.num_layers, batch,
                              self.hidden_size), device=self.on)

        encoded, states = self.encoder(sentence, states)

        word = torch.tensor([self.sos] * batch, device=self.on)
        shortened = []

        for _ in range(self.timesteps):

            distribution, states = self.QFunc(word, states, encoded)

            word = distribution.argmax(-1)
            shortened.append(word)

        return torch.cat(shortened, dim=0)

    def synchronise(self, QFunc):
        self.load_state_dict(QFunc.state_dict())
        return self


class D(nn.Module):
    '''
    takes a sentence, determine if it's real (1) or generated (0) {first char: normal char}
    1 trained on real text
    2 trained on generated text
    '''

    def __init__(self, voc_size, hidden_size, timesteps, on, num_layers=5):
        super().__init__()
        self.encoder = Encoder(voc_size, hidden_size, num_layers)
        self.score = nn.Linear(hidden_size * (num_layers + 1), 1)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.on = on

    def forward(self, sentence):
        '''
        sentence: a batch of timesteps -> shape: timesteps, batch
        '''
        batch = sentence.shape[1]
        states = torch.zeros((self.num_layers, batch,
                              self.hidden_size), device=self.on)

        gru_out, states = self.encoder(sentence, states)

        gru_out = F.relu(gru_out[-1])

        output = torch.cat([gru_out, states.view(
            batch, (self.num_layers * self.hidden_size))], dim=-1)

        output = self.score(output)

        return F.sigmoid(output)


class R(nn.Module):
    '''
    takes a one hot encoded input [or softmax] and recreate the original input
    1 trained on data Q generates
    '''

    def __init__(self, voc_size, hidden_size, timesteps, on, sos, num_layers=3):
        super().__init__()
        self.encoder = Encoder(voc_size, hidden_size, num_layers)
        decoder = Decoder(voc_size, hidden_size, timesteps, num_layers)
        self.decoder = FullDecoder(voc_size, decoder)

        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.num_layers = num_layers
        self.on = on
        self.sos = sos

    def forward(self, shortened):
        '''
        shortened is a batch of sentence tensors -> shape: timesteps, batch
        '''
        batch = shortened.shape[1]
        states = torch.zeros((self.num_layers, batch,
                              self.hidden_size), device=self.on)
        encoded, states = self.encoder(shortened, states)

        original = []
        word = torch.tensor([self.sos] * batch, device=self.on)

        for _ in range(self.timesteps):

            distribution, states = self.decoder(word, states, encoded)

            word = distribution.argmax(-1)
            original.append(word)

        return torch.cat(original, dim=0)


class Step:
    def __init__(self, E):
        self.E = E

    def __call__(self):
        return random.poisson(self.E) + 1


class Epsilon:
    def __init__(self, epsilon_0, decay):
        self.epsilon_0 = epsilon_0
        self.decay = decay

    def __call__(self):
        self.epsilon_0 *= self.decay
        return self.epsilon_0


def train_one_batch(batched_data, D, GUpdate, GTarget, R,
                    Doptim, Goptim, Roptim,
                    DLossFunc, GLossFunc, RLossFunc,
                    epsilon, stepFunc, storage, sReward, sync=False, RTeacher=True):

    # train D
    is_real = D(batched_data.clone())
    true_value = torch.ones_like(is_real, device=D.on)

    loss = DLossFunc(is_real, true_value)

    generated = GUpdate(batched_data.clone())
    is_fake = D(generated)
    false_value = torch.zeros_like(is_fake, device=D.on)

    loss += DLossFunc(is_fake, false_value)

    Doptim.zero_grad()
    loss.backward()
    Doptim.step()

    # train R
    # teacher forcing
    batch = batched_data.shape[1]
    states = torch.zeros((R.num_layers, batch, R.hidden_size), device=R.on)

    encoded, states = R.encoder(generated, states)

    word = torch.tensor([R.sos] * batch, device=R.on)

    loss = torch.tensor(0., device=R.on)

    for batched_word in batched_data.clone():
        output, states = R.decoder(word, states, encoded)
        loss += RLossFunc(output.squeeze_(0), batched_word)
        word = batched_word if RTeacher else output.argmax(-1)

    Roptim.zero_grad()
    loss.backward()
    Roptim.step()

    if sync:
        GTarget.synchronise(GUpdate)

    all_steps = len(batched_data)

    current_step = 0

    # train G
    # decoder of G is trained with RL

    states = torch.zeros((GUpdate.num_layers, batch,
                          GUpdate.hidden_size), device=GUpdate.on)

    encoded, states = GUpdate.encoder(batched_data.clone(), states)

    word = torch.tensor([GUpdate.sos] * batch, device=GUpdate.on)
    shortened = []
    while True:
        step = stepFunc()

        if not current_step + step < all_steps:
            step = all_steps - current_step
        cState = [word, states, encoded]
        action = word
        State = cState.copy()

        for _ in range(step):
            action, states = take_action(GUpdate.QFunc, State, epsilon)
            shortened.append(action)

        nState = [action, states, encoded]
        reward = torch.tensor(0., device=GUpdate.on)
        word = action

        if not current_step + step < all_steps:
            nState = [torch.tensor([GUpdate.pad] * batch,
                                   device=GUpdate.on), states, encoded]

            shortened = torch.cat(shortened, dim=0)

            states = torch.zeros(
                (R.num_layers, batch, R.hidden_size), device=R.on)
            encoded, states = R.encoder(shortened, states)
            cross_entropy_loss = torch.tensor(0., device=R.on)

            word = torch.tensor([R.sos] * batch, device=R.on)

            for batched_word in batched_data.clone():
                output, states = R.decoder(word, states, encoded)
                cross_entropy_loss += RLossFunc(
                    output.squeeze_(0), batched_word)
                word = batched_word if RTeacher else output.argmax(-1)

            reward = D(shortened) - cross_entropy_loss +\
                short_reward(shortened, GUpdate.pad, sReward, GUpdate.on)

            break

        current_step += step

        storage.save(cState, nState, reward)

    storage.optimize(GUpdate.QFunc, Goptim,
                     GTarget.QFunc, GLossFunc, GUpdate.on)

    storage.refresh()


def predict(D=None, G=None, R=None, weight_dir=None, on='cpu'):
    if all([D, G, R]):
        pass
    elif weight_dir:
        D = torch.load(f=join(weight_dir, 'Discriminator.pth'),
                       map_location=on)
        G = torch.load(f=join(weight_dir, 'Generator.pth'),
                       map_location=on)
        R = torch.load(f=join(weight_dir, 'Reconstructor.pth'),
                       map_location=on)
    else:
        raise ValueError


def test():
    class TestDataset(Dataset):
        def __init__(self, low, high, shape):
            self.data = torch.randint(low, high, shape)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data[index]

        def to(self, device):
            self.data = self.data.to(device)
            return self
    voc_size = 233
    timesteps = 31
    batch = 311
    hidden_size = 141
    num_layers = 2
    epsilon = Epsilon(0, 0)
    storage = Storage(0)
    step = Step(.5)
    sReward = 0
    on = 'cuda'
    lr = 1e-3
    SOS = 0
    PAD = 1
    dataset = TestDataset(0, voc_size, (timesteps, batch)).to(on)
    dis = D(voc_size, hidden_size, timesteps, on=on, num_layers=1).to(on)
    gen = G(voc_size, hidden_size, timesteps, on=on,
            sos_pad=(SOS, PAD), num_layers=num_layers).to(on)
    rec = R(voc_size, hidden_size, timesteps, on=on,
            sos=SOS, num_layers=num_layers).to(on)
    do = optim.SGD(dis.parameters(), lr=lr)
    go = optim.SGD(gen.parameters(), lr=lr)
    ro = optim.SGD(rec.parameters(), lr=lr)
    train_one_batch(dataset[:], dis, gen, gen, rec,
                    do, go, ro,
                    F.binary_cross_entropy, F.mse_loss, F.cross_entropy,
                    epsilon, step, storage, sReward, 'weight')


def main():
    parser = ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, required=True)
    parser.add_argument('-j', '--json', type=str, required=True)
    parser.add_argument('-b', '--batch', type=int, default=32)
    parser.add_argument('-t', '--timesteps', type=int, default=120)
    parser.add_argument('-hi', '--hidden', type=int, default=600)
    parser.add_argument('-th', '--threshold', type=int, default=500)
    parser.add_argument('-g', '--gru', type=int, default=3)
    parser.add_argument('-lr', '--lr', type=float, default=1e-3)
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-r', '--reward', type=float, default=.005)
    parser.add_argument('-S', '--sentence', action='store_true')
    parser.add_argument('-md', '--Mdecay', type=float, default=.25)
    parser.add_argument('-ed', '--Edecay', type=float, default=.0)
    parser.add_argument('-ep', '--epsilon', type=float, default=.1)
    parser.add_argument('-c', '--categorical', action='store_true')
    parser.add_argument('-E', '--step', type=float, default=.5)
    parser.add_argument('-sy', '--sync', type=int, default=100)
    parser.add_argument('-p', '--predict', type=str, default='')
    parser.add_argument('-W', '--weight_dir', type=str, default='weight_dir')
    parser.add_argument('-sl', '--selflearn', action='store_true')
    parser.add_argument('--off_gpu', action='store_true')
    args = parser.parse_args()

    epochs = args.epochs
    json_file = args.json
    batch_size = args.batch
    hidden_size = args.hidden
    timesteps = args.timesteps
    threshold = args.threshold
    lr = args.lr
    sReward = args.reward
    step_E = args.step
    num_layers = args.gru
    device = args.device if cuda.is_available() else 'cpu'
    on_gpu = not args.off_gpu
    memory_decay = args.Mdecay
    epsilon_decay = exp2(-args.Edecay)
    epsilon_0 = args.epsilon
    categorical = args.categorical
    syncOn = args.sync
    weight_dir = args.weight_dir
    RTeacher = not args.selflearn

    if args.sentence:
        dataset = ASDataset(filename=json_file,
                            threshold=threshold,
                            batch_size=batch_size,
                            timesteps=timesteps,
                            device=device,
                            on_gpu=on_gpu)
    else:
        dataset = ARDataset(filename=json_file,
                            threshold=threshold,
                            batch_size=batch_size,
                            device=device,
                            on_gpu=on_gpu)

    SOS = dataset.to_index('__SOS__')
    PAD = dataset.to_index('__PAD__')

    voc_size = dataset.size

    data_loader = DataLoader(dataset,
                             batch_size=timesteps,
                             shuffle=True,
                             drop_last=True)

    dis = D(voc_size=voc_size,
            hidden_size=hidden_size,
            timesteps=timesteps,
            num_layers=num_layers,
            on=device)

    gen_update = G(voc_size=voc_size,
                   hidden_size=hidden_size,
                   timesteps=timesteps,
                   sos_pad=(SOS, PAD),
                   on=device,
                   num_layers=num_layers)

    gen_target = G(voc_size=voc_size,
                   hidden_size=hidden_size,
                   timesteps=timesteps,
                   sos_pad=(SOS, PAD),
                   on=device,
                   num_layers=num_layers)

    rec = R(voc_size=voc_size,
            hidden_size=hidden_size,
            timesteps=timesteps,
            on=device,
            sos=SOS,
            num_layers=num_layers)

    dis_optim = optim.RMSprop(params=dis.parameters(), lr=lr)
    gen_optim = optim.RMSprop(params=gen_update.parameters(), lr=lr)
    rec_optim = optim.RMSprop(params=rec.parameters(), lr=lr)

    dis_loss = nn.BCELoss()
    gen_loss = nn.SmoothL1Loss()
    rec_loss = nn.CrossEntropyLoss()

    epsilon = Epsilon(epsilon_0, epsilon_decay)
    step = Step(E=step_E)
    storage = Storage(memory_decay)

    makedirs(weight_dir, exist_ok=True)

    for epoch in range(1, 1 + epochs):
        print('Epoch: {}/{}'.format(epoch, epochs))
        i = 0
        for batch in data_loader:
            i %= syncOn
            sync = (i == 0)
            i += 1

            train_one_batch(batched_data=batch,
                            D=D, GUpdate=gen_update, GTarget=gen_target, R=R,
                            Doptim=dis_optim, Goptim=gen_optim, Roptim=rec_optim,
                            DLossFunc=dis_loss, GLossFunc=gen_loss, RLossFunc=rec_loss,
                            epsilon=epsilon, stepFunc=step, storage=storage,
                            sReward=sReward, sync=sync, RTeacher=RTeacher)

        torch.save(obj=dis.state_dict(),
                   f=join(weight_dir, 'Discriminator_{:03d}.pth'.format(epoch)))
        torch.save(obj=gen_update.state_dict(),
                   f=join(weight_dir, 'Generator_{:03d}.pth'.format(epoch)))
        torch.save(obj=rec.state_dict(),
                   f=join(weight_dir, 'Reconstructor_{:03d}.pth'.format(epoch)))
        # for prediction use
        torch.save(obj=dis.state_dict(),
                   f=join(weight_dir, 'Discriminator.pth'))
        torch.save(obj=gen_update.state_dict(),
                   f=join(weight_dir, 'Generator.pth'))
        torch.save(obj=rec.state_dict(),
                   f=join(weight_dir, 'Reconstructor.pth'))


# test()
main()
