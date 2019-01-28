'''
Techniques to use:
Dueling DQN
Double DQN
Replay Buffer
# Multi Steps
'''
from argparse import ArgumentParser

import numpy as np
import torch
from numpy import random
from torch import cuda, nn, optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset

from basic_models import AttnDecoder as Decoder
from basic_models import Encoder
from utils import AmazonReviewDataset as ARDataset


class Storage:
    def __init__(self):
        pass

    def save(self, data):
        pass

    def erase(self):
        pass


class Q(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        pass

    def synchronise(self, Qfunc):
        self.load_state_dict(Qfunc.state_dict())
        return self


class D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        pass


class R(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        pass


class L:
    def __init__(self, E):
        self.E = E

    def sample(self):
        return random.poisson(self.E)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('epochs', type=int)
    parser.add_argument('json', type=str)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--timesteps', type=int, default=120)
    parser.add_argument('--threshold', type=int, default=500)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--decay', type=float, default=.25)
    parser.add_argument('--epsilon', type=float, default=.1)
    parser.add_argument('--off_gpu', action='store_true')
    args = parser.parse_args()

    json_file = args.json
    batch_size = args.batch
    timesteps = args.timesteps
    threshold = args.threshold
    device = args.device if cuda.is_available() else 'cpu'
    on_gpu = not args.off_gpu

    dataset = ARDataset(filename=json_file,
                        threshold=threshold,
                        batch_size=batch_size,
                        device=device,
                        on_gpu=on_gpu)

    SOS = dataset.to_index('__SOS__')

    data_loader = DataLoader(dataset,
                             batch_size=timesteps,
                             shuffle=True,
                             drop_last=True)

    Qfunction = Q()
    Qtarget = Q().synchronise(Qfunction)
