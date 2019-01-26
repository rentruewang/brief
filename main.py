from argparse import ArgumentParser

import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader

parser = ArgumentParser()
parser.add_argument('epochs', type=int, required=True)
parser.add_argument('json', type=str, required=True)
parser.add_argument('--penalty', type=float, default=.01)
parser.add_argument('--maxlen', type=int, default=120)
parser.add_argument('--decay', type=float, default=.25)
