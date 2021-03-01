"""
Techniques to use:
Dualing DQN
Double DQN
Replay Buffer
Multi Steps

In the module, D, G, R's submodules are to be used directly.
`forward` is implemented for prediction use.
"""
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
from datasets import AmazonFullDataset as AFDataset
from datasets import AmazonReviewDataset as ARDataset
from datasets import AmazonSentenceDataset as ASDataset


def ROUGE(inputs, reference, pad, N=1):
    """
    input and reference are both numpy arrays shape: -> timesteps
    """
    inputs = [i for i in inputs if i != pad]
    reference = [r for r in reference if r != pad]

    inp_grams = []
    ref_grams = []

    for i in range(len(inputs) - N + 1):
        inp_grams.append(inputs[i : i + N])

    for i in range(len(reference) - N + 1):
        ref_grams.append(reference[i : i + N])

    count = sum(1 for t in inp_grams if t in ref_grams)

    return count / len(ref_grams)


def short_reward(sentence, pad, sReward, on):
    """
    sentence: a batch of sentences -> shape: timesteps, batch
    """
    batch = sentence.shape[1]
    reward = torch.zeros([batch], device=on)
    pads = torch.tensor([pad] * batch, device=on)

    for i in range(int(sentence.shape[0])):
        reward += (sentence[i] == pads).float()

    reward *= sReward
    return reward


def select_value(QOutput, indexList):

    assert int(QOutput.shape[0]) == len(indexList)

    output = []
    for batch, index in enumerate(indexList):
        output.append(QOutput[batch, index])
    return torch.stack(output, dim=0)


def take_action(QFunc, states, epsilon, categorical=False):

    output, states = QFunc(*states)

    if random.uniform(low=0.0, high=1.0) < epsilon():
        index = torch.randint(low=0, high=output.shape[-1], size=[output.shape[0]])
    elif categorical:
        index = Categorical(F.softmax(output, dim=-1)).sample()
    else:
        index = output.argmax(-1)
    return index, states


class Storage:
    def __init__(self, decay_value, multistep=True, tdlambda=-1):
        self.decay_value = decay_value
        self.data = []
        self.multistep = multistep
        self.tdlambda = tdlambda
        assert tdlambda < 1

    def save(self, states, actions, rewards):
        self.data.append([states, actions, rewards])

    def refresh(self):
        to_discard = []
        for index in range(len(self.data)):
            if random.uniform(low=0.0, high=1.0) < self.decay_value:
                to_discard.append(index)
        for index in reversed(to_discard):
            self.data.pop(index)

    def optimize(self, QFunc, Qoptim, QEval, QlossFunc, on):
        if self.tdlambda >= 0:
            for S, action, reward in self.data:
                loss = torch.tensor(0.0, device=on)
                Q_values = []
                for st, ac in zip(S[:-1], action):
                    output, _ = QFunc(*st)
                    Q_values.append(select_value(output.squeeze_(0), ac))
                Q_NSval = []
                for st, ac in zip(S[1:], action):
                    output, _ = QEval(*st)
                    Q_NSval.append(output.squeeze_(0).max(-1)[0])

                coefficient = 1
                for Q, next_Q, rew in zip(Q_values[:-1], Q_NSval[:-1], reward[:-1]):
                    loss += coefficient * QlossFunc(Q, rew + next_Q)
                    coefficient *= self.tdlambda
                loss *= 1 - self.tdlambda
                loss += coefficient * QlossFunc(Q_values[-1], reward[-1] + Q_NSval[-1])

                Qoptim.zero_grad()
                loss.backward(retain_graph=True)
                # S carries information all the way back to encoder
                # S = [action, states, gru_out]
                Qoptim.step()
        elif self.multistep:
            for S, action, reward in self.data:
                output, _ = QFunc(*S[0])
                Q_value = select_value(output.squeeze_(0), action[0])
                Q_NSval, _ = QEval(*S[-1])
                Q_NSval = Q_NSval.squeeze_(0).max(-1)[0]
                r = sum(reward)
                loss = QlossFunc(Q_value, r + Q_NSval)

                Qoptim.zero_grad()
                loss.backward(retain_graph=True)
                # S carries information all the way back to encoder
                # S = [action, states, gru_out]
                Qoptim.step()
        else:
            for S, action, reward in self.data:
                loss = torch.tensor(0.0, device=on)
                Q_values = []
                for st, ac in zip(S[:-1], action):
                    output, _ = QFunc(*st)
                    Q_values.append(select_value(output.squeeze_(0), ac))
                output, _ = QEval(*S[-1])
                Q_values.append(output.squeeze_(0).max(-1)[0])

                for i, r in enumerate(reward):
                    loss += QlossFunc(Q_values[i], r + Q_values[i + 1])

                Qoptim.zero_grad()
                loss.backward(retain_graph=True)
                # S carries information all the way back to encoder
                # S = [action, states, gru_out]
                Qoptim.step()


class Dual(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.advantage = nn.Linear(
            in_features=in_features, out_features=out_features, bias=bias
        )
        self.average = nn.Linear(in_features=in_features, out_features=1, bias=bias)

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
    """
    takes an unfinished sentence, evaluate the best choice
    1 trained independently using generator as reward
    2 trained with R as a VAE module
    """

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
        """
        sentence: a batch of sentences -> shape: timesteps, batch
        """
        batch = sentence.shape[1]

        states = torch.zeros((self.num_layers, batch, self.hidden_size), device=self.on)

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
    """
    takes a sentence, determine if it's real (1) or generated (0)
    1 trained on real text
    2 trained on generated text
    """

    def __init__(self, voc_size, hidden_size, timesteps, on, num_layers=5):
        super().__init__()
        self.encoder = Encoder(voc_size, hidden_size, num_layers)
        self.score = nn.Linear(hidden_size * (num_layers + 1), 1)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.on = on

    def forward(self, sentence):
        """
        sentence: a batch of timesteps -> shape: timesteps, batch
        """
        batch = sentence.shape[1]
        states = torch.zeros((self.num_layers, batch, self.hidden_size), device=self.on)

        gru_out, states = self.encoder(sentence, states)
        gru_out = F.relu(gru_out[-1])

        output = torch.cat(
            [gru_out, states.view(batch, (self.num_layers * self.hidden_size))], dim=-1
        )
        output = self.score(output)

        return F.sigmoid(output)


class R(nn.Module):
    """
    takes a one hot encoded input [or softmax] and recreate the original input
    1 trained on data Q generates
    """

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
        """
        shortened is a batch of sentence tensors -> shape: timesteps, batch
        """
        batch = shortened.shape[1]
        states = torch.zeros((self.num_layers, batch, self.hidden_size), device=self.on)
        encoded, states = self.encoder(shortened, states)

        original = []
        word = torch.tensor([self.sos] * batch, device=self.on)

        for _ in range(self.timesteps):

            distribution, states = self.decoder(word, states, encoded)
            word = distribution.argmax(-1)
            original.append(word)

        return torch.cat(original, dim=0)


class S(nn.Module):
    """
    takes a sentence, determine its score
    1 trained on real text
    2 trained on generated text
    """

    def __init__(self, voc_size, hidden_size, timesteps, on, num_layers=5):
        super().__init__()
        self.encoder = Encoder(voc_size, hidden_size, num_layers)
        self.score = nn.Linear(hidden_size * (num_layers + 1), 5)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.on = on

    def forward(self, sentence):
        """
        sentence: a batch of timesteps -> shape: timesteps, batch
        """
        batch = sentence.shape[1]
        states = torch.zeros((self.num_layers, batch, self.hidden_size), device=self.on)

        gru_out, states = self.encoder(sentence, states)
        gru_out = F.relu(gru_out[-1])

        output = torch.cat(
            [gru_out, states.view(batch, (self.num_layers * self.hidden_size))], dim=-1
        )
        output = self.score(output)

        return F.softmax(output, -1)


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


def train_one_batch(
    batched_data,
    D,
    GUpdate,
    GTarget,
    R,
    Doptim,
    Goptim,
    Roptim,
    DLossFunc,
    GLossFunc,
    RLossFunc,
    epsilon,
    stepFunc,
    storage,
    sReward,
    sync=False,
    RTeacher=True,
    categorical=False,
):

    # train D
    is_real = D(batched_data)
    true_value = torch.ones_like(is_real, device=D.on)

    loss = DLossFunc(is_real, true_value)

    generated = GUpdate(batched_data)
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
    loss = torch.tensor(0.0, device=R.on)

    for batched_word in batched_data:
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
    states = torch.zeros(
        (GUpdate.num_layers, batch, GUpdate.hidden_size), device=GUpdate.on
    )
    encoded, states = GUpdate.encoder(batched_data, states)

    word = torch.tensor([GUpdate.sos] * batch, device=GUpdate.on)
    pad = torch.tensor([GUpdate.pad] * batch, device=GUpdate.on)
    shortened = []

    while True:
        step = stepFunc()
        if current_step + step > all_steps:
            step = all_steps - current_step

        states_list, reward_list, action_list = [], [], []
        S = [word, states, encoded]
        states_list.append(S)

        for _ in range(step):
            action, states = take_action(GUpdate.QFunc, S, epsilon, categorical)
            S = [action, states, encoded]
            states_list.append(S)
            reward_list.append(torch.tensor([0.0] * batch, device=GUpdate.on))
            action_list.append(action.squeeze_(0))
            shortened.append(action)

        current_step += step
        if current_step >= all_steps:
            _, states = take_action(GUpdate.QFunc, S, epsilon, categorical)
            action_list.append(pad)
            S = [action, states, encoded]
            states_list.append(S)

            shortened = torch.stack(shortened, dim=0)
            loss = torch.tensor([0.0] * batch, device=GUpdate.on)

            s = torch.zeros((R.num_layers, batch, R.hidden_size), device=R.on)
            Rencoded, s = R.encoder(shortened, s)

            word = torch.tensor([R.sos] * batch, device=R.on)
            for B in batched_data:
                output, s = R.decoder(word, s, Rencoded)
                output.squeeze_(0)
                for i in range(batch):
                    loss[i] += RLossFunc(output[i : i + 1], B[i : i + 1])
                word = B if RTeacher else output.argmax(-1)

            realistic = D(shortened).squeeze(-1)
            short = short_reward(shortened, GUpdate.pad, sReward, GUpdate.on)
            reward_list.append(realistic + short - loss)

            storage.save(states_list, action_list, reward_list)
            storage.optimize(
                GUpdate.QFunc, Goptim, GTarget.QFunc, GLossFunc, GUpdate.on
            )

            break

        storage.save(states_list, action_list, reward_list)
        storage.optimize(GUpdate.QFunc, Goptim, GTarget.QFunc, GLossFunc, GUpdate.on)


def F_train_one_batch(
    batched_data,
    D,
    GUpdate,
    GTarget,
    R,
    S,
    Doptim,
    Goptim,
    Roptim,
    Soptim,
    DLossFunc,
    GLossFunc,
    RLossFunc,
    SLossFunc,
    epsilon,
    stepFunc,
    storage,
    sReward,
    sync=False,
    RTeacher=True,
    categorical=False,
    rouge=False,
    until=3,
):

    sentences, summaries, scores = batched_data
    """
    sentences: a batch of sentences with normal ordering -> shape: batch, timesteps
    summaries: a batch of sentences with normal ordering -> shape: batch, timesteps
    scores: a batch of scores -> shape: batch
    """
    sentences = sentences.permute(1, 0)
    summaries = summaries.permute(1, 0)

    # train D
    print("training D", end="\r")
    DLoss = torch.tensor(0.0, device=D.on)

    is_real = D(sentences)
    true_value = torch.ones_like(is_real, device=D.on)

    DLoss += DLossFunc(is_real, true_value)

    is_real = D(summaries)
    true_value = torch.ones_like(is_real, device=D.on)

    DLoss += DLossFunc(is_real, true_value)

    generated = GUpdate(sentences)
    is_fake = D(generated)
    false_value = torch.zeros_like(is_fake, device=D.on)

    DLoss += DLossFunc(is_fake, false_value)

    Doptim.zero_grad()
    DLoss.backward()
    Doptim.step()

    # train S
    print("training S", end="\r")
    SLoss = torch.tensor(0.0, device=S.on)

    predicted = S(sentences)
    SLoss += SLossFunc(predicted, scores)

    predicted = S(summaries)
    SLoss += SLossFunc(predicted, scores)

    Soptim.zero_grad()
    SLoss.backward()
    Soptim.step()

    # train R
    # teacher forcing
    print("training R", "\r")
    batch = sentences.shape[1]
    states = torch.zeros((R.num_layers, batch, R.hidden_size), device=R.on)

    encoded, states = R.encoder(generated, states)

    word = torch.tensor([R.sos] * batch, device=R.on)
    RLoss = torch.tensor(0.0, device=R.on)

    for batched_word in sentences:
        Routput, states = R.decoder(word, states, encoded)
        RLoss += RLossFunc(Routput.squeeze_(0), batched_word)
        word = batched_word if RTeacher else Routput.argmax(-1)

    Roptim.zero_grad()
    RLoss.backward()
    Roptim.step()

    if sync:
        GTarget.synchronise(GUpdate)

    all_steps = len(sentences)
    current_step = 0

    # train G
    # decoder of G is trained with RL
    print("training G", end="\r")
    states = torch.zeros(
        (GUpdate.num_layers, batch, GUpdate.hidden_size), device=GUpdate.on
    )
    encoded, states = GUpdate.encoder(sentences, states)

    word = torch.tensor([GUpdate.sos] * batch, device=GUpdate.on)
    pad = torch.tensor([GUpdate.pad] * batch, device=GUpdate.on)
    shortened = []

    while True:
        step = stepFunc()
        if current_step + step > all_steps:
            step = all_steps - current_step

        states_list, reward_list, action_list = [], [], []
        STATES = [word, states, encoded]
        states_list.append(STATES)

        for _ in range(step):
            action, states = take_action(GUpdate.QFunc, STATES, epsilon, categorical)
            STATES = [action, states, encoded]
            states_list.append(STATES)
            reward_list.append(torch.tensor([0.0] * batch, device=GUpdate.on))
            action_list.append(action.squeeze_(0))
            shortened.append(action)

        current_step += step
        if current_step >= all_steps:
            _, states = take_action(GUpdate.QFunc, STATES, epsilon, categorical)
            action_list.append(pad)
            STATES = [action, states, encoded]
            states_list.append(STATES)

            shortened = torch.stack(shortened, dim=0)
            RLoss = torch.tensor([0.0] * batch, device=GUpdate.on)

            s = torch.zeros((R.num_layers, batch, R.hidden_size), device=R.on)
            Rencoded, s = R.encoder(shortened, s)

            word = torch.tensor([R.sos] * batch, device=R.on)
            for B in sentences:
                Routput, s = R.decoder(word, s, Rencoded)
                Routput.squeeze_(0)
                for i in range(batch):
                    RLoss[i] += RLossFunc(Routput[i : i + 1], B[i : i + 1])
                word = B if RTeacher else Routput.argmax(-1)

            SLoss = torch.tensor([0.0] * batch, device=GUpdate.on)
            P_scores = S(shortened)
            for i in range(batch):
                SLoss[i] = SLossFunc(P_scores[i : i + 1], scores[i : i + 1])

            # rouge
            rouge_metrics = torch.tensor([0.0] * batch, device=GUpdate.on)
            if rouge:
                for i in range(batch):
                    sho_i = shortened[:, i].detach().numpy()
                    sum_i = summaries[:, i].detach().numpy()
                    for u in range(1, until + 1):
                        rouge_metrics[i] += (
                            ROUGE(sho_i, sum_i, pad=GUpdate.pad, N=u) / until
                        )

            realistic = D(shortened).squeeze(-1)
            short = short_reward(shortened, GUpdate.pad, sReward, GUpdate.on)
            reward_list.append(realistic + short + rouge_metrics - RLoss - SLoss)

            storage.save(states_list, action_list, reward_list)
            storage.optimize(
                GUpdate.QFunc, Goptim, GTarget.QFunc, GLossFunc, GUpdate.on
            )

            break

        storage.save(states_list, action_list, reward_list)
        storage.optimize(GUpdate.QFunc, Goptim, GTarget.QFunc, GLossFunc, GUpdate.on)


def summarize_input(G=None, weight_dir=None, on="cpu", wi_iw=None):

    print(
        "Rules: In sentence prediction, every `word` has to be separated,"
        + 'including .(periods) ,(commas), "(quotes) etc.'
    )

    if G:
        pass
    elif weight_dir:
        G = torch.load(f=join(weight_dir, "Generator.pt"), map_location=on)
    else:
        raise FileNotFoundError("File Not Found")

    if wi_iw:
        word_index, index_word = wi_iw
    else:
        word_index = torch.load(f=join(weight_dir, "to_index.pt"))
        index_word = torch.load(f=join(weight_dir, "to_word.pt"))

    def to_index(word):
        if word in word_index.keys():
            return word_index[word]
        else:
            return word_index["__UNK__"]

    def to_word(index):
        return index_word[index]

    with torch.no_grad():
        while True:
            print("Input a sentence, or type `:quit` to leave.\n")
            input_sentence = input()
            if input_sentence == ":quit":
                break
            input_sentence = [to_index(word) for word in input_sentence.split(" ")]
            l = len(input_sentence)
            if l < G.timesteps:
                for _ in range(G.timesteps - len(input_sentence)):
                    input_sentence.append(to_index("__PAD__"))

            input_tensor = torch.tensor(input_sentence, device="cpu").unsqueeze(1)
            output = G(input_tensor)
            output_sentence = output.squeeze_(1).numpy()
            output_sentence = " ".join(to_word(index) for index in output_sentence)
            print(output_sentence)


def test():
    voc_size = 233
    timesteps = 13
    batch = 311
    hidden_size = 141
    num_layers = 2
    epsilon = Epsilon(0, 0)
    storage = Storage(0, multistep=True, tdlambda=0.9)
    step = Step(1)
    sReward = 0
    on = "cuda" if cuda.is_available() else "cpu"
    print("on: {}".format(on))
    lr = 1e-3
    SOS = 0
    PAD = 1
    dis = D(voc_size, hidden_size, timesteps, on=on, num_layers=1).to(on)
    gen = G(
        voc_size,
        hidden_size,
        timesteps,
        on=on,
        sos_pad=(SOS, PAD),
        num_layers=num_layers,
    ).to(on)
    rec = R(voc_size, hidden_size, timesteps, on=on, sos=SOS, num_layers=num_layers).to(
        on
    )
    scr = S(voc_size, hidden_size, timesteps, on=on, num_layers=1).to(on)
    do = optim.SGD(dis.parameters(), lr=lr)
    go = optim.SGD(gen.parameters(), lr=lr)
    ro = optim.SGD(rec.parameters(), lr=lr)
    so = optim.SGD(scr.parameters(), lr=lr)
    print("S")
    summarize_input(
        gen,
        wi_iw=[
            {"0": 0, "1": 1, "2": 2, "__UNK__": "u", "__PAD__": 99},
            [str(i) for i in range(voc_size)],
        ],
    )

    print("F")
    F_train_one_batch(
        (
            torch.randint(0, voc_size, (batch, timesteps)).to(on),
            torch.randint(0, voc_size, (batch, timesteps)).to(on),
            torch.randint(0, 5, (batch,)).to(on),
        ),
        dis,
        gen,
        gen,
        rec,
        scr,
        do,
        go,
        ro,
        so,
        F.binary_cross_entropy,
        F.mse_loss,
        F.cross_entropy,
        F.cross_entropy,
        epsilon,
        step,
        storage,
        sReward,
        "weight",
    )

    print("N")

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

    dataset = TestDataset(0, voc_size, (timesteps, batch)).to(on)
    print(dataset[:].shape)
    train_one_batch(
        dataset[:],
        dis,
        gen,
        gen,
        rec,
        do,
        go,
        ro,
        F.binary_cross_entropy,
        F.mse_loss,
        F.cross_entropy,
        epsilon,
        step,
        storage,
        sReward,
        "weight",
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, required=True)
    parser.add_argument("-j", "--json", type=str, required=True)
    parser.add_argument("-b", "--batch", type=int, default=32)
    parser.add_argument("-t", "--timesteps", type=int, default=120)
    parser.add_argument("-hi", "--hidden", type=int, default=600)
    parser.add_argument("-th", "--threshold", type=int, default=500)
    parser.add_argument("-g", "--gru", type=int, default=3)
    parser.add_argument("-lr", "--lr", type=float, default=1e-3)
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument("-r", "--reward", type=float, default=0.005)
    parser.add_argument("-S", "--sentence", action="store_true")
    parser.add_argument("-F", "--full", action="store_true")
    parser.add_argument("-md", "--Mdecay", type=float, default=0.25)
    parser.add_argument("-ed", "--Edecay", type=float, default=0.0)
    parser.add_argument("-ep", "--epsilon", type=float, default=0.1)
    parser.add_argument("-c", "--categorical", action="store_true")
    parser.add_argument("-E", "--step", type=float, default=0.5)
    parser.add_argument("-sy", "--sync", type=int, default=100)
    parser.add_argument("-p", "--predict", action="store_true")
    parser.add_argument("-W", "--weight_dir", type=str, default="weight_dir")
    parser.add_argument("-sl", "--selflearn", action="store_true")
    parser.add_argument("-ss", "--singlestep", action="store_true")
    parser.add_argument("-tdl", "--tdlambda", type=float, default=-1)
    parser.add_argument("-ro", "--rouge", action="store_true")
    parser.add_argument("--off_gpu", action="store_true")
    args = parser.parse_args()

    epochs = args.epochs
    json_file = args.json
    batch_size = args.batch
    hidden_size = args.hidden
    timesteps = args.timesteps
    threshold = args.threshold
    full = args.full
    lr = args.lr
    sReward = args.reward
    step_E = args.step
    num_layers = args.gru
    device = args.device if cuda.is_available() else "cpu"
    on_gpu = not args.off_gpu
    memory_decay = args.Mdecay
    epsilon_decay = exp2(-args.Edecay)
    epsilon_0 = args.epsilon
    categorical = args.categorical
    syncOn = args.sync
    multistep = not args.singlestep
    tdlambda = args.tdlambda
    weight_dir = args.weight_dir
    RTeacher = not args.selflearn
    rouge = args.rouge
    predict = args.predict

    if full:
        """
        if full:
            data is not transposed: shape -> batch, timesteps
        else:
            data is transposed: shape -> timesteps, batch
        """

        dataset = AFDataset(
            filename=json_file,
            threshold=threshold,
            timesteps=timesteps,
            device=device,
            on_gpu=on_gpu,
        )
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
    else:
        if args.sentence:
            dataset = ASDataset(
                filename=json_file,
                threshold=threshold,
                batch_size=batch_size,
                timesteps=timesteps,
                device=device,
                on_gpu=on_gpu,
            )
        else:
            dataset = ARDataset(
                filename=json_file,
                threshold=threshold,
                batch_size=batch_size,
                device=device,
                on_gpu=on_gpu,
            )
        data_loader = DataLoader(
            dataset, batch_size=timesteps, shuffle=True, drop_last=True
        )

    SOS = dataset.to_index("__SOS__")
    PAD = dataset.to_index("__PAD__")

    torch.save(obj=dataset.pair._to_index, f=join(weight_dir, "to_index.pt"))
    torch.save(obj=dataset.pair._to_word, f=join(weight_dir, "to_word.pt"))

    voc_size = dataset.size

    dis = D(
        voc_size=voc_size,
        hidden_size=hidden_size,
        timesteps=timesteps,
        num_layers=num_layers,
        on=device,
    ).to(device)

    gen_update = G(
        voc_size=voc_size,
        hidden_size=hidden_size,
        timesteps=timesteps,
        sos_pad=(SOS, PAD),
        on=device,
        num_layers=num_layers,
    ).to(device)

    gen_target = G(
        voc_size=voc_size,
        hidden_size=hidden_size,
        timesteps=timesteps,
        sos_pad=(SOS, PAD),
        on=device,
        num_layers=num_layers,
    ).to(device)

    rec = R(
        voc_size=voc_size,
        hidden_size=hidden_size,
        timesteps=timesteps,
        on=device,
        sos=SOS,
        num_layers=num_layers,
    ).to(device)

    dis_optim = optim.RMSprop(params=dis.parameters(), lr=lr)
    gen_optim = optim.RMSprop(params=gen_update.parameters(), lr=lr)
    rec_optim = optim.RMSprop(params=rec.parameters(), lr=lr)

    dis_loss = nn.BCELoss()
    gen_loss = nn.SmoothL1Loss()
    rec_loss = nn.CrossEntropyLoss()

    if full:
        scr = S(
            voc_size=voc_size,
            hidden_size=hidden_size,
            timesteps=timesteps,
            num_layers=num_layers,
            on=device,
        ).to(device)
        scr_optim = optim.RMSprop(params=scr.parameters(), lr=lr)
        scr_loss = nn.CrossEntropyLoss()

    epsilon = Epsilon(epsilon_0, epsilon_decay)
    step = Step(E=step_E)
    storage = Storage(memory_decay, multistep=multistep, tdlambda=tdlambda)

    makedirs(weight_dir, exist_ok=True)

    for epoch in range(1, 1 + epochs):
        print("Epoch: {}/{}".format(epoch, epochs))
        i = 0
        for batch in data_loader:
            i %= syncOn
            sync = i == 0
            i += 1
            if full:
                F_train_one_batch(
                    batched_data=batch,
                    D=dis,
                    GUpdate=gen_update,
                    GTarget=gen_target,
                    R=rec,
                    S=scr,
                    Doptim=dis_optim,
                    Goptim=gen_optim,
                    Roptim=rec_optim,
                    Soptim=scr_optim,
                    DLossFunc=dis_loss,
                    GLossFunc=gen_loss,
                    RLossFunc=rec_loss,
                    SLossFunc=scr_loss,
                    epsilon=epsilon,
                    stepFunc=step,
                    storage=storage,
                    sReward=sReward,
                    sync=sync,
                    RTeacher=RTeacher,
                    categorical=categorical,
                    rouge=rouge,
                )
            else:
                train_one_batch(
                    batched_data=batch,
                    D=dis,
                    GUpdate=gen_update,
                    GTarget=gen_target,
                    R=rec,
                    Doptim=dis_optim,
                    Goptim=gen_optim,
                    Roptim=rec_optim,
                    DLossFunc=dis_loss,
                    GLossFunc=gen_loss,
                    RLossFunc=rec_loss,
                    epsilon=epsilon,
                    stepFunc=step,
                    storage=storage,
                    sReward=sReward,
                    sync=sync,
                    RTeacher=RTeacher,
                    categorical=categorical,
                )

        torch.save(
            obj=dis.state_dict(),
            f=join(weight_dir, "Discriminator_{:03d}.pt".format(epoch)),
        )
        torch.save(
            obj=gen_update.state_dict(),
            f=join(weight_dir, "Generator_{:03d}.pt".format(epoch)),
        )
        torch.save(
            obj=rec.state_dict(),
            f=join(weight_dir, "Reconstructor_{:03d}.pt".format(epoch)),
        )
        # for prediction use
        torch.save(obj=dis.state_dict(), f=join(weight_dir, "Discriminator.pt"))
        torch.save(obj=gen_update.state_dict(), f=join(weight_dir, "Generator.pt"))
        torch.save(obj=rec.state_dict(), f=join(weight_dir, "Reconstructor.pt"))

        if full:
            torch.save(
                obj=scr.state_dict(),
                f=join(weight_dir, "ScorePredict_{:03d}.pt".format(epoch)),
            )
            torch.save(obj=scr.state_dict(), f=join(weight_dir, "ScorePredict.pt"))

    if predict:
        summarize_input(
            G=gen_target,
            weight_dir=weight_dir,
            on="cpu",
            wi_iw=[dataset.pair._to_index, dataset.pair._to_word],
        )


main()
