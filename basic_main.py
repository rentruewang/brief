from argparse import ArgumentParser

import torch
from numpy import random
from torch import cuda
from torch.nn import BCELoss, CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from basic_models import Discriminator, Q_Generator, Reconstructor
from datasets import AmazonReviewDataset

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("epochs", type=int)
    parser.add_argument("json", type=str)
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=120)
    parser.add_argument("--hidden", type=int, default=400)
    parser.add_argument("--rnn_layers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--penalty", type=float, default=0.01)
    parser.add_argument("--maxlen", type=int, default=120)
    parser.add_argument("--decay", type=float, default=0.25)
    parser.add_argument("--off_gpu", action="store_true")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--predicted", type=int, default=1)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--size", type=list, default=[157, 241])
    # parser.add_argument('--Q', action='store_true')
    args = parser.parse_args()

    if args.test:

        class TestDataset(Dataset):
            def __init__(self, *size):
                self.data = torch.randint(0, 100, size=size, device=args.device)

            def __len__(self):
                return len(self.data)

            def __getitem__(self, index):
                return self.data[index]

            def to_index(self, _):
                return 0

            @property
            def size(self):
                return len(self.data) + 1

        dataset = TestDataset(*args.size)
    else:
        dataset = AmazonReviewDataset(
            filename=args.json,
            threshold=args.threshold,
            batch_size=args.batch,
            device=args.device,
            on_gpu=not args.off_gpu,
        )

    dataloader = DataLoader(
        dataset, batch_size=args.maxlen, shuffle=True, drop_last=True
    )

    sos_value = dataset.to_index("__SOS__")

    Q_func = Q_Generator(
        voc_size=dataset.size,
        hidden_size=args.hidden,
        time_steps=args.timesteps,
        sos_value=sos_value,
        max_len=args.maxlen,
        device=args.device,
        epsilon=args.epsilon,
        num_layers=args.rnn_layers,
    ).to(args.device)

    reconstructor = Reconstructor(
        voc_size=dataset.size,
        hidden_size=args.hidden,
        time_steps=args.timesteps,
        sos_value=sos_value,
        max_len=args.maxlen,
        device=args.device,
        num_layers=args.rnn_layers,
    ).to(args.device)

    discriminator = Discriminator(
        voc_size=dataset.size,
        hidden_size=args.hidden,
        device=args.device,
        num_layers=args.rnn_layers,
    ).to(args.device)

    print("Q")
    print(Q_func)
    print("R")
    print(reconstructor)
    print("D")
    print(discriminator)

    binary_crossentropy = BCELoss()
    mse_loss = MSELoss()
    categorical_crossentropy = CrossEntropyLoss()

    Q_optimizer = Adam(Q_func.parameters(), lr=args.lr)
    R_optimizer = Adam(reconstructor.parameters(), lr=args.lr)
    D_optimizer = Adam(discriminator.parameters(), lr=args.lr)

    for epoch in range(1, 1 + args.epochs):

        print("epoch {} / {}".format(epoch, args.epochs))
        avgloss = []

        for (q_input, dis_input) in zip(dataloader, dataloader):

            loss = torch.tensor(0.0, device=args.device)

            # train discriminator
            dis_output = discriminator(dis_input)
            ones = torch.ones_like(dis_output, device=args.device)

            loss += binary_crossentropy(dis_output, ones)

            (sentences, _) = Q_func(dis_input)

            dis_output = discriminator(sentences)
            zeros = torch.zeros_like(dis_output, device=args.device)

            loss += binary_crossentropy(dis_output, zeros)

            D_optimizer.zero_grad()
            loss.backward()
            D_optimizer.step()

            avgloss.append(loss.item())

            cuda.empty_cache()

            # train Q function
            (sentences, Q_values) = Q_func(q_input)

            dis_output = discriminator(sentences)
            ones = torch.ones_like(dis_output)

            for i in range(1, len(ones)):
                ones[i:] -= args.penalty

            loss = mse_loss(Q_values, ones)

            Q_optimizer.zero_grad()
            loss.backward()
            Q_optimizer.step()

            avgloss.append(loss.item())

            cuda.empty_cache()

            # train VAE
            distribution = reconstructor(F.softmax(sentences, dim=-1))

            loss = categorical_crossentropy(
                distribution.permute(1, 2, 0), sentences.permute(1, 0)
            )

            R_optimizer.zero_grad()
            Q_optimizer.zero_grad()
            loss.backward()
            R_optimizer.step()
            Q_optimizer.step()

            avgloss.append(loss.item())

            cuda.empty_cache()

        print("average loss: {}".format(sum(avgloss) / len(avgloss)))

    if not args.test:
        with torch.no_grad(), open("predict.txt", "w+") as predicted:
            for _ in range(args.predicted):
                random_index = random.randint(low=0, high=len(dataset) - args.maxlen)
                sentences = dataset[random_index : random_index + args.maxlen]
                for j in range(sentences.shape[1]):
                    input_sentence = sentences[:, j : j + 1]
                    output_sentence, _ = Q_func(input_sentence)
                    output_sentence.squeeze_()
                    words = []
                    for input_word in input_sentence:
                        word = dataset.to_word(input_word.squeeze_().item())
                        words.append(word)
                    predicted.write(" ".join(words))
                    words = []
                    for output_word in output_sentence:
                        word = dataset.to_word(output_word.squeeze_().item())
                        words.append(word)
                    predicted.wirte(" ".join(words))
                    predicted.write("------------")
