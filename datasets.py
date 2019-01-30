'''text source: Amazon review data
'''
import json
import re
from os.path import expanduser

import numpy as np
import torch
from numpy import random
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class Pair:
    def __init__(self, threshold):
        self.threshold = threshold
        self.__reserved = {}
        self._to_index = {}
        self._to_word = ['__SOS__', '__PAD__', '__UNK__']
        self.__built = False

    def __iadd__(self, other):
        try:
            self.__reserved[other] += 1
        except KeyError:
            self.__reserved[other] = 1
        return self

    def build(self):
        if self.__built:
            return
        for key in self.__reserved.keys():
            if self.__reserved[key] >= self.threshold:
                self._to_word.append(key)
        for index, word in enumerate(self._to_word):
            self._to_index[word] = index
        self.__built = True

    @property
    def size(self):
        return len(self._to_word)

    def to_index(self, word):
        try:
            return self._to_index[word]
        except KeyError:
            return self._to_index['__UNK__']


class AmazonReviewDataset(Dataset):

    def __init__(self,
                 filename,
                 threshold,
                 batch_size,
                 device,
                 on_gpu=False):
        super().__init__()
        with open(expanduser(filename)) as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                file.seek(0)
                data = [json.loads(l) for l in file.readlines()]

        text = ' '.join(d['reviewText']
                        for d in data if len(d['reviewText']) != 0).lower()
        text = ' '.join(self.preprocess(text))
        print('processed')

        self.pair = Pair(threshold)

        for word in text.split(' '):
            self.pair += word

        self.pair.build()

        self.on_gpu = on_gpu

        self.device = device

        self.text = self.to_tensor(text, batch_size)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        if self.on_gpu:
            return self.text[index]
        else:
            return self.text[index].to(self.device)

    def preprocess(self, string):

        string = re.sub(pattern="[a-z]*&.;", repl=' ', string=string)

        replaced_chars = ["\\", "\n"]
        special_chars = ["...", ',', "'", '"', '(', ')', '.']

        for char in replaced_chars:
            string = string.replace(char, ' ')

        string = string.replace('!', '.')

        for char in special_chars:
            string = string.replace(char, ' ' + char + ' ')
        l = string.replace('  ', ' ').split('.')
        return [item.strip() + ' .' for item in l]

    def to_tensor(self, text, batch_size):
        tensor_list = []
        encoded = []
        for word in text.split(' '):
            encoded.append(self.pair.to_index(word))

        step = len(encoded) // batch_size
        for index in range(0, len(encoded) - step, step):
            tensor_list.append(encoded[index:index + step])
        tensor_list = sorted(tensor_list, key=lambda x: len(x), reverse=True)
        tensor_list = [torch.tensor(line, dtype=torch.long)
                       for line in tensor_list]

        if self.on_gpu:
            tensor_list = [tensor.to(self.device) for tensor in tensor_list]

        tensor_list = pad_sequence(
            tensor_list, padding_value=self.pair._to_index['__PAD__'])

        return tensor_list

    @property
    def threshold(self):
        return self.pair.threshold

    @property
    def size(self):
        return self.pair.size

    def to_index(self, word):
        return self.pair.to_index(word)

    def to_word(self, index):
        return self.pair._to_word[index]

    def to(self, device):
        self.device = device
        if self.on_gpu:
            self.text = self.text.to(device)


class AmazonSentenceDataset(Dataset):

    def __init__(self,
                 filename,
                 threshold,
                 batch_size,
                 timesteps,
                 device,
                 on_gpu=False):
        super().__init__()
        with open(expanduser(filename)) as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                file.seek(0)
                data = [json.loads(l) for l in file.readlines()]

        text = ' '.join(d['reviewText']
                        for d in data if len(d['reviewText']) != 0).lower()
        text = self.preprocess(text)
        print('processed')

        self.pair = Pair(threshold)

        for word in (' '.join(text)).split(' '):
            self.pair += word

        self.pair.build()

        self.on_gpu = on_gpu

        self.device = device

        self.text = self.sentence_tensor(text, timesteps, batch_size)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        if self.on_gpu:
            return self.text[index]
        else:
            return self.text[index].to(self.device)

    def preprocess(self, string):

        string = re.sub(pattern="[a-z]*&.;", repl=' ', string=string)

        replaced_chars = ["\\", "\n"]
        special_chars = ["...", ',', "'", '"', '(', ')', '.']

        for char in replaced_chars:
            string = string.replace(char, ' ')

        string = string.replace('!', '.')

        for char in special_chars:
            string = string.replace(char, ' ' + char + ' ')
        l = string.replace('  ', ' ').split('.')
        return [item.strip()+' .' for item in l]

    def sentence_tensor(self, sentences, timesteps, batch_size):

        def convert_line(line):
            l = []
            for word in line.split(' '):
                l.append(self.pair.to_index(word))
            return l

        sentences = [' '.join(s.split(' ')[:timesteps]) for s in sentences]
        sentences = sorted(sentences, key=lambda x: len(x), reverse=True)
        sentences = [convert_line(line) for line in sentences]

        sentences_t = [torch.tensor(s, dtype=torch.long) for s in sentences]

        sentences_t = pad_sequence(sentences_t,
                                   padding_value=self.pair._to_index['__PAD__'])

        tensor_list = []

        for i in range(0, int(sentences_t.shape[1])-batch_size, batch_size):
            tensor_list.append(sentences_t[:, i:i+batch_size])

        tensor_list = torch.cat(tensor_list, dim=0)

        if self.on_gpu:
            tensor_list = tensor_list.to(self.device)

        return tensor_list

    @property
    def threshold(self):
        return self.pair.threshold

    @property
    def size(self):
        return self.pair.size

    def to_index(self, word):
        return self.pair.to_index(word)

    def to_word(self, index):
        return self.pair._to_word[index]

    def to(self, device):
        self.device = device
        if self.on_gpu:
            self.text = self.text.to(device)
