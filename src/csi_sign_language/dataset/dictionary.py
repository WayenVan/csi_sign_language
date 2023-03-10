import numpy as np
import json as j
from itertools import chain


class Dictionary:

    def __init__(self):
        self.__d = None
        self.__reverse_d = None

    @property
    def dictionary(self):
        return self.__d, self.__reverse_d

    @dictionary.setter
    def dictionary(self, d):
        self.__d = d
        self.__reverse_d = {v: k for k, v in d.items()}

    def fit(self, data, special_token=('<PAD>',)):
        flattened = list(chain(*data))
        unique = dict.fromkeys(flattened)
        all_tokens = list(special_token) + list(unique)
        self.dictionary = {k: v for v, k in list(zip(range(len(all_tokens)), all_tokens))}

    def value2index(self, data):
        d = self.dictionary[0]
        ret = []
        for values in data:
            ret.append([d[item] for item in values])
        return ret

    def index2value(self, data):
        d = self.dictionary[1]
        ret = []
        for indexes in data:
            ret.append([d[item] for item in indexes])
        return ret

    def load_from(self, file_name):
        with open(file_name, 'r') as f:
            d = j.load(f)
        self.dictionary = d

    def save_to(self, file_name):
        with open(file_name, 'w') as f:
            j.dump(self.dictionary[0], f)
