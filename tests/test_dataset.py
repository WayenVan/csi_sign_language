import unittest
from unittest import TestCase
from csi_sign_language.dataset.dictionary import Dictionary


class TestDictionary(TestCase):

    def test_fit(self):
        d = Dictionary()
        d.fit([['a', 'b', 'c'], ['d', 'e'], ['e', 'f']], special_token=('<PAD>', '<TOKEN>'))
        print(d.dictionary[0])
        self.assertListEqual(list(d.dictionary[0].values()), list(range(8)))
        self.assertListEqual(list(d.dictionary[1].values()), ['<PAD>', '<TOKEN>', 'a', 'b', 'c', 'd', 'e', 'f'])

    def test_gloss2index(self):
        d = Dictionary()
        d.fit([['a', 'b', 'c'], ['d', 'e'], ['e', 'f']], special_token=('<PAD>', '<TOKEN>'))
        data = [['a', 'b', 'c'], ['b', 'c', 'a']]
        self.assertListEqual(d.value2index(data), [[2, 3, 4], [3, 4, 2]])

if __name__ == '__main__':
    unittest.main()
