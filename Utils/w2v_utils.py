# -*- coding: utf-8 -*-
# @Time    : 2022/4/18 16:33
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : w2v_utils.py
# @Software: PyCharm
# @Note    :
import torch
from gensim.models import Word2Vec


class Embedding():
    def __init__(self, w2v_path):
        self.w2v_path = w2v_path
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = self.make_embedding()

    def add_embedding(self, word):
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)

    def make_embedding(self):
        self.embedding_matrix = []
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size
        for i, word in enumerate(self.embedding.wv.key_to_index):
            # e.g. self.word2index['魯'] = 1
            # e.g. self.index2word[1] = '魯'
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding.wv.get_vector(word, norm=True))
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        self.add_embedding("<UNK>")
        print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix

    def sentence_word2idx(self, sen):
        sentence_idx = []
        for word in sen:
            if (word in self.word2idx.keys()):
                sentence_idx.append(self.word2idx[word])
            else:
                sentence_idx.append(self.word2idx["<UNK>"])
        return sentence_idx

    def get_word_embedding(self, sen):
        sentence_idx = self.sentence_word2idx(sen)
        word_embedding = self.embedding_matrix[sentence_idx]
        return word_embedding

    def get_sentence_embedding(self, sen):
        word_embedding = self.get_word_embedding(sen)
        sen_embedding = torch.sum(word_embedding, dim=0)
        return sen_embedding

    def labels_to_tensor(self, y):
        y = [int(label) for label in y]
        return torch.LongTensor(y)
