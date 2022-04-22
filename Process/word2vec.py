# -*- coding: utf-8 -*-
# @Time    : 2022/4/15 17:23
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : word2vec.py
# @Software: PyCharm
# @Note    :
import os
import json
from gensim.models import Word2Vec
from Utils import clean_comment


def collect_sentences(dataset_path, unsup_train_size):
    """
    收集用来训练词向量的句子

    :param dataset_path: 数据集路径
    :param unsup_train_size: 用来训练词向量的无监督数据量
    :return:
    """
    train_path = os.path.join(dataset_path, 'label', 'train', 'raw')
    val_path = os.path.join(dataset_path, 'label', 'val', 'raw')
    test_path = os.path.join(dataset_path, 'label', 'test', 'raw')
    unlabel_path = os.path.join(dataset_path, 'unlabel', 'raw')

    sentences = collect_label_sentences(train_path) + collect_label_sentences(val_path) \
                + collect_label_sentences(test_path) + collect_unlabel_sentences(unlabel_path, unsup_train_size)
    return sentences


def collect_label_sentences(path):
    sentences = []
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        post = json.load(open(filepath, 'r', encoding='utf-8'))
        sentences.append(post['source']['content'])
        for commnet in post['comment']:
            sentences.append(clean_comment(commnet['content']))
    return sentences


def collect_unlabel_sentences(path, unsup_train_size):
    sentences = []
    for i, filename in enumerate(os.listdir(path)):
        if i == unsup_train_size:
            break
        filepath = os.path.join(path, filename)
        post = json.load(open(filepath, 'r', encoding='utf-8'))
        sentences.append(post['source']['content'])
        for commnet in post['comment']:
            sentences.append(clean_comment(commnet['content']))
    return sentences


def train_word2vec(sentences, vector_size):
    model = Word2Vec(sentences, vector_size=vector_size, window=5, min_count=5, workers=12, epochs=10, sg=1)
    return model
