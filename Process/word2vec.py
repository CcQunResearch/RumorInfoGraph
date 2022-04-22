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
    label_path = os.path.join(dataset_path, 'label', 'raw')
    unlabel_path = os.path.join(dataset_path, 'unlabel', 'raw')

    sentences = []
    for filename in os.listdir(label_path):
        filepath = os.path.join(label_path, filename)
        post = json.load(open(filepath, 'r', encoding='utf-8'))
        sentences.append(post['source']['content'])
        for commnet in post['comment']:
            sentences.append(clean_comment(commnet['content']))
    for i, filename in enumerate(os.listdir(unlabel_path)):
        if i == unsup_train_size:
            break
        filepath = os.path.join(unlabel_path, filename)
        post = json.load(open(filepath, 'r', encoding='utf-8'))
        sentences.append(post['source']['content'])
        for commnet in post['comment']:
            sentences.append(clean_comment(commnet['content']))
    return sentences


def train_word2vec(sentences, vector_size):
    model = Word2Vec(sentences, vector_size=vector_size, window=5, min_count=5, workers=12, epochs=10, sg=1)
    return model
