# -*- coding: utf-8 -*-
# @Time    : 2022/4/15 17:27
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : process.py
# @Software: PyCharm
# @Note    :
import os
from Process import process_weibo_dataset, collect_sentences, train_word2vec
from Utils import Embedding
from Main import WeiboDataset

if __name__ == '__main__':
    dirname = os.path.dirname(os.path.abspath(__file__))
    ###################
    # 1.预处理微博数据集 #
    ###################
    raw_dataset_path = os.path.join(dirname, '..', 'Data', 'Weibo', 'raw')
    output_path = os.path.join(dirname, '..', 'Data', 'Weibo', 'processed', 'label')

    process_weibo_dataset(raw_dataset_path, output_path)

    ##############
    # 2.训练词向量 #
    ##############
    dataset_path = os.path.join(dirname, '..', 'Data', 'Weibo', 'processed')
    model_path = os.path.join(dirname, '..', 'Model', 'w2v.model')
    vector_size = 100
    sentences = collect_sentences(dataset_path, 200000)
    # model = train_word2vec(sentences, vector_size)
    # model.save(model_path)

    #################
    # 3.制作PyG数据集 #
    #################
    train_path = os.path.join(dirname, '..', 'Data', 'Weibo', 'processed', 'label','train')
    val_path = os.path.join(dirname, '..', 'Data', 'Weibo', 'processed', 'label', 'val')
    test_path = os.path.join(dirname, '..', 'Data', 'Weibo', 'processed', 'label', 'test')
    unlabel_path = os.path.join(dirname, '..', 'Data', 'Weibo', 'processed', 'unlabel')

    word2vec = Embedding(model_path)

    train_dataset = WeiboDataset(train_path, word2vec)
    val_dataset = WeiboDataset(val_path, word2vec)
    test_dataset = WeiboDataset(test_path, word2vec)
    unlabel_dataset = WeiboDataset(unlabel_path, word2vec, clean=False)