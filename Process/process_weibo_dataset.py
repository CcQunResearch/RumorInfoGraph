# -*- coding: utf-8 -*-
# @Time    : 2022/4/14 20:02
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : process_weibo_dataset.py
# @Software: PyCharm
# @Note    : 处理微博标注数据集
import os
import json
import shutil
from Utils import write_json


def process_weibo_dataset(dataset_path, output_path):
    """
    预处理weibo数据集，与无标注数据的格式相统一

    :param dataset_path: weibo数据集路径
    :param output_path: 输出路径
    """
    label_path = os.path.join(dataset_path, 'Weibo.txt')

    post_id_list = []
    post_label_list = []

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    else:
        shutil.rmtree(output_path)
        os.mkdir(output_path)

    f = open(label_path, 'r', encoding='utf-8')
    post_list = f.readlines()
    for post in post_list:
        post_id_list.append(post.split()[0].strip()[4:])
        post_label_list.append(int(post.split()[1].strip()[-1]))

    for i, post_id in enumerate(post_id_list):
        reverse_dict = {}
        comment_index = 0
        comment_list = []

        post_path = os.path.join(dataset_path, 'post', f'{post_id}.json')
        post = json.load(open(post_path, 'r', encoding='utf-8'))
        source = {
            'content': post[0]['text'],
            'user id': post[0]['uid'],
            'tweet id': post[0]['mid'],
            'label': post_label_list[i]
        }

        for i in range(1, len(post)):
            comment_list.append({'comment id': comment_index, 'parent': -2, 'children': []})
            reverse_dict[post[i]['mid']] = comment_index
            comment_index += 1
        for i in range(1, len(post)):
            comment_list[i - 1]['content'] = post[i]['text']
            comment_list[i - 1]['user id'] = post[i]['uid']
            comment_list[i - 1]['user name'] = post[i]['username']
            if post[i]['parent'] == source['tweet id']:
                comment_list[i - 1]['parent'] = -1
            else:
                parent_index = reverse_dict[post[i]['parent']]
                comment_list[i - 1]['parent'] = parent_index
                comment_list[parent_index]['children'].append(i - 1)

        write_json({'source': source, 'comment': comment_list}, os.path.join(output_path, f'{post_id}.json'))
