# -*- coding: utf-8 -*-
# @Time    : 2022/4/23 9:07
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : pargs.py
# @Software: PyCharm
# @Note    :
import argparse

def pargs():
    parser = argparse.ArgumentParser(description="Preprocess")

    parser.add_argument('--no_step_one', action='store_false', help='do not step one', dest='step_one')
    parser.add_argument('--no_step_two', action='store_false', help='do not step two', dest='step_two')
    parser.add_argument('--no_step_three', action='store_false', help='do not step three', dest='step_three')

    parser.add_argument('--vector_size', type=int, help='word embedding size', default=100)
    parser.add_argument('--unsup_train_size', type=int, help='word embedding unlabel data train size', default=200000)

    args = parser.parse_args()
    return args