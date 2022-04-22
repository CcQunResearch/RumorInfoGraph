# -*- coding: utf-8 -*-
# @Time    : 2022/4/15 11:26
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : rw_utils.py
# @Software: PyCharm
# @Note    :
import json


def write_json(dict, path):
    with open(path, 'w', encoding='utf-8') as file_obj:
        json.dump(dict, file_obj, indent=4, ensure_ascii=False)
