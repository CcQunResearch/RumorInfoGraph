# -*- coding: utf-8 -*-
# @Time    : 2022/4/21 12:02
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : str_utils.py
# @Software: PyCharm
# @Note    :
import re

def clean_comment(comment_text):
    match_res = re.match('回复@.*?:', comment_text)
    if match_res:
        return comment_text[len(match_res.group()):]
    else:
        return comment_text