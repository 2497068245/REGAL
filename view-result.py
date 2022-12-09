# -*- coding: utf-8 -*-
# @Time    : 2022/12/9 16:13
# @Author  : CX
# @Email   : 2497068245@qq.com
# @File    : view-result.py
# @Software: PyCharm
import numpy as np
import os
import sys

if __name__ == "__main__":
    print("当前工作目录:", os.path.abspath(os.getcwd()))
    os.chdir("E:\workplace\GithubDesktop\REGAL")
    print("当前工作目录:", os.path.abspath(os.getcwd()))
    result = np.load("emb/arenas990-1.emb.npy")
    print(result)
    print("读取成功")