""" 유틸리티 함수 모음 """

# python
import os
import shutil
import random

# 3rd-party
import torch
import numpy as np


def seed_everything(seed):
    '''
    Seeds all the libraries for reproducability
    :param int seed: Seed
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_opts():
    if not os.path.exists('opts.py'):
        shutil.copy('opts_template.py', 'opts.py')


make_opts()
