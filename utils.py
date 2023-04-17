"""
utils.py
"""

import os
import random
import logging
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import torch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class CurvePlotter:
    def __init__(self, title, xlabel, ylabel, i=1):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.fignum = i
        self.fig = plt.figure(num=self.fignum, figsize=(7,5))
        self.values = defaultdict(list)

    def update_values(self, label, val):
        self.values[label].append(val)

    def plot_learning_curve(self, label):
        plt.figure(self.fignum)
        plt.plot(np.arange(len(self.values[label])), self.values[label], label=label, marker='o', markersize=2)

    def save_fig(self, save_path):
        plt.figure(self.fignum)
        plt.title(self.title)
        plt.legend()
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.grid(alpha=0.2)
        plt.savefig(save_path)
        plt.close(self.fignum)


def seed_everything(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name):
    os.makedirs(f'./results/{name}', exist_ok=True)

    logging.basicConfig(
        filename=f'./results/{name}/{name}.log',
        filemode='w',
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    logger = logging.getLogger(name)
    logging.info(f'Start {name}')

    return logger


def log_configs(configs):
    msg = '\n===== Training Configs ====='
    
    for k, v in vars(configs).items():
        msg += f'\n\t{k}: {v}'
    
    logging.info(msg)


