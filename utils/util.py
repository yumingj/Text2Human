import logging
import os
import random
import sys
import time
from shutil import get_terminal_size

import numpy as np
import torch

logger = logging.getLogger('base')


def make_exp_dirs(opt):
    """Make dirs for experiments."""
    path_opt = opt['path'].copy()
    if opt['is_train']:
        overwrite = True if 'debug' in opt['name'] else False
        os.makedirs(path_opt.pop('experiments_root'), exist_ok=overwrite)
        os.makedirs(path_opt.pop('models'), exist_ok=overwrite)
    else:
        os.makedirs(path_opt.pop('results_root'))


def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ProgressBar(object):
    """A progress bar which can print the progress.

    Modified from:
    https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    """

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (
            bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print(f'terminal width is too small ({terminal_width}), '
                  'please consider widen the terminal for better '
                  'progressbar visualization')
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write(f"[{' ' * self.bar_width}] 0/{self.task_num}, "
                             f'elapsed: 0s, ETA:\nStart...\n')
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write(
                '\033[J'
            )  # clean the output (remove extra chars since last display)
            sys.stdout.write(
                f'[{bar_chars}] {self.completed}/{self.task_num}, '
                f'{fps:.1f} task/s, elapsed: {int(elapsed + 0.5)}s, '
                f'ETA: {eta:5}s\n{msg}\n')
        else:
            sys.stdout.write(
                f'completed: {self.completed}, elapsed: {int(elapsed + 0.5)}s, '
                f'{fps:.1f} tasks/s')
        sys.stdout.flush()


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Imported from
    https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0  # running average = running sum / running count
        self.sum = 0  # running sum
        self.count = 0  # running count

    def update(self, val, n=1):
        # n = batch_size

        # val = batch accuracy for an attribute
        # self.val = val

        # sum = 100 * accumulative correct predictions for this attribute
        self.sum += val * n

        # count = total samples so far
        self.count += n

        # avg = 100 * avg accuracy for this attribute
        # for all the batches so far
        self.avg = self.sum / self.count
