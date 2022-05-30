import argparse
import logging
import os.path as osp
import random

import torch

from data.pose_attr_dataset import DeepFashionAttrPoseDataset
from models import create_model
from utils.logger import get_root_logger
from utils.options import dict2str, dict_to_nonedict, parse
from utils.util import make_exp_dirs, set_random_seed


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    args = parser.parse_args()
    opt = parse(args.opt, is_train=False)

    # mkdir and loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}.log")
    logger = get_root_logger(
        logger_name='base', log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(opt))

    # convert to NoneDict, which returns None for missing keys
    opt = dict_to_nonedict(opt)

    # random seed
    seed = opt['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info(f'Random seed: {seed}')
    set_random_seed(seed)

    test_dataset = DeepFashionAttrPoseDataset(
        pose_dir=opt['pose_dir'],
        texture_ann_dir=opt['texture_ann_file'],
        shape_ann_path=opt['shape_ann_path'])
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=4, shuffle=False)
    logger.info(f'Number of test set: {len(test_dataset)}.')

    model = create_model(opt)
    _ = model.inference(test_loader, opt['path']['results_root'])


if __name__ == '__main__':
    main()
