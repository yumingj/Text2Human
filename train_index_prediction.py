import argparse
import logging
import os
import os.path as osp
import random
import time

import torch

from data.segm_attr_dataset import DeepFashionAttrSegmDataset
from models import create_model
from utils.logger import MessageLogger, get_root_logger, init_tb_logger
from utils.options import dict2str, dict_to_nonedict, parse
from utils.util import make_exp_dirs


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    args = parser.parse_args()
    opt = parse(args.opt, is_train=True)

    # mkdir and loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}.log")
    logger = get_root_logger(
        logger_name='base', log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(opt))
    # initialize tensorboard logger
    tb_logger = None
    if opt['use_tb_logger'] and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir='./tb_logger/' + opt['name'])

    # convert to NoneDict, which returns None for missing keys
    opt = dict_to_nonedict(opt)

    # set up data loader
    train_dataset = DeepFashionAttrSegmDataset(
        img_dir=opt['train_img_dir'],
        segm_dir=opt['segm_dir'],
        pose_dir=opt['pose_dir'],
        ann_dir=opt['train_ann_file'],
        xflip=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=opt['batch_size'],
        shuffle=True,
        num_workers=opt['num_workers'],
        drop_last=True)
    logger.info(f'Number of train set: {len(train_dataset)}.')
    opt['max_iters'] = opt['num_epochs'] * len(
        train_dataset) // opt['batch_size']

    val_dataset = DeepFashionAttrSegmDataset(
        img_dir=opt['train_img_dir'],
        segm_dir=opt['segm_dir'],
        pose_dir=opt['pose_dir'],
        ann_dir=opt['val_ann_file'])
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=1, shuffle=False)
    logger.info(f'Number of val set: {len(val_dataset)}.')

    test_dataset = DeepFashionAttrSegmDataset(
        img_dir=opt['test_img_dir'],
        segm_dir=opt['segm_dir'],
        pose_dir=opt['pose_dir'],
        ann_dir=opt['test_ann_file'])
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=1, shuffle=False)
    logger.info(f'Number of test set: {len(test_dataset)}.')

    current_iter = 0
    best_epoch = None
    best_acc = 0

    model = create_model(opt)

    data_time, iter_time = 0, 0
    current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    for epoch in range(opt['num_epochs']):
        lr = model.update_learning_rate(epoch)

        for _, batch_data in enumerate(train_loader):
            data_time = time.time() - data_time

            current_iter += 1

            model.feed_data(batch_data)
            model.optimize_parameters()

            iter_time = time.time() - iter_time
            if current_iter % opt['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': [lr]})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            data_time = time.time()
            iter_time = time.time()

        if epoch % opt['val_freq'] == 0:
            save_dir = f'{opt["path"]["visualization"]}/valset/epoch_{epoch:03d}'  # noqa
            os.makedirs(save_dir, exist_ok=opt['debug'])
            val_acc = model.inference(val_loader, save_dir)

            save_dir = f'{opt["path"]["visualization"]}/testset/epoch_{epoch:03d}'  # noqa
            os.makedirs(save_dir, exist_ok=opt['debug'])
            test_acc = model.inference(test_loader, save_dir)

            logger.info(
                f'Epoch: {epoch}, val_acc: {val_acc: .4f}, test_acc: {test_acc: .4f}.'
            )

            if test_acc > best_acc:
                best_epoch = epoch
                best_acc = test_acc

            logger.info(f'Best epoch: {best_epoch}, '
                        f'Best test acc: {best_acc: .4f}.')

            # save model
            model.save_network(
                f'{opt["path"]["models"]}/models_epoch{epoch}.pth')


if __name__ == '__main__':
    main()
