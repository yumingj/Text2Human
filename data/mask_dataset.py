import os
import os.path
import random

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


class MaskDataset(data.Dataset):

    def __init__(self, segm_dir, ann_dir, downsample_factor=2, xflip=False):

        self._segm_path = segm_dir
        self._image_fnames = []

        self.downsample_factor = downsample_factor
        self.xflip = xflip

        # load attributes
        assert os.path.exists(f'{ann_dir}/upper_fused.txt')
        for idx, row in enumerate(
                open(os.path.join(f'{ann_dir}/upper_fused.txt'), 'r')):
            annotations = row.split()
            self._image_fnames.append(annotations[0])

    def _open_file(self, path_prefix, fname):
        return open(os.path.join(path_prefix, fname), 'rb')

    def _load_segm(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        fname = f'{fname[:-4]}_segm.png'
        with self._open_file(self._segm_path, fname) as f:
            segm = Image.open(f)
            if self.downsample_factor != 1:
                width, height = segm.size
                width = width // self.downsample_factor
                height = height // self.downsample_factor
                segm = segm.resize(
                    size=(width, height), resample=Image.NEAREST)
            segm = np.array(segm)
        # segm = segm[:, :, np.newaxis].transpose(2, 0, 1)
        return segm.astype(np.float32)

    def __getitem__(self, index):
        segm = self._load_segm(index)

        if self.xflip and random.random() > 0.5:
            segm = segm[:, ::-1].copy()

        segm = torch.from_numpy(segm).long()

        return_dict = {'segm': segm, 'img_name': self._image_fnames[index]}

        return return_dict

    def __len__(self):
        return len(self._image_fnames)
