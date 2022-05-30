import os
import os.path
import random

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


class DeepFashionAttrPoseDataset(data.Dataset):

    def __init__(self,
                 pose_dir,
                 texture_ann_dir,
                 shape_ann_path,
                 downsample_factor=2,
                 xflip=False):
        self._densepose_path = pose_dir
        self._image_fnames_target = []
        self._image_fnames = []
        self.upper_fused_attrs = []
        self.lower_fused_attrs = []
        self.outer_fused_attrs = []
        self.shape_attrs = []

        self.downsample_factor = downsample_factor
        self.xflip = xflip

        # load attributes
        assert os.path.exists(f'{texture_ann_dir}/upper_fused.txt')
        for idx, row in enumerate(
                open(os.path.join(f'{texture_ann_dir}/upper_fused.txt'), 'r')):
            annotations = row.split()
            self._image_fnames_target.append(annotations[0])
            self._image_fnames.append(f'{annotations[0].split(".")[0]}.png')
            self.upper_fused_attrs.append(int(annotations[1]))

        assert len(self._image_fnames_target) == len(self.upper_fused_attrs)

        assert os.path.exists(f'{texture_ann_dir}/lower_fused.txt')
        for idx, row in enumerate(
                open(os.path.join(f'{texture_ann_dir}/lower_fused.txt'), 'r')):
            annotations = row.split()
            assert self._image_fnames_target[idx] == annotations[0]
            self.lower_fused_attrs.append(int(annotations[1]))

        assert len(self._image_fnames_target) == len(self.lower_fused_attrs)

        assert os.path.exists(f'{texture_ann_dir}/outer_fused.txt')
        for idx, row in enumerate(
                open(os.path.join(f'{texture_ann_dir}/outer_fused.txt'), 'r')):
            annotations = row.split()
            assert self._image_fnames_target[idx] == annotations[0]
            self.outer_fused_attrs.append(int(annotations[1]))

        assert len(self._image_fnames_target) == len(self.outer_fused_attrs)

        assert os.path.exists(shape_ann_path)
        for idx, row in enumerate(open(os.path.join(shape_ann_path), 'r')):
            annotations = row.split()
            assert self._image_fnames_target[idx] == annotations[0]
            self.shape_attrs.append([int(i) for i in annotations[1:]])

    def _open_file(self, path_prefix, fname):
        return open(os.path.join(path_prefix, fname), 'rb')

    def _load_densepose(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        fname = f'{fname[:-4]}_densepose.png'
        with self._open_file(self._densepose_path, fname) as f:
            densepose = Image.open(f)
            if self.downsample_factor != 1:
                width, height = densepose.size
                width = width // self.downsample_factor
                height = height // self.downsample_factor
                densepose = densepose.resize(
                    size=(width, height), resample=Image.NEAREST)
            # channel-wise IUV order, [3, H, W]
            densepose = np.array(densepose)[:, :, 2:].transpose(2, 0, 1)
        return densepose.astype(np.float32)

    def __getitem__(self, index):
        pose = self._load_densepose(index)
        shape_attr = self.shape_attrs[index]
        shape_attr = torch.LongTensor(shape_attr)

        if self.xflip and random.random() > 0.5:
            pose = pose[:, :, ::-1].copy()

        upper_fused_attr = self.upper_fused_attrs[index]
        lower_fused_attr = self.lower_fused_attrs[index]
        outer_fused_attr = self.outer_fused_attrs[index]

        pose = pose / 12. - 1

        return_dict = {
            'densepose': pose,
            'img_name': self._image_fnames_target[index],
            'shape_attr': shape_attr,
            'upper_fused_attr': upper_fused_attr,
            'lower_fused_attr': lower_fused_attr,
            'outer_fused_attr': outer_fused_attr,
        }

        return return_dict

    def __len__(self):
        return len(self._image_fnames)
