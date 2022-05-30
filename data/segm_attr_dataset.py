import os
import os.path
import random

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


class DeepFashionAttrSegmDataset(data.Dataset):

    def __init__(self,
                 img_dir,
                 segm_dir,
                 pose_dir,
                 ann_dir,
                 downsample_factor=2,
                 xflip=False):
        self._img_path = img_dir
        self._densepose_path = pose_dir
        self._segm_path = segm_dir
        self._image_fnames = []
        self.upper_fused_attrs = []
        self.lower_fused_attrs = []
        self.outer_fused_attrs = []

        self.downsample_factor = downsample_factor
        self.xflip = xflip

        # load attributes
        assert os.path.exists(f'{ann_dir}/upper_fused.txt')
        for idx, row in enumerate(
                open(os.path.join(f'{ann_dir}/upper_fused.txt'), 'r')):
            annotations = row.split()
            self._image_fnames.append(annotations[0])
            # assert self._image_fnames[idx] == annotations[0]
            self.upper_fused_attrs.append(int(annotations[1]))

        assert len(self._image_fnames) == len(self.upper_fused_attrs)

        assert os.path.exists(f'{ann_dir}/lower_fused.txt')
        for idx, row in enumerate(
                open(os.path.join(f'{ann_dir}/lower_fused.txt'), 'r')):
            annotations = row.split()
            assert self._image_fnames[idx] == annotations[0]
            self.lower_fused_attrs.append(int(annotations[1]))

        assert len(self._image_fnames) == len(self.lower_fused_attrs)

        assert os.path.exists(f'{ann_dir}/outer_fused.txt')
        for idx, row in enumerate(
                open(os.path.join(f'{ann_dir}/outer_fused.txt'), 'r')):
            annotations = row.split()
            assert self._image_fnames[idx] == annotations[0]
            self.outer_fused_attrs.append(int(annotations[1]))

        assert len(self._image_fnames) == len(self.outer_fused_attrs)

        # remove the overlapping item between upper cls and lower cls
        # cls 21 can appear with upper clothes
        # cls 4 can appear with lower clothes
        self.upper_cls = [1., 4.]
        self.lower_cls = [3., 5., 21.]
        self.outer_cls = [2.]
        self.other_cls = [
            11., 18., 7., 8., 9., 10., 12., 16., 17., 19., 20., 22., 23., 15.,
            14., 13., 0., 6.
        ]

    def _open_file(self, path_prefix, fname):
        return open(os.path.join(path_prefix, fname), 'rb')

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(self._img_path, fname) as f:
            image = Image.open(f)
            if self.downsample_factor != 1:
                width, height = image.size
                width = width // self.downsample_factor
                height = height // self.downsample_factor
                image = image.resize(
                    size=(width, height), resample=Image.LANCZOS)
            image = np.array(image)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image

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
        segm = segm[:, :, np.newaxis].transpose(2, 0, 1)
        return segm.astype(np.float32)

    def __getitem__(self, index):
        image = self._load_raw_image(index)
        pose = self._load_densepose(index)
        segm = self._load_segm(index)

        if self.xflip and random.random() > 0.5:
            assert image.ndim == 3  # CHW
            image = image[:, :, ::-1].copy()
            pose = pose[:, :, ::-1].copy()
            segm = segm[:, :, ::-1].copy()

        image = torch.from_numpy(image)
        segm = torch.from_numpy(segm)

        upper_fused_attr = self.upper_fused_attrs[index]
        lower_fused_attr = self.lower_fused_attrs[index]
        outer_fused_attr = self.outer_fused_attrs[index]

        # mask 0: denotes the common codebook,
        # mask (attr + 1): denotes the texture-specific codebook
        mask = torch.zeros_like(segm)
        if upper_fused_attr != 17:
            for cls in self.upper_cls:
                mask[segm == cls] = upper_fused_attr + 1

        if lower_fused_attr != 17:
            for cls in self.lower_cls:
                mask[segm == cls] = lower_fused_attr + 1

        if outer_fused_attr != 17:
            for cls in self.outer_cls:
                mask[segm == cls] = outer_fused_attr + 1

        pose = pose / 12. - 1
        image = image / 127.5 - 1

        return_dict = {
            'image': image,
            'densepose': pose,
            'segm': segm,
            'texture_mask': mask,
            'img_name': self._image_fnames[index]
        }

        return return_dict

    def __len__(self):
        return len(self._image_fnames)
