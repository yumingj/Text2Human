import os
import os.path

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


class ParsingGenerationDeepFashionAttrSegmDataset(data.Dataset):

    def __init__(self, segm_dir, pose_dir, ann_file, downsample_factor=2):
        self._densepose_path = pose_dir
        self._segm_path = segm_dir
        self._image_fnames = []
        self.attrs = []

        self.downsample_factor = downsample_factor

        # training, ground-truth available
        assert os.path.exists(ann_file)
        for row in open(os.path.join(ann_file), 'r'):
            annotations = row.split()
            self._image_fnames.append(annotations[0])
            self.attrs.append([int(i) for i in annotations[1:]])

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
        return segm.astype(np.float32)

    def __getitem__(self, index):
        pose = self._load_densepose(index)
        segm = self._load_segm(index)
        attr = self.attrs[index]

        pose = torch.from_numpy(pose)
        segm = torch.LongTensor(segm)
        attr = torch.LongTensor(attr)

        pose = pose / 12. - 1

        return_dict = {
            'densepose': pose,
            'segm': segm,
            'attr': attr,
            'img_name': self._image_fnames[index]
        }

        return return_dict

    def __len__(self):
        return len(self._image_fnames)
