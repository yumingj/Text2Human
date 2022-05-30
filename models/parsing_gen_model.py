import logging
import math
from collections import OrderedDict

import mmcv
import numpy as np
import torch
from torchvision.utils import save_image

from models.archs.fcn_arch import FCNHead
from models.archs.shape_attr_embedding_arch import ShapeAttrEmbedding
from models.archs.unet_arch import ShapeUNet
from models.losses.accuracy import accuracy
from models.losses.cross_entropy_loss import CrossEntropyLoss

logger = logging.getLogger('base')


class ParsingGenModel():
    """Paring Generation model.
    """

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda')
        self.is_train = opt['is_train']

        self.attr_embedder = ShapeAttrEmbedding(
            dim=opt['embedder_dim'],
            out_dim=opt['embedder_out_dim'],
            cls_num_list=opt['attr_class_num']).to(self.device)
        self.parsing_encoder = ShapeUNet(
            in_channels=opt['encoder_in_channels']).to(self.device)
        self.parsing_decoder = FCNHead(
            in_channels=opt['fc_in_channels'],
            in_index=opt['fc_in_index'],
            channels=opt['fc_channels'],
            num_convs=opt['fc_num_convs'],
            concat_input=opt['fc_concat_input'],
            dropout_ratio=opt['fc_dropout_ratio'],
            num_classes=opt['fc_num_classes'],
            align_corners=opt['fc_align_corners'],
        ).to(self.device)

        self.init_training_settings()

        self.palette = [[0, 0, 0], [255, 250, 250], [220, 220, 220],
                        [250, 235, 215], [255, 250, 205], [211, 211, 211],
                        [70, 130, 180], [127, 255, 212], [0, 100, 0],
                        [50, 205, 50], [255, 255, 0], [245, 222, 179],
                        [255, 140, 0], [255, 0, 0], [16, 78, 139],
                        [144, 238, 144], [50, 205, 174], [50, 155, 250],
                        [160, 140, 88], [213, 140, 88], [90, 140, 90],
                        [185, 210, 205], [130, 165, 180], [225, 141, 151]]

    def init_training_settings(self):
        optim_params = []
        for v in self.attr_embedder.parameters():
            if v.requires_grad:
                optim_params.append(v)
        for v in self.parsing_encoder.parameters():
            if v.requires_grad:
                optim_params.append(v)
        for v in self.parsing_decoder.parameters():
            if v.requires_grad:
                optim_params.append(v)
        # set up optimizers
        self.optimizer = torch.optim.Adam(
            optim_params,
            self.opt['lr'],
            weight_decay=self.opt['weight_decay'])
        self.log_dict = OrderedDict()
        self.entropy_loss = CrossEntropyLoss().to(self.device)

    def feed_data(self, data):
        self.pose = data['densepose'].to(self.device)
        self.attr = data['attr'].to(self.device)
        self.segm = data['segm'].to(self.device)

    def optimize_parameters(self):
        self.attr_embedder.train()
        self.parsing_encoder.train()
        self.parsing_decoder.train()

        self.attr_embedding = self.attr_embedder(self.attr)
        self.pose_enc = self.parsing_encoder(self.pose, self.attr_embedding)
        self.seg_logits = self.parsing_decoder(self.pose_enc)

        loss = self.entropy_loss(self.seg_logits, self.segm)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_dict['loss_total'] = loss

    def get_vis(self, save_path):
        img_cat = torch.cat([
            self.pose,
            self.segm,
        ], dim=3).detach()
        img_cat = ((img_cat + 1) / 2)

        img_cat = img_cat.clamp_(0, 1)

        save_image(img_cat, save_path, nrow=1, padding=4)

    def inference(self, data_loader, save_dir):
        self.attr_embedder.eval()
        self.parsing_encoder.eval()
        self.parsing_decoder.eval()

        acc = 0
        num = 0

        for _, data in enumerate(data_loader):
            pose = data['densepose'].to(self.device)
            attr = data['attr'].to(self.device)
            segm = data['segm'].to(self.device)
            img_name = data['img_name']

            num += pose.size(0)
            with torch.no_grad():
                attr_embedding = self.attr_embedder(attr)
                pose_enc = self.parsing_encoder(pose, attr_embedding)
                seg_logits = self.parsing_decoder(pose_enc)
            seg_pred = seg_logits.argmax(dim=1)
            acc += accuracy(seg_logits, segm)
            palette_label = self.palette_result(segm.cpu().numpy())
            palette_pred = self.palette_result(seg_pred.cpu().numpy())
            pose_numpy = ((pose[0] + 1) / 2. * 255.).expand(
                3,
                pose[0].size(1),
                pose[0].size(2),
            ).cpu().numpy().clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
            concat_result = np.concatenate(
                (pose_numpy, palette_pred, palette_label), axis=1)
            mmcv.imwrite(concat_result, f'{save_dir}/{img_name[0]}')

        self.attr_embedder.train()
        self.parsing_encoder.train()
        self.parsing_decoder.train()
        return (acc / num).item()

    def get_current_log(self):
        return self.log_dict

    def update_learning_rate(self, epoch):
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int): Warmup iter numbers. -1 for no warmup.
                Default: -1.
        """
        lr = self.optimizer.param_groups[0]['lr']

        if self.opt['lr_decay'] == 'step':
            lr = self.opt['lr'] * (
                self.opt['gamma']**(epoch // self.opt['step']))
        elif self.opt['lr_decay'] == 'cos':
            lr = self.opt['lr'] * (
                1 + math.cos(math.pi * epoch / self.opt['num_epochs'])) / 2
        elif self.opt['lr_decay'] == 'linear':
            lr = self.opt['lr'] * (1 - epoch / self.opt['num_epochs'])
        elif self.opt['lr_decay'] == 'linear2exp':
            if epoch < self.opt['turning_point'] + 1:
                # learning rate decay as 95%
                # at the turning point (1 / 95% = 1.0526)
                lr = self.opt['lr'] * (
                    1 - epoch / int(self.opt['turning_point'] * 1.0526))
            else:
                lr *= self.opt['gamma']
        elif self.opt['lr_decay'] == 'schedule':
            if epoch in self.opt['schedule']:
                lr *= self.opt['gamma']
        else:
            raise ValueError('Unknown lr mode {}'.format(self.opt['lr_decay']))
        # set learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def save_network(self, save_path):
        """Save networks.
        """

        save_dict = {}
        save_dict['embedder'] = self.attr_embedder.state_dict()
        save_dict['encoder'] = self.parsing_encoder.state_dict()
        save_dict['decoder'] = self.parsing_decoder.state_dict()

        torch.save(save_dict, save_path)

    def load_network(self):
        checkpoint = torch.load(self.opt['pretrained_parsing_gen'])

        self.attr_embedder.load_state_dict(checkpoint['embedder'], strict=True)
        self.attr_embedder.eval()

        self.parsing_encoder.load_state_dict(
            checkpoint['encoder'], strict=True)
        self.parsing_encoder.eval()

        self.parsing_decoder.load_state_dict(
            checkpoint['decoder'], strict=True)
        self.parsing_decoder.eval()

    def palette_result(self, result):
        seg = result[0]
        palette = np.array(self.palette)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]
        return color_seg
