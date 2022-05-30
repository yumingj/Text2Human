import logging
import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from models.archs.fcn_arch import MultiHeadFCNHead
from models.archs.unet_arch import UNet
from models.archs.vqgan_arch import (Decoder, DecoderRes, Encoder,
                                     VectorQuantizerSpatialTextureAware,
                                     VectorQuantizerTexture)
from models.losses.accuracy import accuracy
from models.losses.cross_entropy_loss import CrossEntropyLoss

logger = logging.getLogger('base')


class VQGANTextureAwareSpatialHierarchyInferenceModel():

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda')
        self.is_train = opt['is_train']

        self.top_encoder = Encoder(
            ch=opt['top_ch'],
            num_res_blocks=opt['top_num_res_blocks'],
            attn_resolutions=opt['top_attn_resolutions'],
            ch_mult=opt['top_ch_mult'],
            in_channels=opt['top_in_channels'],
            resolution=opt['top_resolution'],
            z_channels=opt['top_z_channels'],
            double_z=opt['top_double_z'],
            dropout=opt['top_dropout']).to(self.device)
        self.decoder = Decoder(
            in_channels=opt['top_in_channels'],
            resolution=opt['top_resolution'],
            z_channels=opt['top_z_channels'],
            ch=opt['top_ch'],
            out_ch=opt['top_out_ch'],
            num_res_blocks=opt['top_num_res_blocks'],
            attn_resolutions=opt['top_attn_resolutions'],
            ch_mult=opt['top_ch_mult'],
            dropout=opt['top_dropout'],
            resamp_with_conv=True,
            give_pre_end=False).to(self.device)
        self.top_quantize = VectorQuantizerTexture(
            1024, opt['embed_dim'], beta=0.25).to(self.device)
        self.top_quant_conv = torch.nn.Conv2d(opt["top_z_channels"],
                                              opt['embed_dim'],
                                              1).to(self.device)
        self.top_post_quant_conv = torch.nn.Conv2d(opt['embed_dim'],
                                                   opt["top_z_channels"],
                                                   1).to(self.device)
        self.load_top_pretrain_models()

        self.bot_encoder = Encoder(
            ch=opt['bot_ch'],
            num_res_blocks=opt['bot_num_res_blocks'],
            attn_resolutions=opt['bot_attn_resolutions'],
            ch_mult=opt['bot_ch_mult'],
            in_channels=opt['bot_in_channels'],
            resolution=opt['bot_resolution'],
            z_channels=opt['bot_z_channels'],
            double_z=opt['bot_double_z'],
            dropout=opt['bot_dropout']).to(self.device)
        self.bot_decoder_res = DecoderRes(
            in_channels=opt['bot_in_channels'],
            resolution=opt['bot_resolution'],
            z_channels=opt['bot_z_channels'],
            ch=opt['bot_ch'],
            num_res_blocks=opt['bot_num_res_blocks'],
            ch_mult=opt['bot_ch_mult'],
            dropout=opt['bot_dropout'],
            give_pre_end=False).to(self.device)
        self.bot_quantize = VectorQuantizerSpatialTextureAware(
            opt['bot_n_embed'],
            opt['embed_dim'],
            beta=0.25,
            spatial_size=opt['codebook_spatial_size']).to(self.device)
        self.bot_quant_conv = torch.nn.Conv2d(opt["bot_z_channels"],
                                              opt['embed_dim'],
                                              1).to(self.device)
        self.bot_post_quant_conv = torch.nn.Conv2d(opt['embed_dim'],
                                                   opt["bot_z_channels"],
                                                   1).to(self.device)

        self.load_bot_pretrain_network()

        self.guidance_encoder = UNet(
            in_channels=opt['encoder_in_channels']).to(self.device)
        self.index_decoder = MultiHeadFCNHead(
            in_channels=opt['fc_in_channels'],
            in_index=opt['fc_in_index'],
            channels=opt['fc_channels'],
            num_convs=opt['fc_num_convs'],
            concat_input=opt['fc_concat_input'],
            dropout_ratio=opt['fc_dropout_ratio'],
            num_classes=opt['fc_num_classes'],
            align_corners=opt['fc_align_corners'],
            num_head=18).to(self.device)

        self.init_training_settings()

    def init_training_settings(self):
        optim_params = []
        for v in self.guidance_encoder.parameters():
            if v.requires_grad:
                optim_params.append(v)
        for v in self.index_decoder.parameters():
            if v.requires_grad:
                optim_params.append(v)
        # set up optimizers
        if self.opt['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(
                optim_params,
                self.opt['lr'],
                weight_decay=self.opt['weight_decay'])
        elif self.opt['optimizer'] == 'SGD':
            self.optimizer = torch.optim.SGD(
                optim_params,
                self.opt['lr'],
                momentum=self.opt['momentum'],
                weight_decay=self.opt['weight_decay'])
        self.log_dict = OrderedDict()
        if self.opt['loss_function'] == 'cross_entropy':
            self.loss_func = CrossEntropyLoss().to(self.device)

    def load_top_pretrain_models(self):
        # load pretrained vqgan for segmentation mask
        top_vae_checkpoint = torch.load(self.opt['top_vae_path'])
        self.top_encoder.load_state_dict(
            top_vae_checkpoint['encoder'], strict=True)
        self.decoder.load_state_dict(
            top_vae_checkpoint['decoder'], strict=True)
        self.top_quantize.load_state_dict(
            top_vae_checkpoint['quantize'], strict=True)
        self.top_quant_conv.load_state_dict(
            top_vae_checkpoint['quant_conv'], strict=True)
        self.top_post_quant_conv.load_state_dict(
            top_vae_checkpoint['post_quant_conv'], strict=True)
        self.top_encoder.eval()
        self.top_quantize.eval()
        self.top_quant_conv.eval()
        self.top_post_quant_conv.eval()

    def load_bot_pretrain_network(self):
        checkpoint = torch.load(self.opt['bot_vae_path'])
        self.bot_encoder.load_state_dict(
            checkpoint['bot_encoder'], strict=True)
        self.bot_decoder_res.load_state_dict(
            checkpoint['bot_decoder_res'], strict=True)
        self.decoder.load_state_dict(checkpoint['decoder'], strict=True)
        self.bot_quantize.load_state_dict(
            checkpoint['bot_quantize'], strict=True)
        self.bot_quant_conv.load_state_dict(
            checkpoint['bot_quant_conv'], strict=True)
        self.bot_post_quant_conv.load_state_dict(
            checkpoint['bot_post_quant_conv'], strict=True)

        self.bot_encoder.eval()
        self.bot_decoder_res.eval()
        self.decoder.eval()
        self.bot_quantize.eval()
        self.bot_quant_conv.eval()
        self.bot_post_quant_conv.eval()

    def top_encode(self, x, mask):
        h = self.top_encoder(x)
        h = self.top_quant_conv(h)
        quant, _, _ = self.top_quantize(h, mask)
        quant = self.top_post_quant_conv(quant)

        return quant, quant

    def feed_data(self, data):
        self.image = data['image'].to(self.device)
        self.texture_mask = data['texture_mask'].float().to(self.device)
        self.get_gt_indices()

        self.texture_tokens = F.interpolate(
            self.texture_mask, size=(32, 16),
            mode='nearest').view(self.image.size(0), -1).long()

    def bot_encode(self, x, mask):
        h = self.bot_encoder(x)
        h = self.bot_quant_conv(h)
        _, _, (_, _, indices_list) = self.bot_quantize(h, mask)

        return indices_list

    def get_gt_indices(self):
        self.quant_t, self.feature_t = self.top_encode(self.image,
                                                       self.texture_mask)
        self.gt_indices_list = self.bot_encode(self.image, self.texture_mask)

    def index_to_image(self, index_bottom_list, texture_mask):
        quant_b = self.bot_quantize.get_codebook_entry(
            index_bottom_list, texture_mask,
            (index_bottom_list[0].size(0), index_bottom_list[0].size(1),
             index_bottom_list[0].size(2),
             self.opt["bot_z_channels"]))  #.permute(0, 3, 1, 2)
        quant_b = self.bot_post_quant_conv(quant_b)
        bot_dec_res = self.bot_decoder_res(quant_b)

        dec = self.decoder(self.quant_t, bot_h=bot_dec_res)

        return dec

    def get_vis(self, pred_img_index, rec_img_index, texture_mask, save_path):
        rec_img = self.index_to_image(rec_img_index, texture_mask)
        pred_img = self.index_to_image(pred_img_index, texture_mask)

        base_img = self.decoder(self.quant_t)
        img_cat = torch.cat([
            self.image,
            rec_img,
            base_img,
            pred_img,
        ], dim=3).detach()
        img_cat = ((img_cat + 1) / 2)
        img_cat = img_cat.clamp_(0, 1)
        save_image(img_cat, save_path, nrow=1, padding=4)

    def optimize_parameters(self):
        self.guidance_encoder.train()
        self.index_decoder.train()

        self.feature_enc = self.guidance_encoder(self.feature_t)
        self.memory_logits_list = self.index_decoder(self.feature_enc)

        loss = 0
        for i in range(18):
            loss += self.loss_func(
                self.memory_logits_list[i],
                self.gt_indices_list[i],
                ignore_index=-1)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_dict['loss_total'] = loss

    def inference(self, data_loader, save_dir):
        self.guidance_encoder.eval()
        self.index_decoder.eval()

        acc = 0
        num = 0

        for _, data in enumerate(data_loader):
            self.feed_data(data)
            img_name = data['img_name']

            num += self.image.size(0)

            texture_mask_flatten = self.texture_tokens.view(-1)
            min_encodings_indices_list = [
                torch.full(
                    texture_mask_flatten.size(),
                    fill_value=-1,
                    dtype=torch.long,
                    device=texture_mask_flatten.device) for _ in range(18)
            ]
            with torch.no_grad():
                self.feature_enc = self.guidance_encoder(self.feature_t)
                memory_logits_list = self.index_decoder(self.feature_enc)
            # memory_indices_pred = memory_logits.argmax(dim=1)
            batch_acc = 0
            for codebook_idx, memory_logits in enumerate(memory_logits_list):
                region_of_interest = texture_mask_flatten == codebook_idx
                if torch.sum(region_of_interest) > 0:
                    memory_indices_pred = memory_logits.argmax(dim=1).view(-1)
                    batch_acc += torch.sum(
                        memory_indices_pred[region_of_interest] ==
                        self.gt_indices_list[codebook_idx].view(
                            -1)[region_of_interest])
                    memory_indices_pred = memory_indices_pred
                    min_encodings_indices_list[codebook_idx][
                        region_of_interest] = memory_indices_pred[
                            region_of_interest]
            min_encodings_indices_return_list = [
                min_encodings_indices.view(self.gt_indices_list[0].size())
                for min_encodings_indices in min_encodings_indices_list
            ]
            batch_acc = batch_acc / self.gt_indices_list[codebook_idx].numel(
            ) * self.image.size(0)
            acc += batch_acc
            self.get_vis(min_encodings_indices_return_list,
                         self.gt_indices_list, self.texture_mask,
                         f'{save_dir}/{img_name[0]}')

        self.guidance_encoder.train()
        self.index_decoder.train()
        return (acc / num).item()

    def load_network(self):
        checkpoint = torch.load(self.opt['pretrained_models'])
        self.guidance_encoder.load_state_dict(
            checkpoint['guidance_encoder'], strict=True)
        self.guidance_encoder.eval()

        self.index_decoder.load_state_dict(
            checkpoint['index_decoder'], strict=True)
        self.index_decoder.eval()

    def save_network(self, save_path):
        """Save networks.

        Args:
            net (nn.Module): Network to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
        """

        save_dict = {}
        save_dict['guidance_encoder'] = self.guidance_encoder.state_dict()
        save_dict['index_decoder'] = self.index_decoder.state_dict()

        torch.save(save_dict, save_path)

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

    def get_current_log(self):
        return self.log_dict
