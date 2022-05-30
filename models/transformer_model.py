import logging
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.distributions as dists
import torch.nn.functional as F
from torchvision.utils import save_image

from models.archs.transformer_arch import TransformerMultiHead
from models.archs.vqgan_arch import (Decoder, Encoder, VectorQuantizer,
                                     VectorQuantizerTexture)

logger = logging.getLogger('base')


class TransformerTextureAwareModel():
    """Texture-Aware Diffusion based Transformer model.
    """

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda')
        self.is_train = opt['is_train']

        # VQVAE for image
        self.img_encoder = Encoder(
            ch=opt['img_ch'],
            num_res_blocks=opt['img_num_res_blocks'],
            attn_resolutions=opt['img_attn_resolutions'],
            ch_mult=opt['img_ch_mult'],
            in_channels=opt['img_in_channels'],
            resolution=opt['img_resolution'],
            z_channels=opt['img_z_channels'],
            double_z=opt['img_double_z'],
            dropout=opt['img_dropout']).to(self.device)
        self.img_decoder = Decoder(
            in_channels=opt['img_in_channels'],
            resolution=opt['img_resolution'],
            z_channels=opt['img_z_channels'],
            ch=opt['img_ch'],
            out_ch=opt['img_out_ch'],
            num_res_blocks=opt['img_num_res_blocks'],
            attn_resolutions=opt['img_attn_resolutions'],
            ch_mult=opt['img_ch_mult'],
            dropout=opt['img_dropout'],
            resamp_with_conv=True,
            give_pre_end=False).to(self.device)
        self.img_quantizer = VectorQuantizerTexture(
            opt['img_n_embed'], opt['img_embed_dim'],
            beta=0.25).to(self.device)
        self.img_quant_conv = torch.nn.Conv2d(opt["img_z_channels"],
                                              opt['img_embed_dim'],
                                              1).to(self.device)
        self.img_post_quant_conv = torch.nn.Conv2d(opt['img_embed_dim'],
                                                   opt["img_z_channels"],
                                                   1).to(self.device)
        self.load_pretrained_image_vae()

        # VAE for segmentation mask
        self.segm_encoder = Encoder(
            ch=opt['segm_ch'],
            num_res_blocks=opt['segm_num_res_blocks'],
            attn_resolutions=opt['segm_attn_resolutions'],
            ch_mult=opt['segm_ch_mult'],
            in_channels=opt['segm_in_channels'],
            resolution=opt['segm_resolution'],
            z_channels=opt['segm_z_channels'],
            double_z=opt['segm_double_z'],
            dropout=opt['segm_dropout']).to(self.device)
        self.segm_quantizer = VectorQuantizer(
            opt['segm_n_embed'],
            opt['segm_embed_dim'],
            beta=0.25,
            sane_index_shape=True).to(self.device)
        self.segm_quant_conv = torch.nn.Conv2d(opt["segm_z_channels"],
                                               opt['segm_embed_dim'],
                                               1).to(self.device)
        self.load_pretrained_segm_vae()

        # define sampler
        self._denoise_fn = TransformerMultiHead(
            codebook_size=opt['codebook_size'],
            segm_codebook_size=opt['segm_codebook_size'],
            texture_codebook_size=opt['texture_codebook_size'],
            bert_n_emb=opt['bert_n_emb'],
            bert_n_layers=opt['bert_n_layers'],
            bert_n_head=opt['bert_n_head'],
            block_size=opt['block_size'],
            latent_shape=opt['latent_shape'],
            embd_pdrop=opt['embd_pdrop'],
            resid_pdrop=opt['resid_pdrop'],
            attn_pdrop=opt['attn_pdrop'],
            num_head=opt['num_head']).to(self.device)

        self.num_classes = opt['codebook_size']
        self.shape = tuple(opt['latent_shape'])
        self.num_timesteps = 1000

        self.mask_id = opt['codebook_size']
        self.loss_type = opt['loss_type']
        self.mask_schedule = opt['mask_schedule']

        self.sample_steps = opt['sample_steps']

        self.init_training_settings()

    def load_pretrained_image_vae(self):
        # load pretrained vqgan for segmentation mask
        img_ae_checkpoint = torch.load(self.opt['img_ae_path'])
        self.img_encoder.load_state_dict(
            img_ae_checkpoint['encoder'], strict=True)
        self.img_decoder.load_state_dict(
            img_ae_checkpoint['decoder'], strict=True)
        self.img_quantizer.load_state_dict(
            img_ae_checkpoint['quantize'], strict=True)
        self.img_quant_conv.load_state_dict(
            img_ae_checkpoint['quant_conv'], strict=True)
        self.img_post_quant_conv.load_state_dict(
            img_ae_checkpoint['post_quant_conv'], strict=True)
        self.img_encoder.eval()
        self.img_decoder.eval()
        self.img_quantizer.eval()
        self.img_quant_conv.eval()
        self.img_post_quant_conv.eval()

    def load_pretrained_segm_vae(self):
        # load pretrained vqgan for segmentation mask
        segm_ae_checkpoint = torch.load(self.opt['segm_ae_path'])
        self.segm_encoder.load_state_dict(
            segm_ae_checkpoint['encoder'], strict=True)
        self.segm_quantizer.load_state_dict(
            segm_ae_checkpoint['quantize'], strict=True)
        self.segm_quant_conv.load_state_dict(
            segm_ae_checkpoint['quant_conv'], strict=True)
        self.segm_encoder.eval()
        self.segm_quantizer.eval()
        self.segm_quant_conv.eval()

    def init_training_settings(self):
        optim_params = []
        for v in self._denoise_fn.parameters():
            if v.requires_grad:
                optim_params.append(v)
        # set up optimizer
        self.optimizer = torch.optim.Adam(
            optim_params,
            self.opt['lr'],
            weight_decay=self.opt['weight_decay'])
        self.log_dict = OrderedDict()

    @torch.no_grad()
    def get_quantized_img(self, image, texture_mask):
        encoded_img = self.img_encoder(image)
        encoded_img = self.img_quant_conv(encoded_img)

        # img_tokens_input is the continual index for the input of transformer
        # img_tokens_gt_list is the index for 18 texture-aware codebooks respectively
        _, _, [_, img_tokens_input, img_tokens_gt_list
               ] = self.img_quantizer(encoded_img, texture_mask)

        # reshape the tokens
        b = image.size(0)
        img_tokens_input = img_tokens_input.view(b, -1)
        img_tokens_gt_return_list = [
            img_tokens_gt.view(b, -1) for img_tokens_gt in img_tokens_gt_list
        ]

        return img_tokens_input, img_tokens_gt_return_list

    @torch.no_grad()
    def decode(self, quant):
        quant = self.img_post_quant_conv(quant)
        dec = self.img_decoder(quant)
        return dec

    @torch.no_grad()
    def decode_image_indices(self, indices_list, texture_mask):
        quant = self.img_quantizer.get_codebook_entry(
            indices_list, texture_mask,
            (indices_list[0].size(0), self.shape[0], self.shape[1],
             self.opt["img_z_channels"]))
        dec = self.decode(quant)

        return dec

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(
                1, self.num_timesteps + 1, (b, ), device=device).long()
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt

        else:
            raise ValueError

    def q_sample(self, x_0, x_0_gt_list, t):
        # samples q(x_t | x_0)
        # randomly set token to mask with probability t/T
        # x_t, x_0_ignore = x_0.clone(), x_0.clone()
        x_t = x_0.clone()

        mask = torch.rand_like(x_t.float()) < (
            t.float().unsqueeze(-1) / self.num_timesteps)
        x_t[mask] = self.mask_id
        # x_0_ignore[torch.bitwise_not(mask)] = -1

        # for every gt token list, we also need to do the mask
        x_0_gt_ignore_list = []
        for x_0_gt in x_0_gt_list:
            x_0_gt_ignore = x_0_gt.clone()
            x_0_gt_ignore[torch.bitwise_not(mask)] = -1
            x_0_gt_ignore_list.append(x_0_gt_ignore)

        return x_t, x_0_gt_ignore_list, mask

    def _train_loss(self, x_0, x_0_gt_list):
        b, device = x_0.size(0), x_0.device

        # choose what time steps to compute loss at
        t, pt = self.sample_time(b, device, 'uniform')

        # make x noisy and denoise
        if self.mask_schedule == 'random':
            x_t, x_0_gt_ignore_list, mask = self.q_sample(
                x_0=x_0, x_0_gt_list=x_0_gt_list, t=t)
        else:
            raise NotImplementedError

        # sample p(x_0 | x_t)
        x_0_hat_logits_list = self._denoise_fn(
            x_t, self.segm_tokens, self.texture_tokens, t=t)

        # Always compute ELBO for comparison purposes
        cross_entropy_loss = 0
        for x_0_hat_logits, x_0_gt_ignore in zip(x_0_hat_logits_list,
                                                 x_0_gt_ignore_list):
            cross_entropy_loss += F.cross_entropy(
                x_0_hat_logits.permute(0, 2, 1),
                x_0_gt_ignore,
                ignore_index=-1,
                reduction='none').sum(1)
        vb_loss = cross_entropy_loss / t
        vb_loss = vb_loss / pt
        vb_loss = vb_loss / (math.log(2) * x_0.shape[1:].numel())
        if self.loss_type == 'elbo':
            loss = vb_loss
        elif self.loss_type == 'mlm':
            denom = mask.float().sum(1)
            denom[denom == 0] = 1  # prevent divide by 0 errors.
            loss = cross_entropy_loss / denom
        elif self.loss_type == 'reweighted_elbo':
            weight = (1 - (t / self.num_timesteps))
            loss = weight * cross_entropy_loss
            loss = loss / (math.log(2) * x_0.shape[1:].numel())
        else:
            raise ValueError

        return loss.mean(), vb_loss.mean()

    def feed_data(self, data):
        self.image = data['image'].to(self.device)
        self.segm = data['segm'].to(self.device)
        self.texture_mask = data['texture_mask'].to(self.device)
        self.input_indices, self.gt_indices_list = self.get_quantized_img(
            self.image, self.texture_mask)

        self.texture_tokens = F.interpolate(
            self.texture_mask, size=self.shape,
            mode='nearest').view(self.image.size(0), -1).long()

        self.segm_tokens = self.get_quantized_segm(self.segm)
        self.segm_tokens = self.segm_tokens.view(self.image.size(0), -1)

    def optimize_parameters(self):
        self._denoise_fn.train()

        loss, vb_loss = self._train_loss(self.input_indices,
                                         self.gt_indices_list)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_dict['loss'] = loss
        self.log_dict['vb_loss'] = vb_loss

        self._denoise_fn.eval()

    @torch.no_grad()
    def get_quantized_segm(self, segm):
        segm_one_hot = F.one_hot(
            segm.squeeze(1).long(),
            num_classes=self.opt['segm_num_segm_classes']).permute(
                0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        encoded_segm_mask = self.segm_encoder(segm_one_hot)
        encoded_segm_mask = self.segm_quant_conv(encoded_segm_mask)
        _, _, [_, _, segm_tokens] = self.segm_quantizer(encoded_segm_mask)

        return segm_tokens

    def sample_fn(self, temp=1.0, sample_steps=None):
        self._denoise_fn.eval()

        b, device = self.image.size(0), 'cuda'
        x_t = torch.ones(
            (b, np.prod(self.shape)), device=device).long() * self.mask_id
        unmasked = torch.zeros_like(x_t, device=device).bool()
        sample_steps = list(range(1, sample_steps + 1))

        texture_mask_flatten = self.texture_tokens.view(-1)

        # min_encodings_indices_list would be used to visualize the image
        min_encodings_indices_list = [
            torch.full(
                texture_mask_flatten.size(),
                fill_value=-1,
                dtype=torch.long,
                device=texture_mask_flatten.device) for _ in range(18)
        ]

        for t in reversed(sample_steps):
            print(f'Sample timestep {t:4d}', end='\r')
            t = torch.full((b, ), t, device=device, dtype=torch.long)

            # where to unmask
            changes = torch.rand(
                x_t.shape, device=device) < 1 / t.float().unsqueeze(-1)
            # don't unmask somewhere already unmasked
            changes = torch.bitwise_xor(changes,
                                        torch.bitwise_and(changes, unmasked))
            # update mask with changes
            unmasked = torch.bitwise_or(unmasked, changes)

            x_0_logits_list = self._denoise_fn(
                x_t, self.segm_tokens, self.texture_tokens, t=t)

            changes_flatten = changes.view(-1)
            ori_shape = x_t.shape  # [b, h*w]
            x_t = x_t.view(-1)  # [b*h*w]
            for codebook_idx, x_0_logits in enumerate(x_0_logits_list):
                if torch.sum(texture_mask_flatten[changes_flatten] ==
                             codebook_idx) > 0:
                    # scale by temperature
                    x_0_logits = x_0_logits / temp
                    x_0_dist = dists.Categorical(logits=x_0_logits)
                    x_0_hat = x_0_dist.sample().long()
                    x_0_hat = x_0_hat.view(-1)

                    # only replace the changed indices with corresponding codebook_idx
                    changes_segm = torch.bitwise_and(
                        changes_flatten, texture_mask_flatten == codebook_idx)

                    # x_t would be the input to the transformer, so the index range should be continual one
                    x_t[changes_segm] = x_0_hat[
                        changes_segm] + 1024 * codebook_idx
                    min_encodings_indices_list[codebook_idx][
                        changes_segm] = x_0_hat[changes_segm]

            x_t = x_t.view(ori_shape)  # [b, h*w]

        min_encodings_indices_return_list = [
            min_encodings_indices.view(ori_shape)
            for min_encodings_indices in min_encodings_indices_list
        ]

        self._denoise_fn.train()

        return min_encodings_indices_return_list

    def get_vis(self, image, gt_indices, predicted_indices, texture_mask,
                save_path):
        # original image
        ori_img = self.decode_image_indices(gt_indices, texture_mask)
        # pred image
        pred_img = self.decode_image_indices(predicted_indices, texture_mask)
        img_cat = torch.cat([
            image,
            ori_img,
            pred_img,
        ], dim=3).detach()
        img_cat = ((img_cat + 1) / 2)
        img_cat = img_cat.clamp_(0, 1)
        save_image(img_cat, save_path, nrow=1, padding=4)

    def inference(self, data_loader, save_dir):
        self._denoise_fn.eval()

        for _, data in enumerate(data_loader):
            img_name = data['img_name']
            self.feed_data(data)
            b = self.image.size(0)
            with torch.no_grad():
                sampled_indices_list = self.sample_fn(
                    temp=1, sample_steps=self.sample_steps)
            for idx in range(b):
                self.get_vis(self.image[idx:idx + 1], [
                    gt_indices[idx:idx + 1]
                    for gt_indices in self.gt_indices_list
                ], [
                    sampled_indices[idx:idx + 1]
                    for sampled_indices in sampled_indices_list
                ], self.texture_mask[idx:idx + 1],
                             f'{save_dir}/{img_name[idx]}')

        self._denoise_fn.train()

    def get_current_log(self):
        return self.log_dict

    def update_learning_rate(self, epoch, iters=None):
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
        elif self.opt['lr_decay'] == 'warm_up':
            if iters <= self.opt['warmup_iters']:
                lr = self.opt['lr'] * float(iters) / self.opt['warmup_iters']
            else:
                lr = self.opt['lr']
        else:
            raise ValueError('Unknown lr mode {}'.format(self.opt['lr_decay']))
        # set learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def save_network(self, net, save_path):
        """Save networks.

        Args:
            net (nn.Module): Network to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
        """
        state_dict = net.state_dict()
        torch.save(state_dict, save_path)

    def load_network(self):
        checkpoint = torch.load(self.opt['pretrained_sampler'])
        self._denoise_fn.load_state_dict(checkpoint, strict=True)
        self._denoise_fn.eval()
