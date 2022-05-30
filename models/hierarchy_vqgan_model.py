import math
import sys
from collections import OrderedDict

sys.path.append('..')
import lpips
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from models.archs.vqgan_arch import (Decoder, DecoderRes, Discriminator,
                                     Encoder,
                                     VectorQuantizerSpatialTextureAware,
                                     VectorQuantizerTexture)
from models.losses.vqgan_loss import (DiffAugment, adopt_weight,
                                      calculate_adaptive_weight, hinge_d_loss)


class HierarchyVQSpatialTextureAwareModel():

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda')
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

        self.disc = Discriminator(
            opt['n_channels'], opt['ndf'],
            n_layers=opt['disc_layers']).to(self.device)
        self.perceptual = lpips.LPIPS(net="vgg").to(self.device)
        self.perceptual_weight = opt['perceptual_weight']
        self.disc_start_step = opt['disc_start_step']
        self.disc_weight_max = opt['disc_weight_max']
        self.diff_aug = opt['diff_aug']
        self.policy = "color,translation"

        self.load_discriminator_models()

        self.disc.train()

        self.fix_decoder = opt['fix_decoder']

        self.init_training_settings()

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

    def init_training_settings(self):
        self.log_dict = OrderedDict()
        self.configure_optimizers()

    def configure_optimizers(self):
        optim_params = []
        for v in self.bot_encoder.parameters():
            if v.requires_grad:
                optim_params.append(v)
        for v in self.bot_decoder_res.parameters():
            if v.requires_grad:
                optim_params.append(v)
        for v in self.bot_quantize.parameters():
            if v.requires_grad:
                optim_params.append(v)
        for v in self.bot_quant_conv.parameters():
            if v.requires_grad:
                optim_params.append(v)
        for v in self.bot_post_quant_conv.parameters():
            if v.requires_grad:
                optim_params.append(v)
        if not self.fix_decoder:
            for name, v in self.decoder.named_parameters():
                if v.requires_grad:
                    if 'up.0' in name:
                        optim_params.append(v)
                    if 'up.1' in name:
                        optim_params.append(v)
                    if 'up.2' in name:
                        optim_params.append(v)
                    if 'up.3' in name:
                        optim_params.append(v)

        self.optimizer = torch.optim.Adam(optim_params, lr=self.opt['lr'])

        self.disc_optimizer = torch.optim.Adam(
            self.disc.parameters(), lr=self.opt['lr'])

    def load_discriminator_models(self):
        # load pretrained vqgan for segmentation mask
        top_vae_checkpoint = torch.load(self.opt['top_vae_path'])
        self.disc.load_state_dict(
            top_vae_checkpoint['discriminator'], strict=True)

    def save_network(self, save_path):
        """Save networks.
        """

        save_dict = {}
        save_dict['bot_encoder'] = self.bot_encoder.state_dict()
        save_dict['bot_decoder_res'] = self.bot_decoder_res.state_dict()
        save_dict['decoder'] = self.decoder.state_dict()
        save_dict['bot_quantize'] = self.bot_quantize.state_dict()
        save_dict['bot_quant_conv'] = self.bot_quant_conv.state_dict()
        save_dict['bot_post_quant_conv'] = self.bot_post_quant_conv.state_dict(
        )
        save_dict['discriminator'] = self.disc.state_dict()
        torch.save(save_dict, save_path)

    def load_network(self):
        checkpoint = torch.load(self.opt['pretrained_models'])
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

    def optimize_parameters(self, data, step):
        self.bot_encoder.train()
        self.bot_decoder_res.train()
        if not self.fix_decoder:
            self.decoder.train()
        self.bot_quantize.train()
        self.bot_quant_conv.train()
        self.bot_post_quant_conv.train()

        loss, d_loss = self.training_step(data, step)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if step > self.disc_start_step:
            self.disc_optimizer.zero_grad()
            d_loss.backward()
            self.disc_optimizer.step()

    def top_encode(self, x, mask):
        h = self.top_encoder(x)
        h = self.top_quant_conv(h)
        quant, _, _ = self.top_quantize(h, mask)
        quant = self.top_post_quant_conv(quant)
        return quant

    def bot_encode(self, x, mask):
        h = self.bot_encoder(x)
        h = self.bot_quant_conv(h)
        quant, emb_loss, info = self.bot_quantize(h, mask)
        quant = self.bot_post_quant_conv(quant)
        bot_dec_res = self.bot_decoder_res(quant)
        return bot_dec_res, emb_loss, info

    def decode(self, quant_top, bot_dec_res):
        dec = self.decoder(quant_top, bot_h=bot_dec_res)
        return dec

    def forward_step(self, input, mask):
        with torch.no_grad():
            quant_top = self.top_encode(input, mask)
        bot_dec_res, diff, _ = self.bot_encode(input, mask)
        dec = self.decode(quant_top, bot_dec_res)
        return dec, diff

    def feed_data(self, data):
        x = data['image'].float().to(self.device)
        mask = data['texture_mask'].float().to(self.device)

        return x, mask

    def training_step(self, data, step):
        x, mask = self.feed_data(data)
        xrec, codebook_loss = self.forward_step(x, mask)

        # get recon/perceptual loss
        recon_loss = torch.abs(x.contiguous() - xrec.contiguous())
        p_loss = self.perceptual(x.contiguous(), xrec.contiguous())
        nll_loss = recon_loss + self.perceptual_weight * p_loss
        nll_loss = torch.mean(nll_loss)

        # augment for input to discriminator
        if self.diff_aug:
            xrec = DiffAugment(xrec, policy=self.policy)

        # update generator
        logits_fake = self.disc(xrec)
        g_loss = -torch.mean(logits_fake)
        last_layer = self.decoder.conv_out.weight
        d_weight = calculate_adaptive_weight(nll_loss, g_loss, last_layer,
                                             self.disc_weight_max)
        d_weight *= adopt_weight(1, step, self.disc_start_step)
        loss = nll_loss + d_weight * g_loss + codebook_loss

        self.log_dict["loss"] = loss
        self.log_dict["l1"] = recon_loss.mean().item()
        self.log_dict["perceptual"] = p_loss.mean().item()
        self.log_dict["nll_loss"] = nll_loss.item()
        self.log_dict["g_loss"] = g_loss.item()
        self.log_dict["d_weight"] = d_weight
        self.log_dict["codebook_loss"] = codebook_loss.item()

        if step > self.disc_start_step:
            if self.diff_aug:
                logits_real = self.disc(
                    DiffAugment(x.contiguous().detach(), policy=self.policy))
            else:
                logits_real = self.disc(x.contiguous().detach())
            logits_fake = self.disc(xrec.contiguous().detach(
            ))  # detach so that generator isn"t also updated
            d_loss = hinge_d_loss(logits_real, logits_fake)
            self.log_dict["d_loss"] = d_loss
        else:
            d_loss = None

        return loss, d_loss

    @torch.no_grad()
    def inference(self, data_loader, save_dir):
        self.bot_encoder.eval()
        self.bot_decoder_res.eval()
        self.decoder.eval()
        self.bot_quantize.eval()
        self.bot_quant_conv.eval()
        self.bot_post_quant_conv.eval()

        loss_total = 0
        num = 0

        for _, data in enumerate(data_loader):
            img_name = data['img_name'][0]
            x, mask = self.feed_data(data)
            xrec, _ = self.forward_step(x, mask)

            recon_loss = torch.abs(x.contiguous() - xrec.contiguous())
            p_loss = self.perceptual(x.contiguous(), xrec.contiguous())
            nll_loss = recon_loss + self.perceptual_weight * p_loss
            nll_loss = torch.mean(nll_loss)
            loss_total += nll_loss

            num += x.size(0)

            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                # convert logits to indices
                xrec = torch.argmax(xrec, dim=1, keepdim=True)
                xrec = F.one_hot(xrec, num_classes=x.shape[1])
                xrec = xrec.squeeze(1).permute(0, 3, 1, 2).float()
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)

            img_cat = torch.cat([x, xrec], dim=3).detach()
            img_cat = ((img_cat + 1) / 2)
            img_cat = img_cat.clamp_(0, 1)
            save_image(
                img_cat, f'{save_dir}/{img_name}.png', nrow=1, padding=4)

        return (loss_total / num).item()

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
