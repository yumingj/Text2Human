import math
import sys
from collections import OrderedDict

sys.path.append('..')
import lpips
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from models.archs.vqgan_arch import (Decoder, Discriminator, Encoder,
                                     VectorQuantizer, VectorQuantizerTexture)
from models.losses.segmentation_loss import BCELossWithQuant
from models.losses.vqgan_loss import (DiffAugment, adopt_weight,
                                      calculate_adaptive_weight, hinge_d_loss)


class VQModel():

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.device = torch.device('cuda')
        self.encoder = Encoder(
            ch=opt['ch'],
            num_res_blocks=opt['num_res_blocks'],
            attn_resolutions=opt['attn_resolutions'],
            ch_mult=opt['ch_mult'],
            in_channels=opt['in_channels'],
            resolution=opt['resolution'],
            z_channels=opt['z_channels'],
            double_z=opt['double_z'],
            dropout=opt['dropout']).to(self.device)
        self.decoder = Decoder(
            in_channels=opt['in_channels'],
            resolution=opt['resolution'],
            z_channels=opt['z_channels'],
            ch=opt['ch'],
            out_ch=opt['out_ch'],
            num_res_blocks=opt['num_res_blocks'],
            attn_resolutions=opt['attn_resolutions'],
            ch_mult=opt['ch_mult'],
            dropout=opt['dropout'],
            resamp_with_conv=True,
            give_pre_end=False).to(self.device)
        self.quantize = VectorQuantizer(
            opt['n_embed'], opt['embed_dim'], beta=0.25).to(self.device)
        self.quant_conv = torch.nn.Conv2d(opt["z_channels"], opt['embed_dim'],
                                          1).to(self.device)
        self.post_quant_conv = torch.nn.Conv2d(opt['embed_dim'],
                                               opt["z_channels"],
                                               1).to(self.device)

    def init_training_settings(self):
        self.loss = BCELossWithQuant()
        self.log_dict = OrderedDict()
        self.configure_optimizers()

    def save_network(self, save_path):
        """Save networks.

        Args:
            net (nn.Module): Network to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
        """

        save_dict = {}
        save_dict['encoder'] = self.encoder.state_dict()
        save_dict['decoder'] = self.decoder.state_dict()
        save_dict['quantize'] = self.quantize.state_dict()
        save_dict['quant_conv'] = self.quant_conv.state_dict()
        save_dict['post_quant_conv'] = self.post_quant_conv.state_dict()
        save_dict['discriminator'] = self.disc.state_dict()
        torch.save(save_dict, save_path)

    def load_network(self):
        checkpoint = torch.load(self.opt['pretrained_models'])
        self.encoder.load_state_dict(checkpoint['encoder'], strict=True)
        self.decoder.load_state_dict(checkpoint['decoder'], strict=True)
        self.quantize.load_state_dict(checkpoint['quantize'], strict=True)
        self.quant_conv.load_state_dict(checkpoint['quant_conv'], strict=True)
        self.post_quant_conv.load_state_dict(
            checkpoint['post_quant_conv'], strict=True)

    def optimize_parameters(self, data, current_iter):
        self.encoder.train()
        self.decoder.train()
        self.quantize.train()
        self.quant_conv.train()
        self.post_quant_conv.train()

        loss = self.training_step(data)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward_step(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def feed_data(self, data):
        x = data['segm']
        x = F.one_hot(x, num_classes=self.opt['num_segm_classes'])

        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float().to(self.device)

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


class VQSegmentationModel(VQModel):

    def __init__(self, opt):
        super().__init__(opt)
        self.colorize = torch.randn(3, opt['num_segm_classes'], 1,
                                    1).to(self.device)

        self.init_training_settings()

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()) +
            list(self.quantize.parameters()) +
            list(self.quant_conv.parameters()) +
            list(self.post_quant_conv.parameters()),
            lr=self.opt['lr'],
            betas=(0.5, 0.9))

    def training_step(self, data):
        x = self.feed_data(data)
        xrec, qloss = self.forward_step(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="train")
        self.log_dict.update(log_dict_ae)
        return aeloss

    def to_rgb(self, x):
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x

    @torch.no_grad()
    def inference(self, data_loader, save_dir):
        self.encoder.eval()
        self.decoder.eval()
        self.quantize.eval()
        self.quant_conv.eval()
        self.post_quant_conv.eval()

        loss_total = 0
        loss_bce = 0
        loss_quant = 0
        num = 0

        for _, data in enumerate(data_loader):
            img_name = data['img_name'][0]
            x = self.feed_data(data)
            xrec, qloss = self.forward_step(x)
            _, log_dict_ae = self.loss(qloss, x, xrec, split="val")

            loss_total += log_dict_ae['val/total_loss']
            loss_bce += log_dict_ae['val/bce_loss']
            loss_quant += log_dict_ae['val/quant_loss']

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

        return (loss_total / num).item(), (loss_bce /
                                           num).item(), (loss_quant /
                                                         num).item()


class VQImageModel(VQModel):

    def __init__(self, opt):
        super().__init__(opt)
        self.disc = Discriminator(
            opt['n_channels'], opt['ndf'],
            n_layers=opt['disc_layers']).to(self.device)
        self.perceptual = lpips.LPIPS(net="vgg").to(self.device)
        self.perceptual_weight = opt['perceptual_weight']
        self.disc_start_step = opt['disc_start_step']
        self.disc_weight_max = opt['disc_weight_max']
        self.diff_aug = opt['diff_aug']
        self.policy = "color,translation"

        self.disc.train()

        self.init_training_settings()

    def feed_data(self, data):
        x = data['image']

        return x.float().to(self.device)

    def init_training_settings(self):
        self.log_dict = OrderedDict()
        self.configure_optimizers()

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()) +
            list(self.quantize.parameters()) +
            list(self.quant_conv.parameters()) +
            list(self.post_quant_conv.parameters()),
            lr=self.opt['lr'])

        self.disc_optimizer = torch.optim.Adam(
            self.disc.parameters(), lr=self.opt['lr'])

    def training_step(self, data, step):
        x = self.feed_data(data)
        xrec, codebook_loss = self.forward_step(x)

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

    def optimize_parameters(self, data, step):
        self.encoder.train()
        self.decoder.train()
        self.quantize.train()
        self.quant_conv.train()
        self.post_quant_conv.train()

        loss, d_loss = self.training_step(data, step)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if step > self.disc_start_step:
            self.disc_optimizer.zero_grad()
            d_loss.backward()
            self.disc_optimizer.step()

    @torch.no_grad()
    def inference(self, data_loader, save_dir):
        self.encoder.eval()
        self.decoder.eval()
        self.quantize.eval()
        self.quant_conv.eval()
        self.post_quant_conv.eval()

        loss_total = 0
        num = 0

        for _, data in enumerate(data_loader):
            img_name = data['img_name'][0]
            x = self.feed_data(data)
            xrec, _ = self.forward_step(x)

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


class VQImageSegmTextureModel(VQImageModel):

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda')
        self.encoder = Encoder(
            ch=opt['ch'],
            num_res_blocks=opt['num_res_blocks'],
            attn_resolutions=opt['attn_resolutions'],
            ch_mult=opt['ch_mult'],
            in_channels=opt['in_channels'],
            resolution=opt['resolution'],
            z_channels=opt['z_channels'],
            double_z=opt['double_z'],
            dropout=opt['dropout']).to(self.device)
        self.decoder = Decoder(
            in_channels=opt['in_channels'],
            resolution=opt['resolution'],
            z_channels=opt['z_channels'],
            ch=opt['ch'],
            out_ch=opt['out_ch'],
            num_res_blocks=opt['num_res_blocks'],
            attn_resolutions=opt['attn_resolutions'],
            ch_mult=opt['ch_mult'],
            dropout=opt['dropout'],
            resamp_with_conv=True,
            give_pre_end=False).to(self.device)
        self.quantize = VectorQuantizerTexture(
            opt['n_embed'], opt['embed_dim'], beta=0.25).to(self.device)
        self.quant_conv = torch.nn.Conv2d(opt["z_channels"], opt['embed_dim'],
                                          1).to(self.device)
        self.post_quant_conv = torch.nn.Conv2d(opt['embed_dim'],
                                               opt["z_channels"],
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

        self.disc.train()

        self.init_training_settings()

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
        self.encoder.eval()
        self.decoder.eval()
        self.quantize.eval()
        self.quant_conv.eval()
        self.post_quant_conv.eval()

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

    def encode(self, x, mask):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h, mask)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward_step(self, input, mask):
        quant, diff, _ = self.encode(input, mask)
        dec = self.decode(quant)
        return dec, diff
