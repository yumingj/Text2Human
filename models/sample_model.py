import logging

import numpy as np
import torch
import torch.distributions as dists
import torch.nn.functional as F
from torchvision.utils import save_image

from models.archs.fcn_arch import FCNHead, MultiHeadFCNHead
from models.archs.shape_attr_embedding_arch import ShapeAttrEmbedding
from models.archs.transformer_arch import TransformerMultiHead
from models.archs.unet_arch import ShapeUNet, UNet
from models.archs.vqgan_arch import (Decoder, DecoderRes, Encoder,
                                     VectorQuantizer,
                                     VectorQuantizerSpatialTextureAware,
                                     VectorQuantizerTexture)

logger = logging.getLogger('base')


class BaseSampleModel():
    """Base Model"""

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda')

        # hierarchical VQVAE
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
        self.top_post_quant_conv = torch.nn.Conv2d(opt['embed_dim'],
                                                   opt["top_z_channels"],
                                                   1).to(self.device)
        self.load_top_pretrain_models()

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
            spatial_size=opt['bot_codebook_spatial_size']).to(self.device)
        self.bot_post_quant_conv = torch.nn.Conv2d(opt['embed_dim'],
                                                   opt["bot_z_channels"],
                                                   1).to(self.device)
        self.load_bot_pretrain_network()

        # top -> bot prediction
        self.index_pred_guidance_encoder = UNet(
            in_channels=opt['index_pred_encoder_in_channels']).to(self.device)
        self.index_pred_decoder = MultiHeadFCNHead(
            in_channels=opt['index_pred_fc_in_channels'],
            in_index=opt['index_pred_fc_in_index'],
            channels=opt['index_pred_fc_channels'],
            num_convs=opt['index_pred_fc_num_convs'],
            concat_input=opt['index_pred_fc_concat_input'],
            dropout_ratio=opt['index_pred_fc_dropout_ratio'],
            num_classes=opt['index_pred_fc_num_classes'],
            align_corners=opt['index_pred_fc_align_corners'],
            num_head=18).to(self.device)
        self.load_index_pred_network()

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
        self.load_pretrained_segm_token()

        # define sampler
        self.sampler_fn = TransformerMultiHead(
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
        self.load_sampler_pretrained_network()

        self.shape = tuple(opt['latent_shape'])

        self.mask_id = opt['codebook_size']
        self.sample_steps = opt['sample_steps']

    def load_top_pretrain_models(self):
        # load pretrained vqgan
        top_vae_checkpoint = torch.load(self.opt['top_vae_path'])

        self.decoder.load_state_dict(
            top_vae_checkpoint['decoder'], strict=True)
        self.top_quantize.load_state_dict(
            top_vae_checkpoint['quantize'], strict=True)
        self.top_post_quant_conv.load_state_dict(
            top_vae_checkpoint['post_quant_conv'], strict=True)

        self.decoder.eval()
        self.top_quantize.eval()
        self.top_post_quant_conv.eval()

    def load_bot_pretrain_network(self):
        checkpoint = torch.load(self.opt['bot_vae_path'])
        self.bot_decoder_res.load_state_dict(
            checkpoint['bot_decoder_res'], strict=True)
        self.decoder.load_state_dict(checkpoint['decoder'], strict=True)
        self.bot_quantize.load_state_dict(
            checkpoint['bot_quantize'], strict=True)
        self.bot_post_quant_conv.load_state_dict(
            checkpoint['bot_post_quant_conv'], strict=True)

        self.bot_decoder_res.eval()
        self.decoder.eval()
        self.bot_quantize.eval()
        self.bot_post_quant_conv.eval()

    def load_pretrained_segm_token(self):
        # load pretrained vqgan for segmentation mask
        segm_token_checkpoint = torch.load(self.opt['segm_token_path'])
        self.segm_encoder.load_state_dict(
            segm_token_checkpoint['encoder'], strict=True)
        self.segm_quantizer.load_state_dict(
            segm_token_checkpoint['quantize'], strict=True)
        self.segm_quant_conv.load_state_dict(
            segm_token_checkpoint['quant_conv'], strict=True)

        self.segm_encoder.eval()
        self.segm_quantizer.eval()
        self.segm_quant_conv.eval()

    def load_index_pred_network(self):
        checkpoint = torch.load(self.opt['pretrained_index_network'])
        self.index_pred_guidance_encoder.load_state_dict(
            checkpoint['guidance_encoder'], strict=True)
        self.index_pred_decoder.load_state_dict(
            checkpoint['index_decoder'], strict=True)

        self.index_pred_guidance_encoder.eval()
        self.index_pred_decoder.eval()

    def load_sampler_pretrained_network(self):
        checkpoint = torch.load(self.opt['pretrained_sampler'])
        self.sampler_fn.load_state_dict(checkpoint, strict=True)
        self.sampler_fn.eval()

    def bot_index_prediction(self, feature_top, texture_mask):
        self.index_pred_guidance_encoder.eval()
        self.index_pred_decoder.eval()

        texture_tokens = F.interpolate(
            texture_mask, (32, 16), mode='nearest').view(self.batch_size,
                                                         -1).long()

        texture_mask_flatten = texture_tokens.view(-1)
        min_encodings_indices_list = [
            torch.full(
                texture_mask_flatten.size(),
                fill_value=-1,
                dtype=torch.long,
                device=texture_mask_flatten.device) for _ in range(18)
        ]
        with torch.no_grad():
            feature_enc = self.index_pred_guidance_encoder(feature_top)
            memory_logits_list = self.index_pred_decoder(feature_enc)
            for codebook_idx, memory_logits in enumerate(memory_logits_list):
                region_of_interest = texture_mask_flatten == codebook_idx
                if torch.sum(region_of_interest) > 0:
                    memory_indices_pred = memory_logits.argmax(dim=1).view(-1)
                    memory_indices_pred = memory_indices_pred
                    min_encodings_indices_list[codebook_idx][
                        region_of_interest] = memory_indices_pred[
                            region_of_interest]
            min_encodings_indices_return_list = [
                min_encodings_indices.view((1, 32, 16))
                for min_encodings_indices in min_encodings_indices_list
            ]

        return min_encodings_indices_return_list

    def sample_and_refine(self, save_dir=None, img_name=None):
        # sample 32x16 features indices
        sampled_top_indices_list = self.sample_fn(
            temp=1, sample_steps=self.sample_steps)

        for sample_idx in range(self.batch_size):
            sample_indices = [
                sampled_indices_cur[sample_idx:sample_idx + 1]
                for sampled_indices_cur in sampled_top_indices_list
            ]
            top_quant = self.top_quantize.get_codebook_entry(
                sample_indices, self.texture_mask[sample_idx:sample_idx + 1],
                (sample_indices[0].size(0), self.shape[0], self.shape[1],
                 self.opt["top_z_channels"]))

            top_quant = self.top_post_quant_conv(top_quant)

            bot_indices_list = self.bot_index_prediction(
                top_quant, self.texture_mask[sample_idx:sample_idx + 1])

            quant_bot = self.bot_quantize.get_codebook_entry(
                bot_indices_list, self.texture_mask[sample_idx:sample_idx + 1],
                (bot_indices_list[0].size(0), bot_indices_list[0].size(1),
                 bot_indices_list[0].size(2),
                 self.opt["bot_z_channels"]))  #.permute(0, 3, 1, 2)
            quant_bot = self.bot_post_quant_conv(quant_bot)
            bot_dec_res = self.bot_decoder_res(quant_bot)

            dec = self.decoder(top_quant, bot_h=bot_dec_res)

            dec = ((dec + 1) / 2)
            dec = dec.clamp_(0, 1)
            if save_dir is None and img_name is None:
                return dec
            else:
                save_image(
                    dec,
                    f'{save_dir}/{img_name[sample_idx]}',
                    nrow=1,
                    padding=4)

    def sample_fn(self, temp=1.0, sample_steps=None):
        self.sampler_fn.eval()

        x_t = torch.ones((self.batch_size, np.prod(self.shape)),
                         device=self.device).long() * self.mask_id
        unmasked = torch.zeros_like(x_t, device=self.device).bool()
        sample_steps = list(range(1, sample_steps + 1))

        texture_tokens = F.interpolate(
            self.texture_mask, (32, 16),
            mode='nearest').view(self.batch_size, -1).long()

        texture_mask_flatten = texture_tokens.view(-1)

        # min_encodings_indices_list would be used to visualize the image
        min_encodings_indices_list = [
            torch.full(
                texture_mask_flatten.size(),
                fill_value=-1,
                dtype=torch.long,
                device=texture_mask_flatten.device) for _ in range(18)
        ]

        for t in reversed(sample_steps):
            t = torch.full((self.batch_size, ),
                           t,
                           device=self.device,
                           dtype=torch.long)

            # where to unmask
            changes = torch.rand(
                x_t.shape, device=self.device) < 1 / t.float().unsqueeze(-1)
            # don't unmask somewhere already unmasked
            changes = torch.bitwise_xor(changes,
                                        torch.bitwise_and(changes, unmasked))
            # update mask with changes
            unmasked = torch.bitwise_or(unmasked, changes)

            x_0_logits_list = self.sampler_fn(
                x_t, self.segm_tokens, texture_tokens, t=t)

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

        self.sampler_fn.train()

        return min_encodings_indices_return_list

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


class SampleFromParsingModel(BaseSampleModel):
    """SampleFromParsing model.
    """

    def feed_data(self, data):
        self.segm = data['segm'].to(self.device)
        self.texture_mask = data['texture_mask'].to(self.device)
        self.batch_size = self.segm.size(0)

        self.segm_tokens = self.get_quantized_segm(self.segm)
        self.segm_tokens = self.segm_tokens.view(self.batch_size, -1)

    def inference(self, data_loader, save_dir):
        for _, data in enumerate(data_loader):
            img_name = data['img_name']
            self.feed_data(data)
            with torch.no_grad():
                self.sample_and_refine(save_dir, img_name)


class SampleFromPoseModel(BaseSampleModel):
    """SampleFromPose model.
    """

    def __init__(self, opt):
        super().__init__(opt)
        # pose-to-parsing
        self.shape_attr_embedder = ShapeAttrEmbedding(
            dim=opt['shape_embedder_dim'],
            out_dim=opt['shape_embedder_out_dim'],
            cls_num_list=opt['shape_attr_class_num']).to(self.device)
        self.shape_parsing_encoder = ShapeUNet(
            in_channels=opt['shape_encoder_in_channels']).to(self.device)
        self.shape_parsing_decoder = FCNHead(
            in_channels=opt['shape_fc_in_channels'],
            in_index=opt['shape_fc_in_index'],
            channels=opt['shape_fc_channels'],
            num_convs=opt['shape_fc_num_convs'],
            concat_input=opt['shape_fc_concat_input'],
            dropout_ratio=opt['shape_fc_dropout_ratio'],
            num_classes=opt['shape_fc_num_classes'],
            align_corners=opt['shape_fc_align_corners'],
        ).to(self.device)
        self.load_shape_generation_models()

        self.palette = [[0, 0, 0], [255, 250, 250], [220, 220, 220],
                        [250, 235, 215], [255, 250, 205], [211, 211, 211],
                        [70, 130, 180], [127, 255, 212], [0, 100, 0],
                        [50, 205, 50], [255, 255, 0], [245, 222, 179],
                        [255, 140, 0], [255, 0, 0], [16, 78, 139],
                        [144, 238, 144], [50, 205, 174], [50, 155, 250],
                        [160, 140, 88], [213, 140, 88], [90, 140, 90],
                        [185, 210, 205], [130, 165, 180], [225, 141, 151]]

    def load_shape_generation_models(self):
        checkpoint = torch.load(self.opt['pretrained_parsing_gen'])

        self.shape_attr_embedder.load_state_dict(
            checkpoint['embedder'], strict=True)
        self.shape_attr_embedder.eval()

        self.shape_parsing_encoder.load_state_dict(
            checkpoint['encoder'], strict=True)
        self.shape_parsing_encoder.eval()

        self.shape_parsing_decoder.load_state_dict(
            checkpoint['decoder'], strict=True)
        self.shape_parsing_decoder.eval()

    def feed_data(self, data):
        self.pose = data['densepose'].to(self.device)
        self.batch_size = self.pose.size(0)

        self.shape_attr = data['shape_attr'].to(self.device)
        self.upper_fused_attr = data['upper_fused_attr'].to(self.device)
        self.lower_fused_attr = data['lower_fused_attr'].to(self.device)
        self.outer_fused_attr = data['outer_fused_attr'].to(self.device)

    def inference(self, data_loader, save_dir):
        for _, data in enumerate(data_loader):
            img_name = data['img_name']
            self.feed_data(data)
            with torch.no_grad():
                self.generate_parsing_map()
                self.generate_quantized_segm()
                self.generate_texture_map()
                self.sample_and_refine(save_dir, img_name)

    def generate_parsing_map(self):
        with torch.no_grad():
            attr_embedding = self.shape_attr_embedder(self.shape_attr)
            pose_enc = self.shape_parsing_encoder(self.pose, attr_embedding)
            seg_logits = self.shape_parsing_decoder(pose_enc)
        self.segm = seg_logits.argmax(dim=1)
        self.segm = self.segm.unsqueeze(1)

    def generate_quantized_segm(self):
        self.segm_tokens = self.get_quantized_segm(self.segm)
        self.segm_tokens = self.segm_tokens.view(self.batch_size, -1)

    def generate_texture_map(self):
        upper_cls = [1., 4.]
        lower_cls = [3., 5., 21.]
        outer_cls = [2.]

        mask_batch = []
        for idx in range(self.batch_size):
            mask = torch.zeros_like(self.segm[idx])
            upper_fused_attr = self.upper_fused_attr[idx]
            lower_fused_attr = self.lower_fused_attr[idx]
            outer_fused_attr = self.outer_fused_attr[idx]
            if upper_fused_attr != 17:
                for cls in upper_cls:
                    mask[self.segm[idx] == cls] = upper_fused_attr + 1

            if lower_fused_attr != 17:
                for cls in lower_cls:
                    mask[self.segm[idx] == cls] = lower_fused_attr + 1

            if outer_fused_attr != 17:
                for cls in outer_cls:
                    mask[self.segm[idx] == cls] = outer_fused_attr + 1

            mask_batch.append(mask)
        self.texture_mask = torch.stack(mask_batch, dim=0).to(torch.float32)

    def feed_pose_data(self, pose_img):
        # for ui demo

        self.pose = pose_img.to(self.device)
        self.batch_size = self.pose.size(0)

    def feed_shape_attributes(self, shape_attr):
        # for ui demo

        self.shape_attr = shape_attr.to(self.device)

    def feed_texture_attributes(self, texture_attr):
        # for ui demo

        self.upper_fused_attr = texture_attr[0].unsqueeze(0).to(self.device)
        self.lower_fused_attr = texture_attr[1].unsqueeze(0).to(self.device)
        self.outer_fused_attr = texture_attr[2].unsqueeze(0).to(self.device)

    def palette_result(self, result):

        seg = result[0]
        palette = np.array(self.palette)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        # color_seg = color_seg[..., ::-1]
        return color_seg
