import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import xavier_uniform_


class CrossTransformer(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, input1, input2):
        attn_output, attn_weight = self.attention(input1, input2, input2)
        output = input1 + self.dropout1(attn_output)
        output = self.norm1(output)
        return output


class DIRL(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.feat_dim = cfg.model.transformer_encoder.feat_dim
        self.att_dim = cfg.model.transformer_encoder.att_dim
        self.att_head = cfg.model.transformer_encoder.att_head

        self.embed_dim = cfg.model.transformer_encoder.emb_dim

        self.img = nn.Sequential(
            nn.Conv2d(self.feat_dim, self.att_dim, kernel_size=1, padding=0),
        )

        self.w_embedding = nn.Embedding(14, int(self.att_dim / 2))
        self.h_embedding = nn.Embedding(14, int(self.att_dim / 2))

        self.mlp = nn.Sequential(
            nn.Linear(self.att_dim, self.att_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.att_dim * 4, self.att_dim)
        )

        self.num_hidden_layers = cfg.model.transformer_encoder.att_layer
        self.transformer = nn.ModuleList([CrossTransformer(self.att_dim, self.att_head)
                                          for i in range(self.num_hidden_layers)])

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, input_1, input_2):
        batch_size, C, H, W = input_1.size()

        input_1 = self.img(input_1)  # (128,196, 512)
        input_2 = self.img(input_2)

        pos_w = torch.arange(W).cuda()
        pos_h = torch.arange(H).cuda()
        embed_w = self.w_embedding(pos_w)
        embed_h = self.h_embedding(pos_h)
        position_embedding = torch.cat([embed_w.unsqueeze(0).repeat(W, 1, 1),
                                        embed_h.unsqueeze(1).repeat(1, H, 1)],
                                       dim=-1)

        position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch_size, 1, 1,
                                                                                     1)  # (batch, d_model, h, w)
        input_1 = input_1 + position_embedding  # (batch, att_dim, h, w)
        input_2 = input_2 + position_embedding

        input_1 = input_1.view(batch_size, self.att_dim, -1).permute(0, 2, 1)
        input_2 = input_2.view(batch_size, self.att_dim, -1).permute(0, 2, 1)

        #####################
        img_mask = torch.Tensor(np.ones([batch_size, H*W])).cuda()
        input_1_feat = input_1[img_mask.bool()]
        input_2_feat = input_2[img_mask.bool()]
        input_1_feat = self.mlp(input_1_feat)
        input_2_feat = self.mlp(input_2_feat)
        input_1_feat = input_1_feat
        input_2_feat = input_2_feat
        z_a_norm = (input_1_feat - input_1_feat.mean(0)) / input_1_feat.std(0)  # NxN_sxD
        z_b_norm = (input_2_feat - input_2_feat.mean(0)) / input_2_feat.std(0)  # NxN_txD
        # cross-correlation matrix
        B, D = z_a_norm.shape
        c = (z_a_norm.T @ z_b_norm)  # DxD
        c.div_(B)
        # loss
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = c.flatten()[1:].view(D - 1, D + 1)[:, :-1].pow_(2).sum()
        cdcr_loss = (on_diag * 1 + off_diag * 0.003)
        ###################################################

        input_1 = input_1.transpose(0, 1)
        input_2 = input_2.transpose(0, 1)

        input_1_pre = input_1 # h*w, b, att_dim
        input_2_pre = input_2

        for l in self.transformer:
            input_1, input_2 = l(input_1, input_2), l(input_2, input_1)

        input_1_diff = (input_1_pre - input_1).permute(1, 0, 2)
        input_2_diff = (input_2_pre - input_2).permute(1, 0, 2)

        return input_1_diff, input_2_diff, cdcr_loss


class AddSpatialInfo(nn.Module):

    def _create_coord(self, img_feat):
        batch_size, _, h, w = img_feat.size()
        coord_map = img_feat.new_zeros(2, h, w)
        for i in range(h):
            for j in range(w):
                coord_map[0][i][j] = (j * 2.0 / w) - 1
                coord_map[1][i][j] = (i * 2.0 / h) - 1
        sequence = [coord_map] * batch_size
        coord_map_in_batch = torch.stack(sequence)
        return coord_map_in_batch

    def forward(self, img_feat):
        coord_map = self._create_coord(img_feat)
        img_feat_aug = torch.cat([img_feat, coord_map], dim=1)
        return img_feat_aug

