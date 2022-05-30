import torch
import torch.nn.functional as F
from torch import nn


class ShapeAttrEmbedding(nn.Module):

    def __init__(self, dim, out_dim, cls_num_list):
        super(ShapeAttrEmbedding, self).__init__()

        for idx, cls_num in enumerate(cls_num_list):
            setattr(
                self, f'attr_{idx}',
                nn.Sequential(
                    nn.Linear(cls_num, dim), nn.LeakyReLU(),
                    nn.Linear(dim, dim)))
        self.cls_num_list = cls_num_list
        self.attr_num = len(cls_num_list)
        self.fusion = nn.Sequential(
            nn.Linear(dim * self.attr_num, out_dim), nn.LeakyReLU(),
            nn.Linear(out_dim, out_dim))

    def forward(self, attr):
        attr_embedding_list = []
        for idx in range(self.attr_num):
            attr_embed_fc = getattr(self, f'attr_{idx}')
            attr_embedding_list.append(
                attr_embed_fc(
                    F.one_hot(
                        attr[:, idx],
                        num_classes=self.cls_num_list[idx]).to(torch.float32)))
        attr_embedding = torch.cat(attr_embedding_list, dim=1)
        attr_embedding = self.fusion(attr_embedding)

        return attr_embedding
