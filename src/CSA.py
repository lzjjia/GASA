from copy import deepcopy
import torch
from torch import nn
from sklearn import preprocessing
import numpy as np
import warnings
import torch.nn.functional as F

warnings.filterwarnings("ignore", message="Numerical issues were encountered")


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, padding=0, bias=True))

        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(self.dim, self.dim, kernel_size=1)
        self.merge2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):  # (B, dim, l)
        batch_dim = query.size(0)
        query, key, value = [x.contiguous().view(batch_dim, self.dim, self.num_heads)
                             for x in (query, key, value)]
        # query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
        #                      for l, x in zip(self.proj, (query, key, value))]
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge2(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1)).squeeze(-1)
        # return self.merge(x.contiguous().view(-1, batch_dim))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)

    def forward(self, x, source1, source2):
        message = self.attn(x, source1, source2)
        return message


# class AttentionalGNN(nn.Module):
#     def __init__(self, num_support, feature_dim: int, layer_names: list):
#         super().__init__()
#         self.layers = nn.ModuleList([
#             AttentionalPropagation(feature_dim, 4)
#             for _ in range(len(layer_names))])
#         self.names = layer_names
#         self.mlp = MLP([feature_dim * 2, feature_dim * 2, 2048])
#         self.mlp_dis = MLP([2048, feature_dim, feature_dim])
#         nn.init.constant_(self.mlp[-1].bias, 0.0)
#         nn.init.constant_(self.mlp_dis[-1].bias, 0.0)
#         self.mmd = MMD_loss(kernel_type='linear')
#
#     def forward(self, p_nodes_src, p_nodes_tar, dis_nodes_src, dis_nodes_tar, zs, zt):
#         flag = 0
#         dev = p_nodes_src[0].device
#         dis_nodes_src1 = torch.tensor(np.zeros(p_nodes_src.size(), dtype='float32')).to(dev)
#         dis_nodes_tar1 = torch.tensor(np.zeros(p_nodes_src.size(), dtype='float32')).to(dev)
#         # p_nodes_src1 = torch.tensor(np.zeros(dis_nodes_src.size(), dtype='float32')).to(dev)
#         # p_nodes_tar1 = torch.tensor(np.zeros(dis_nodes_tar.size(), dtype='float32')).to(dev)
#
#         for i in range(len(p_nodes_src)):
#             p_nodes_src[i] = torch.tensor(preprocessing.scale(p_nodes_src[i].cpu().detach().numpy())).to(
#                 dev).unsqueeze(0)
#             p_nodes_tar[i] = torch.tensor(preprocessing.scale(p_nodes_tar[i].cpu().detach().numpy())).to(
#                 dev).unsqueeze(0)
#             dis_nodes_src1[i] = self.mlp_dis(
#                 torch.tensor(preprocessing.scale(dis_nodes_src[i].cpu().detach().numpy())).unsqueeze(
#                     0).to(dev))
#             dis_nodes_tar1[i] = self.mlp_dis(
#                 torch.tensor(preprocessing.scale(dis_nodes_tar[i].cpu().detach().numpy())).unsqueeze(
#                     0).to(dev))
#
#         # for layer, name in zip(self.layers, self.names):
#         #
#         #     p_src0, p_src1 = p_nodes_tar[flag].unsqueeze(0), p_nodes_src[flag].unsqueeze(0)   # (B,C,L)
#         #     d_src0, d_src1 = dis_nodes_tar1[flag].unsqueeze(0), dis_nodes_src1[flag].unsqueeze(0)       # (B,C,L)
#         #     delta0, delta1 = layer(p_src1, p_src0, p_src0), layer(p_src0, p_src1, p_src1)   # 1,C,L
#         #     p_nodes_src_temp, p_nodes_tar_temp = torch.einsum('bij,bij->bij', delta0, p_src0), torch.einsum(
#         #         'bij,bij->bij', delta1, p_src1)  # 1,C,L
#         #     delta0, delta1 = layer(dis_nodes_src1[flag].unsqueeze(0), d_src0, p_nodes_src_temp), layer(dis_nodes_tar1[flag].unsqueeze(0), d_src1,
#         #                                                                                  p_nodes_tar_temp)
#         #     p_nodes_src[0] = p_nodes_src[0] + self.mlp(torch.cat([p_nodes_src[flag:flag+1, :], delta0], dim=1))
#         #     p_nodes_tar[0] = p_nodes_tar[0] + self.mlp(torch.cat([p_nodes_tar[flag:flag+1, :], delta1], dim=1))
#         #
#         #     flag += 1
#
#         # p_src0, p_src1 = p_nodes_tar, p_nodes_src  # (B,C,L)
#         # d_src0, d_src1 = dis_nodes_tar1, dis_nodes_src1  # (B,C,L)
#         delta0, delta1 = self.layers[0](p_nodes_src, p_nodes_tar, p_nodes_tar), self.layers[0](p_nodes_tar, p_nodes_src,
#                                                                                                p_nodes_src)  # B,C,L
#         # p_nodes_src_temp, p_nodes_tar_temp = torch.einsum('bij,bij->bij', delta0, p_nodes_tar), torch.einsum(
#         #     'bij,bij->bij', delta1, p_nodes_src)  # B,C,L
#         p_nodes_temp = torch.einsum('bij,bij->bij', delta0, delta1)
#         delta0, delta1 = self.layers[0](dis_nodes_src1, dis_nodes_tar1, p_nodes_temp), self.layers[0](
#             dis_nodes_tar1, dis_nodes_src1, p_nodes_temp)
#         dis_nodes_src = dis_nodes_src + self.mlp(torch.cat([p_nodes_src, delta0], dim=1))
#         dis_nodes_tar = dis_nodes_tar + self.mlp(torch.cat([p_nodes_tar, delta1], dim=1))
#         # p_nodes_src = p_nodes_src + self.mlp(torch.cat([p_nodes_src, delta0], dim=1))
#         # p_nodes_tar = p_nodes_tar + self.mlp(torch.cat([p_nodes_tar, delta1], dim=1))
#
#         # return self.mmd(p_nodes_src[0].squeeze(0).transpose(1, 0), p_nodes_tar[0].squeeze(0).transpose(1, 0))
#         return self.mmd(dis_nodes_src.mean(dim=2), dis_nodes_tar.mean(dim=2))
#         # return dis_nodes_src.mean(dim=2), dis_nodes_tar.mean(dim=2)
#         # return p_nodes_src.mean(dim=2), p_nodes_tar.mean(dim=2)


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        # self.attention = MultiheadAttention(768, num_heads=8)
        self.attention = MultiHeadedAttention(8, feature_dim)
        self.attention2 = MultiHeadedAttention(8, 2048)
        self.Relu = nn.ReLU(inplace=True)
        self.attn_norm = nn.LayerNorm(768)
        self.attn_norm2 = nn.LayerNorm(2048)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, 2048])
        self.mlp_dis = MLP([2048, feature_dim, feature_dim])
        self.mlp_dis2 = MLP([feature_dim,2048,2048])
        nn.init.constant_(self.mlp[-1].bias, 0.0)
        nn.init.constant_(self.mlp_dis[-1].bias, 0.0)

    def forward(self, fs, ft, zs, zt):
        # zs1 = self.mlp_dis(zs.unsqueeze(2)).squeeze(2)
        # zt1 = self.mlp_dis(zs.unsqueeze(2)).squeeze(2)

        delta0 = self.Relu(self.attn_norm(F.dropout(self.attention(fs, ft, ft))))
        delta1 = self.Relu(self.attn_norm(F.dropout(self.attention(ft, fs, fs))))

        delta0 = self.mlp_dis2(delta0.unsqueeze(2)).squeeze(2)
        delta1 = self.mlp_dis2(delta1.unsqueeze(2)).squeeze(2)

        delta0 = self.Relu(self.attn_norm2(F.dropout(self.attention2(zs, zt, delta0))))
        delta1 = self.Relu(self.attn_norm2(F.dropout(self.attention2(zt, zs, delta1))))

        # p_nodes_src_temp, p_nodes_tar_temp = torch.einsum('bi,bi->bi', delta0, ft), torch.einsum(
        #     'bi,bi->bi', delta1, fs)  # B,C
        # delta0 = self.Relu(self.attn_norm(F.dropout(self.attention(zs1, zt1, p_nodes_src_temp))))
        # delta1 = self.Relu(self.attn_norm(F.dropout(self.attention(zt1, zs1, p_nodes_tar_temp))))

        # dis_nodes_src = zs + self.mlp(torch.cat([fs, delta0], dim=1).unsqueeze(2)).squeeze(2)
        # dis_nodes_tar = zt + self.mlp(torch.cat([ft, delta1], dim=1).unsqueeze(2)).squeeze(2)
        dis_nodes_src = zs + delta0
        dis_nodes_tar = zt + delta1

        # p_nodes_src = p_nodes_src + self.mlp(torch.cat([p_nodes_src, delta0], dim=1))
        # p_nodes_tar = p_nodes_tar + self.mlp(torch.cat([p_nodes_tar, delta1], dim=1))

        return dis_nodes_src, dis_nodes_tar

