import numpy as np
from math import sqrt
import torch
import torch.nn as nn
from masking import TriangularCausalMask, ProbMask
# 这段代码实现了全注意力机制和概率注意力机制，并封装了一个注意力层，方便在深度学习模型中使用。
# 全注意力机制计算所有查询和键之间的注意力分数，而概率注意力机制通过采样和选择重要的查询来减少计算量。注意力层负责对输入进行投影和输出投影，以适应不同的特征维度和注意力头数量


class FullAttention(nn.Module):
 # 实现了全注意力机制，即计算所有查询和键之间的注意力分数。
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()


class ProbAttention(nn.Module):
 # 实现了概率注意力机制，通过采样和选择重要的查询来减少计算量。
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
    # 初始化参数：
    # mask_flag：是否使用掩码，默认为 True。
    # factor：控制采样数量的因子，默认为 5。
    # scale：注意力分数的缩放因子，默认为 None。
    # attention_dropout：注意力分数的丢弃率，默认为 0.1

        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)



 # 对查询、键和值进行形状调整。
 # 计算采样数量 U 和 u。
 # 使用 _prob_QK 函数计算重要查询的得分和索引。
 # 对得分进行缩放。
 # 使用 _get_initial_context 函数初始化上下文。
 # 使用 _update_context 函数更新上下文。
    def _prob_QK(self, Q, K, sample_k, n_top):
            # Q [B, H, L, D]
            B, H, L, E = K.shape
            _, _, S, _ = Q.shape

            # 确保 n_top 不超过序列长度 S
            n_top = min(n_top, S)  # 新增的边界检查

            # calculate the sampled Q_K
            K_expand = K.unsqueeze(-3).expand(B, H, S, L, E)
            indx_sample = torch.randint(L, (S, sample_k))
            K_sample = K_expand[:, :, torch.arange(S).unsqueeze(1), indx_sample, :]
            Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

            # find the Top_k query with sparisty measurement
            M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L)
            M_top = M.topk(n_top, sorted=False)[1]

            # use the reduced Q to calculate Q_K
            Q_reduce = Q[torch.arange(B)[:, None, None],
                       torch.arange(H)[None, :, None],
                       M_top, :]
            Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

            return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.sum(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-1)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V)
        return context_in

    def forward(self, queries, keys, values, attn_mask):
            B, L, H, D = queries.shape
            _, S, _, _ = keys.shape

            queries = queries.view(B, H, L, -1)
            keys = keys.view(B, H, S, -1)
            values = values.view(B, H, S, -1)

            # 计算 U 时确保不超过序列长度 S
            max_possible = max(1, int(np.ceil(np.log(S))))  # 至少为1
            U = min(self.factor * max_possible, S)  # 确保 U 不超过 S
            u = self.factor * np.ceil(np.log(L)).astype('int').item()

            scores_top, index = self._prob_QK(queries, keys, u, U)
            # add scale factor
            scale = self.scale or 1. / sqrt(D)
            if scale is not None:
                scores_top = scores_top * scale
            # get the context
            context = self._get_initial_context(values, L)
            # update the context with selected top_k queries
            context = self._update_context(context, values, scores_top, index, L, attn_mask)

            return context.contiguous()

class AttentionLayer(nn.Module):
 # 封装了注意力机制，包括查询、键和值的投影层以及输出投影层。
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        # print(queries.shape)
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        ).view(B, L, -1)

        return self.out_projection(out)