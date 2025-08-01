import torch
import torch.nn as nn
import torch.nn.functional as F

# 子图-分子交叉注意力模块
class CrossGraphAttention(nn.Module):
    def __init__(self, dim):
        super(CrossGraphAttention, self).__init__()
        # 确保输入维度与预期一致
        self.query = nn.Linear(100, dim)
        self.key = nn.Linear(75, dim)
        self.value = nn.Linear(75, dim)
        # 添加一个线性层来调整输出维度以匹配 chem_feat 的维度
        self.output_transform = nn.Linear(dim, 100)

    def forward(self, graph_feat, chem_feat):
        Q = self.query(chem_feat)  # (batch_size, dim)
        K = self.key(graph_feat)   # (batch_size, dim)
        V = self.value(graph_feat) # (batch_size, dim)

        # 计算注意力权重
        attn = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(K.shape[-1], dtype=torch.float32))
        attn = F.softmax(attn, dim=-1)  # (batch_size, batch_size)

        # 应用注意力权重到值向量
        weighted_sum = torch.matmul(attn, V)  # (batch_size, dim)

        # 调整输出维度以匹配 chem_feat
        output = self.output_transform(weighted_sum)  # (batch_size, 100)

        # 残差连接
        return output + chem_feat

# 文本引导融合模块
class TextGuidedGate(nn.Module):
    def __init__(self, text_dim, fusion_dim):
        super(TextGuidedGate, self).__init__()
        # 将 fused_graph_chem 投影到 text_dim
        self.fusion_proj = nn.Linear(fusion_dim, text_dim)
        # 门控层输入维度为 text_dim * 2（text_feat + 投影后的 fused_graph_chem）
        self.gate = nn.Sequential(
            nn.Linear(text_dim * 2, text_dim),
            nn.Sigmoid()
        )

    def forward(self, text_feat, fused_graph_chem):
        # 投影 fused_graph_chem 到 text_dim
        fused_graph_chem_proj = self.fusion_proj(fused_graph_chem)
        # 拼接文本特征和投影后的子图-分子特征
        combined = torch.cat([text_feat, fused_graph_chem_proj], dim=-1)
        # 计算门控值
        gate = self.gate(combined)
        # 门控融合
        return gate * fused_graph_chem_proj + (1 - gate) * text_feat