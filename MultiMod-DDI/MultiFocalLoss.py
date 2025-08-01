# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# 它实现了 Focal Loss 并支持平滑标签交叉熵。
# Focal Loss 是一种用于解决类别不平衡问题和提高模型对难分类样本关注度的损失函数
# MultiFocalLoss特别适合于多类别分类和目标检测任务，这些任务中类别不平衡和样本难易分类的问题尤为突出
class MultiFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average
        # self.cut_off = cut_off

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        # 确保输入无非法值
        assert not torch.isnan(input).any(), "Input contains NaN!"
        assert not torch.isinf(input).any(), "Input contains Inf!"

        logit = F.softmax(input, dim=1)
        device = input.device  # 统一设备管理

        # 将alpha转移到当前设备
        if self.alpha.device != device:
            self.alpha = self.alpha.to(device)

        # 处理高维输入（保持原始逻辑）
        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))

        # 转换target为one-hot编码
        target = target.view(-1, 1)
        idx = target.long()  # 直接使用当前设备上的tensor
        one_hot_key = torch.zeros(target.size(0), self.num_class, device=device)  # 直接在目标设备创建
        one_hot_key.scatter_(1, idx, 1)

        # 标签平滑处理
        if self.smooth is not None:
            one_hot_key = torch.clamp(one_hot_key, self.smooth, 1.0 - self.smooth)

        # 核心计算（增加数值保护）
        pt = (one_hot_key * logit).sum(dim=1)
        pt = torch.clamp(pt, min=1e-7, max=1.0 - 1e-7)  # 关键截断
        logpt = torch.log(pt)

        # 动态alpha选择
        alpha = self.alpha[idx.squeeze()]  # 确保维度匹配

        # 损失计算
        loss = -1 * alpha * torch.pow((1 - pt), self.gamma) * logpt

        # 聚合方式
        return loss.mean() if self.size_average else loss.sum()