import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算二分类的交叉熵损失
        bce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算每个样本的权重
        pt = torch.exp(-bce_loss)
        weights = self.alpha * (1 - pt)**self.gamma

        # 计算Focal Loss
        focal_loss = weights * bce_loss

        # 根据reduction参数选择损失函数的计算方式
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
