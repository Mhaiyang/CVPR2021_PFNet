"""
 @Time    : 2021/7/6 14:31
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : CVPR2021_PFNet
 @File    : loss.py
 @Function: Loss
 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

###################################################################
# ########################## edge loss #############################
###################################################################
class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        laplace = torch.FloatTensor([[-1, -1, -1, ], [-1, 8, -1], [-1, -1, -1]]).view([1, 1, 3, 3])
        # filter shape in Pytorch: out_channel, in_channel, height, width
        self.laplace = nn.Parameter(data=laplace, requires_grad=False)
        self.l1_loss = nn.L1Loss()

    def torchLaplace(self, x):
        edge = torch.abs(F.conv2d(x, self.laplace, padding=1))
        return edge

    def forward(self, y_pred, y_true):
        y_true_edge = self.torchLaplace(y_true)
        y_pred = torch.sigmoid(y_pred)
        y_pred_edge = self.torchLaplace(y_pred)
        edge_loss = self.l1_loss(y_pred_edge, y_true_edge)

        return edge_loss

###################################################################
# ########################## iou loss #############################
###################################################################
class IOU(torch.nn.Module):
    def __init__(self):
        super(IOU, self).__init__()

    def _iou(self, pred, target):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(2, 3))
        union = (pred + target).sum(dim=(2, 3)) - inter
        iou = 1 - (inter / union)

        return iou.mean()

    def forward(self, pred, target):
        return self._iou(pred, target)
