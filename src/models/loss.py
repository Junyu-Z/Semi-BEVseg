import torch
import torch.nn as nn


class Dice_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid_func = nn.Sigmoid()
    
    def forward(self, logits, labels, *args):  # B x 14 x H x W, B x 14 x H x W
        logits = self.sigmoid_func(logits)  # B x 14 x H x W, 0~1
        # labels = labels.float()  # B x 14 x H x W, 0/1

        intersection = 2 * logits * labels  # B x 14 x H x W
        union = (logits + labels)  # B x 14 x H x W
        
        intersection = intersection.sum(dim=0).sum(dim=-1).sum(dim=-1)  # 14
        union = union.sum(dim=0).sum(dim=-1).sum(dim=-1)  # 14
        
        iou = intersection / (union + 1e-5)  # 14
        
        dice_loss = 1 - iou.mean()
        
        return dice_loss

