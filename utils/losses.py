import torch
import torch.nn as nn


class EggsPansLoss:

    def __init__(self, jaccard_weight=0.5, bce_weight=0.5):

        # Keep loss weights
        self.jaccard_weight = jaccard_weight
        self.bce_weight = bce_weight

        # Initialize losses
        self.bce_loss = nn.BCEWithLogitsLoss()

        # Initialize class-wise coefficients
        pixel_ratio = {'egg': 6.9024,
                       'pan': 1.8904}

        self.egg_coeff = 1.0
        self.pan_coeff = pixel_ratio['egg'] / pixel_ratio['pan']

        # Running loss
        self.running_loss = 0.0
        self.running_samples = 0

    def __call__(self, gt, pred):

        gt_egg, gt_pan = gt
        pred_egg, pred_pan = pred

        # BCE Loss
        loss_egg = self.bce_loss(pred_egg, gt_egg) * self.bce_weight
        loss_pan = self.bce_loss(pred_pan, gt_pan) * self.bce_weight

        # Jaccard Loss
        loss_egg += get_jaccard_loss(gt_egg, pred_egg) * self.jaccard_weight
        loss_pan += get_jaccard_loss(gt_pan, pred_pan) * self.jaccard_weight

        # Multiply by coefficients
        loss = loss_egg * self.egg_coeff + loss_pan * self.pan_coeff

        self.running_loss += loss.item()
        self.running_samples += 1

        return loss

    def get_running_loss(self):

        if self.running_samples == 0:
            return 0

        else:
            return self.running_loss / self.running_samples

    def reset_loss(self):
        self.running_loss = 0.0
        self.running_samples = 0


def get_iou(gt, pred, eps=1.0):
    inter = (gt * pred).sum()
    union = gt.sum() + pred.sum() - inter

    return (inter + eps) / (union + eps)


def get_jaccard_loss(gt, pred):
    pred = torch.sigmoid(pred)
    iou = get_iou(gt, pred)
    return 1 - iou

