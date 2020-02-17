import torch
from utils import get_iou


class EggsPansMetricIoU:
    def __init__(self):

        self.running_iou = 0.0
        self.running_samples = 0

    def __call__(self, gt, pred):

        gt_egg, gt_pan = gt
        pred_egg, pred_pan = pred

        # Apply sigmoid and threshold by 0.5
        pred_egg = (torch.sigmoid(pred_egg) >= 0.5).type(pred_egg.dtype)
        pred_pan = (torch.sigmoid(pred_pan) >= 0.5).type(pred_pan.dtype)

        egg_iou = get_iou(gt_egg, pred_egg)
        pan_iou = get_iou(gt_pan, pred_pan)

        self.running_iou += (egg_iou.item() + pan_iou.item()) / 2
        self.running_samples += 1

        return self.running_iou / self.running_samples

    def get_running_iou(self):

        if self.running_samples == 0:
            return 0

        else:
            return self.running_iou / self.running_samples

    def reset_iou(self):
        self.running_iou = 0.0
        self.running_samples = 0
