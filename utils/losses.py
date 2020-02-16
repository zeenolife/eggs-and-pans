def get_iou(gt, pred, eps=1.0):
    inter = (gt * pred).sum()
    union = gt.sum() + pred.sum() - inter + eps

    return (inter + eps) / union


def get_jaccard_loss_and_iou(gt, pred):
    iou = get_iou(gt, pred)
    return 1 - iou, iou

