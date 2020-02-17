import os
import numpy as np
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from models import UNet
from utils import EggsPansDataset, EggsPansLoss, EggsPansMetricIoU


def train_unet(epoch=100):

    # Get all images in train set
    image_names = os.listdir('dataset/train/images/')
    image_names = [name for name in image_names if name.endswith(('.jpg', '.JPG', '.png'))]

    # Split into train and validation sets
    np.random.shuffle(image_names)
    split = int(len(image_names) * 0.9)
    train_image_names = image_names[:split]
    val_image_names = image_names[split:]

    # Create a dataset
    train_dataset = EggsPansDataset('dataset/train', train_image_names, mode='train')
    val_dataset = EggsPansDataset('dataset/train', val_image_names, mode='val')

    # Create a dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Initialize model and transfer to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet()
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', verbose=True)
    loss_obj = EggsPansLoss()
    metrics_obj = EggsPansMetricIoU()

    # Keep best IoU and checkpoint
    best_iou = 0.0

    # Train epochs
    for epoch_idx in range(epoch):

        print('Epoch: {:2}/{}'.format(epoch_idx + 1, epoch))
        # Reset metrics and loss
        loss_obj.reset_loss()
        metrics_obj.reset_iou()

        # Train phase
        model.train()

        # Train epoch
        pbar = tqdm(train_dataloader)
        for imgs, egg_masks, pan_masks in pbar:

            # Convert to device
            imgs = imgs.to(device)
            gt_egg_masks = egg_masks.to(device)
            gt_pan_masks = pan_masks.to(device)

            # Zero gradients
            optim.zero_grad()

            # Forward through net, and get the loss
            pred_egg_masks, pred_pan_masks = model(imgs)

            loss = loss_obj([gt_egg_masks, gt_pan_masks], [pred_egg_masks, pred_pan_masks])
            iou = metrics_obj([gt_egg_masks, gt_pan_masks], [pred_egg_masks, pred_pan_masks])

            # Compute gradients and compute them
            loss.backward()
            optim.step()

            # Update metrics
            pbar.set_description('Loss: {:5.6f}, IoU: {:5.6f}'.format(loss_obj.get_running_loss(),
                                                                      metrics_obj.get_running_iou()))

        print('Validation: ')

        # Reset metrics and loss
        loss_obj.reset_loss()
        metrics_obj.reset_iou()

        # Val phase
        model.eval()

        # Val epoch
        pbar = tqdm(val_dataloader)
        for imgs, egg_masks, pan_masks in pbar:

            # Convert to device
            imgs = imgs.to(device)
            gt_egg_masks = egg_masks.to(device)
            gt_pan_masks = pan_masks.to(device)

            with torch.no_grad():
                # Forward through net, and get the loss
                pred_egg_masks, pred_pan_masks = model(imgs)

                loss = loss_obj([gt_egg_masks, gt_pan_masks], [pred_egg_masks, pred_pan_masks])
                iou = metrics_obj([gt_egg_masks, gt_pan_masks], [pred_egg_masks, pred_pan_masks])

            pbar.set_description('Val Loss: {:5.6f}, IoU: {:5.6f}'.format(loss_obj.get_running_loss(),
                                                                          metrics_obj.get_running_iou()))

        # Save best model
        if best_iou < metrics_obj.get_running_iou():
            best_iou = metrics_obj.get_running_iou()
            torch.save(model.state_dict(), os.path.join('checkpoints/', 'epoch_{}_{:.4f}.pth'.format(
                epoch_idx + 1, metrics_obj.get_running_iou())))

        # Reduce learning rate on plateau
        lr_scheduler.step(metrics_obj.get_running_iou())

        print('\n')
        print('-'*100)


if __name__ == '__main__':
    # Set random seed
    np.random.seed(42)
    train_unet()
