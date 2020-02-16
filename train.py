import os
import copy
import numpy as np
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from models import UNet
from utils import EggsPansDataset, get_jaccard_loss_and_iou


def train_unet(epoch=40):

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
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    # Initialize model and transfer to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet(classes=2)
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Keep best IoU and checkpoit
    best_iou = 0.0
    best_checkpoint = copy.deepcopy(model.state_dict())

    # Train epochs
    for epoch_idx in range(epoch):

        print('Epoch: {:2}/{}'.format(epoch_idx + 1, epoch))

        # Train phase metrics
        running_loss = 0.0
        running_iou = 0.0
        running_samples = 0
        model.train()

        # Train epoch
        pbar = tqdm(train_dataloader, desc='Loss: {}, IoU: {}'.format(0, 0))
        for imgs, masks in pbar:

            # Convert to device
            imgs = imgs.to(device)
            masks = masks.to(device)

            # Zero gradients
            optim.zero_grad()

            # Forward through net, and get the loss
            pred_masks = model(imgs)
            loss, iou = get_jaccard_loss_and_iou(masks, pred_masks)

            # Compute gradients and compute them
            loss.backward()
            optim.step()

            # Update metrics
            running_loss += loss.item()
            running_iou += iou.item()
            running_samples += 1
            pbar.set_description('Loss: {}, IoU: {}'.format(running_loss / running_samples,
                                                            running_iou / running_samples))












if __name__ == '__main__':
    # Set random seed
    np.random.seed(42)
    train_unet()
