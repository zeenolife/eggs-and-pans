import os
import albumentations as albu
import numpy as np
import cv2

from torch.utils.data import Dataset


class EggsPansDataset(Dataset):

    def __init__(self, dataset_dir, names, mode='train', image_size=512):

        # Get image and mask pair names
        self.dataset_dir = dataset_dir
        self.names = [name for name in names if name.endswith(('.jpg', '.JPG', '.png'))]

        # Get class values
        self.classes_dict = {'egg': 128,
                             'pan': 255,
                             }

        # Misc info
        self.image_size = image_size
        self.mode = mode
        assert mode in ('train', 'val', 'test'), 'Mode is unknown: {}'.format(mode)

        # Initialize augmentations
        self.augs = albu.Compose([
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.RandomRotate90(p=0.5),
            albu.CLAHE(p=0.5),
            albu.RandomBrightnessContrast(p=0.5),
            albu.RandomGamma(p=0.5)
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):

        # Read the image
        img = cv2.imread(os.path.join(self.dataset_dir, 'images', self.names[index]))

        # Read the mask, convert to two channel image
        mask = cv2.imread(os.path.join(self.dataset_dir, 'masks', self.names[index]))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Make sure that pair shapes match
        assert img.shape[:2] == mask.shape[:2], 'Image shape {} does not match mask shape {}'.format(img.shape,
                                                                                                     mask.shape)

        # Zero pad images to square
        max_side = max(img.shape[:2])
        vertical_border = (max_side - img.shape[0]) // 2
        horizontal_border = (max_side - img.shape[1]) // 2

        img = cv2.copyMakeBorder(img, vertical_border, vertical_border, horizontal_border, horizontal_border,
                                 cv2.BORDER_CONSTANT, value=0)
        mask = cv2.copyMakeBorder(mask, vertical_border, vertical_border, horizontal_border, horizontal_border,
                                  cv2.BORDER_CONSTANT, value=0)

        # Get egg and pan masks separately
        egg = ((self.classes_dict['egg'] - 64 <= mask) & (mask <= self.classes_dict['egg'] + 64)).astype(np.float32)
        pan = (self.classes_dict['pan'] - 64 <= mask).astype(np.float32)

        # Resize image to default
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        egg = cv2.resize(egg, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        pan = cv2.resize(pan, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        # Add axis
        egg = np.expand_dims(egg, axis=-1)
        pan = np.expand_dims(pan, axis=-1)

        # Apply augmentations
        if self.mode == 'train':
            masks = [egg, pan]
            augmented = self.augs(image=img, masks=masks)
            img, masks = augmented['image'], augmented['masks']
            egg, pan = masks

        # Convert to RGB, normalize, and change from HWC to CHW
        img = self._normalize_img(img)
        img = img.transpose(2, 0, 1)
        egg = egg.transpose(2, 0, 1)
        pan = pan.transpose(2, 0, 1)

        return img, egg, pan

    def _normalize_img(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = img.astype(np.float32)

        return img


if __name__ == '__main__':

    dataset = EggsPansDataset('../dataset/train')
    print(len(dataset))
    for idx in range(len(dataset)):

        pair = dataset[idx]
