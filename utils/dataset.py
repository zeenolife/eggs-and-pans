import os
import numpy as np
import cv2

from torch.utils.data import DataLoader, Dataset


class EggsPansDataset(Dataset):

    def __init__(self, dataset_dir):

        # Get image and mask pair names
        image_names = os.listdir(os.path.join(dataset_dir, 'images'))
        mask_names = os.listdir(os.path.join(dataset_dir, 'masks'))
        self.dataset_dir = dataset_dir
        self.names = [name for name in image_names if name.endswith(('.jpg', '.JPG', '.png'))]

        # Make sure that they are equal
        assert set(image_names) == set(mask_names), 'Image names do not match mask names'

        # Get class values
        self.classes_dict = {'egg': 128,
                             'pan': 255,
                             }

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):

        # Read the image, convert to rgb, normalize
        img = cv2.imread(os.path.join(self.dataset_dir, 'images', self.names[index]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = img.astype(np.float32)

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

        # Merge them together
        mask = np.stack([egg, pan], axis=-1)

        # Convert from HWC to CHW
        img = img.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        return img, mask


if __name__ == '__main__':

    dataset = EggsPansDataset('../dataset/train')
    print(len(dataset))
    for idx in range(len(dataset)):

        pair = dataset[idx]
