import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from models import UNet


def prepare_image(img, img_size=512):

    # Zero pad images to square
    max_side = max(img.shape[:2])
    vertical_border = (max_side - img.shape[0]) // 2
    horizontal_border = (max_side - img.shape[1]) // 2

    img = cv2.copyMakeBorder(img, vertical_border, vertical_border, horizontal_border, horizontal_border,
                             cv2.BORDER_CONSTANT, value=0)

    # Resize image
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    # Normalize image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img = img.astype(np.float32)

    # Change to CHW
    img = img.transpose(2, 0, 1)

    # Convert to torch tensor
    img = torch.from_numpy(img).unsqueeze(0)

    return img


def postprocess_masks(img, mask_egg, mask_pan):

    max_side = max(img.shape[:2])
    min_side = min(img.shape[:2])
    mask_egg = cv2.resize(mask_egg, (max_side, max_side), cv2.INTER_LINEAR)
    mask_pan = cv2.resize(mask_pan, (max_side, max_side), cv2.INTER_LINEAR)

    vertical_border = (max_side - img.shape[0]) // 2
    horizontal_border = (max_side - img.shape[1]) // 2

    if vertical_border == 0:
        mask_egg = mask_egg[:, horizontal_border:min_side + horizontal_border]
        mask_pan = mask_pan[:, horizontal_border:min_side + horizontal_border]

    else:
        mask_egg = mask_egg[vertical_border:min_side + vertical_border, :]
        mask_pan = mask_pan[vertical_border:min_side + vertical_border, :]

    return mask_egg, mask_pan


def test(weights_path):

    # Get all images in train set
    image_names = os.listdir('dataset/train/images/')
    image_names = [name for name in image_names if name.endswith(('.jpg', '.JPG', '.png'))]

    # Initialize model and transfer to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet()
    model = model.to(device)
    model.eval()

    # Load weights
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # Misc info
    img_size = 512

    # Predict on images
    for image_name in tqdm(image_names):

        # Load image, prepare for inference
        img = cv2.imread(os.path.join('dataset/train/images/', image_name))

        img_torch = prepare_image(img, img_size)

        with torch.no_grad():

            # Get predictions for image
            pred_egg_mask, pred_pan_mask = model(img_torch)

            # Threshold by 0.5
            pred_egg_mask = (torch.sigmoid(pred_egg_mask) >= 0.5).type(pred_egg_mask.dtype)
            pred_pan_mask = (torch.sigmoid(pred_pan_mask) >= 0.5).type(pred_pan_mask.dtype)

            pred_egg_mask, pred_pan_mask = pred_egg_mask.cpu().detach().numpy(), pred_pan_mask.cpu().detach().numpy()

        # Resize masks back to original shape
        pred_egg_mask, pred_pan_mask = pred_egg_mask[0][0] * 256, pred_pan_mask[0][0] * 256
        pred_egg_mask, pred_pan_mask = postprocess_masks(img, pred_egg_mask, pred_pan_mask)

        cv2.imwrite('test_vis/' + image_name[:-4] + '_egg' + image_name[-4:], pred_egg_mask)
        cv2.imwrite('test_vis/' + image_name[:-4] + '_pan' + image_name[-4:], pred_pan_mask)
        cv2.imwrite('test_vis/' + image_name, img)


if __name__ == '__main__':
    test('checkpoints/epoch_96_0.9128.pth')
