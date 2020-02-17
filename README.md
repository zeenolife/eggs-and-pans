# eggs-and-pans
Segmentation model for eggs and pans

## Description
1. Architecture:
    - This is a simple UNet with Resnet50 backbone, with two heads for egg and pan class.
2. Augmentations:
    - Albumentations are applied dynamically during training such as Flip90, Horizontal Flip, Vertical Flip, 
    RandomContrastBrightness and etc.
3. Losses:
    - Special coefficient for two classes are applied, such that it balances out pixel-wise class imbalance
    - Combination of two losses are applied, which are Jaccard and BinaryCrossEntropy Losses
4. Metrics:
    - Validation metric is IoU(Intersection over Union)
5. Training:
    - ReduceLROnPlateau is applied
    - Trained on 100 epochs, with 8 image batch size
    
## Installation
```bash
pip install -r requirements.txt
```

## Training
- Extract dataset into ```eggs-and-pans/dataset```
- Create ```checkpoints``` directory
- The data should have the following structure:
```bash
eggs-and-pans
  ├── checkpoints
  └── dataset
        └── train
        |     ├── images
        |     └── masks
        └── test
              └── images
```
- Run train script 
```
python train.py
```

## Testing
- Create ```test_vis``` directory
- Run test script 
```
python test.py
```

## Further Improvements
- Progressive training, steadily increasing image size from 256 till 1024
- Combination of losses: Jaccard + Dice + CrossEntropy + Focal losses 
- Cross validation training
- Augmented testing (flips, multi-scale)
- Better backbone
- Add config file
- Filter out very small objects
- Weight warm up
- Cosine learning rate change