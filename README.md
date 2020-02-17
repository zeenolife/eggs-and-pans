# eggs-and-pans
Segmentation model for eggs and pans



### Further Improvements
- Progressive training, steadily increasing image size from 256 till 1024
- Combination of losses: Jaccard + Dice + CrossEntropy + Focal losses 
- Cross validation training
- Augmented testing (flips, multi-scale)
- Better backbone
- Add config file
- Filter out very small objects
- Weight warm up
- Cosine learning rate change