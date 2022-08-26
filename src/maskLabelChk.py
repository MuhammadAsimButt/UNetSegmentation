from src import config
from imutils import paths
import cv2
import numpy as np
import torch

# get list of mask paths
maskPaths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))

#for idx in range(len(maskPaths)):
for idx in range(5):
    #mask = cv2.imread(maskPaths[idx], cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(maskPaths[idx])
    print("\n\ndata type  of the mask  =", type(mask))
    print("Shape of the mask array on reading from Hard Drive =", mask.shape)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    print("Shape of the mask array after reversing order of channels =", mask.shape)

    mask = mask[:, :, 2]
    print("Shape of the mask array after taking only one channel =", mask.shape)
    print("Unique elements present in mask array =", np.unique(mask))
    # Convert the numpy.ndarray to tensor
    mask_t = torch.from_numpy(mask)
    print("Shape of the mask tensor =", mask_t.size())
    print("Unique elements present in mask tensor =", torch.unique(mask_t))
