#!/home/asim_butt/.conda/envs/UNetSegvenv
#For Python, this is a simple comment, but for the operating system, this line indicates what program must be used to run the file.
#  the #! character combination, which is commonly called hash bang or shebang, and continues with the path to the interpreter.


import argparse
from torch import nn

# Import usert defined modues
from src import config
from myLosses import FocalLoss, mIoULoss
'''
from model import UNet
from dataset import segDataset
'''

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./Semantic segmentation dataset', help='path to your dataset')
    parser.add_argument('--num_epochs', type=int, default=100, help='dnumber of epochs')
    parser.add_argument('--batch', type=int, default=4, help='batch size')
    parser.add_argument('--loss', type=str, default='focalloss', help='focalloss | iouloss | crossentropy')
    return parser.parse_args()

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    args = get_args()
    N_EPOCHS = args.num_epochs
    BACH_SIZE = args.batch
    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
    device = config.DEVICE
    if args.loss == 'focalloss':
        criterion = FocalLoss(gamma=3 / 4).to(device)
    elif args.loss == 'iouloss':
        criterion = mIoULoss(n_classes=4).to(device)
    elif args.loss == 'crossentropy':
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        print('Loss function not found!')

    # Training Our Segmentation Model
    from src.dataset import SegmentationDataset
    from src.model import UNet
    from src import config

    from torch.nn import CrossEntropyLoss  # generally used for segmentation of multiple categories.

    from torch.optim import Adam
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split
    from imutils import paths
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import torch
    import time
    import os
    import torch
    import math
    import numpy as np

    ############ INPUT
    #Use the PyTorch random to generate the input features(X) and labels(y) values.
    #x = torch.randn(3,config.INPUT_IMAGE_WIDTH,config.INPUT_IMAGE_HEIGHT)
    #x = torch.zeros(3, config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT)
    x = torch.ones(3, config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT)
    x=x*4

    #y = torch.randint(4, (1,config.INPUT_IMAGE_WIDTH,config.INPUT_IMAGE_HEIGHT), dtype=torch.long)
    #y = torch.zeros(4, (1, config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT), dtype=torch.long)
    y = torch.ones((1, config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT), dtype=torch.long)
    #y = torch.zeros((1,config.INPUT_IMAGE_WIDTH,config.INPUT_IMAGE_HEIGHT), dtype=torch.long)
    print(f"Shape of y ={y.size()}")
    #y[0] = y[1] =y[2] = y1
    y = y*4
    print(f"x = {x.size()}")
    print(f"y = {y.size()}")
    x=x.numpy()
    #y = y.numpy()

    # include batch dim in x #x = np.transpose(x, (2, 0, 1))
    #print(f"input Image Dimensions after transpose = {x.shape}, max element is = {np.amax(x)}")
    x = np.expand_dims(x, 0)
    print(f"input Image Dimensions after dimension expansion = {x.shape}")
    x = torch.from_numpy(x).to(config.DEVICE)
    print(f"input Image Dimensions after converting to tensor {x.size()}")
    #2.    Categorical  Cross  Entropy  using  Pytorch  PyTorch  categorical  Cross - Entropy  module, the
    # softmax    activation    function    has    already    been    applied    to    the    formula. Therefore  we
    # will  not use  an  activation  function as we  did in the  earlier example.
    # We are   still  using  the  PyTorch  random  to  generate  the  input  features(X) and labels(y)  values.
    # Since this is a multi -  class problem, the input features have five classes(class_0, class_1, class_2, class_3, class_4)


    unet = UNet().to(config.DEVICE)
    lossFunc = CrossEntropyLoss(reduction='none')# reduction='none' to have shape of output same as input image
    opt = Adam(unet.parameters(), lr=config.INIT_LR)

    unet.train()
    print("[train.py, ]:Shape of tensor that was assigned to DEVICE, X=", x.size())
    print("[train.py, ]:SShape of tensor y that was assigned to DEVICE", y.size())
    (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))  # x:input image sample, y:ground-truth label
    #print("[train.py, ]:Shape of tensor that was returned by DEVICE, X=", x.size())
    #print("[train.py, ]:SShape of tensor y that was returned by DEVICE", y.size())
    pred = unet(x)

    n, c, h, w = pred.size()  # batch size, no of classes, d1, d2
    # nt, ct, ht, wt = y.size()
    nt, ht, wt = y.size()
    print("[main.py]: Pred shape:BS, cl, h, w = ", n, c, h, w)
    # print("[train.py, Training Loop ]: Label: BS, cl, ht, wt", nt, ct, ht, wt)
    print("[main.py ]: Label shape: BS, cl, ht, wt = ", nt, ht, wt)
    #print("[main.py ]: Pred type", type(pred))
    print(f"[main.py ]: Label/target  : {y}")
    print(f"Max value of predicted output = {torch.max(pred)}, and min = {torch.min(pred)}")
    loss = lossFunc(pred, y)
    print(f"loss shape ={loss.size()}")
    print(f"loss contents = {loss}")
    opt.zero_grad()

    # RuntimeError: grad can be implicitly created only for scalar outputs (SOL) use .sum()
    #loss.backward()
    loss.sum().backward()
    opt.step()

    predicted_mask_MAX = torch.argmax(pred,dim =1)
    print(f"predicted_mask = {predicted_mask_MAX}")
    print(f"Max value of pred = {torch.max(pred)}, and min = {torch.min(pred)}")

    predicted_mask_MIN = torch.argmin(pred,dim =1)
    print(f"predicted_map = {predicted_mask_MIN}")
    print(f"Max value of loss = {torch.max(loss)}, and min = {torch.min(loss)}")