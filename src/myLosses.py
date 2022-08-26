import torch
import torch.nn as nn
#from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.autograd import Variable
from src import config
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=2):
        super(mIoULoss, self).__init__()
        self.classes = n_classes

    def to_one_hot(self, tensor):
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, self.classes, h, w).to(tensor.device).scatter_(1, tensor.view(n, 1, h, w), 1)
        return one_hot

    def forward(self, inputs, target):
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1)

        # Numerator Product
        target_oneHot = self.to_one_hot(target)
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2)

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2)

        loss = inter / union

        ## Return average loss over classes and batch
        return 1 - loss.mean()
def genMask(inputArray, mappingDict):
    mask = np.select([(np.logical_and(inputArray >= 0 , inputArray <= 10)),
                      (np.logical_and(inputArray >= 11 , inputArray <= 20)),
                      (np.logical_and(inputArray >= 21 , inputArray <= 30)),
                      (np.logical_and(inputArray >= 31 , inputArray <= 40))], [0, 1, 2, 3], inputArray)
    unique_O, counts_O = np.unique(inputArray, return_counts=True)
    print(f"Unique contents of original img ={unique_O} and their counts are {counts_O}, Total entries ={np.sum(counts_O)}")
    unique, counts = np.unique(mask, return_counts=True)
    print(f"Unique contents of original mask ={unique} and their counts are {counts}, Total entries ={np.sum(counts)}")

    # Convert the numpy.ndarray to tensor and then return

    return torch.from_numpy(mask).to(config.DEVICE)

if __name__ == "__main__":
    # My scheme of classes and labels
    #0-10   Label =0
    # 11-20   Label =1
    # 21-30   Label =2
    # 31-40   Label =3
    mapping_Dict = {
        "0": (0,10),
        "1": (11,20),
        "2": (21,30),
        "3": (31,40)
    }
    myImageArrays = []
    myMaskArrays = []
    batchSize = 10
    for i in range(batchSize):
        input = (torch.tensor([[7, 7, 7, 7, 7, 7, 7],
                               [11, 11, 11, 11, 11, 11, 11],
                               [25, 25, 25, 25, 25, 25, 25],
                               [31, 31, 31, 31, 31, 31, 31],
                               [4, 4, 4, 4, 4, 4, 4],
                               [0, 0, 0, 0, 0, 0, 0],
                               [30, 30, 30, 30, 30, 30, 30]], dtype=torch.float32, device=config.DEVICE, requires_grad=True)
                 )+i
        myImageArrays.append(input)
        print(f"input_1 = {input}")
        mask = genMask(input.cpu().detach().numpy(),mapping_Dict)
        print(f"mask_1 -{mask}")
        myMaskArrays.append(mask)

    lossFunc = nn.CrossEntropyLoss(reduction='none')  # reduction='none' to have shape of output same as input image

    totalTrainLoss=0
    for i in range(len(myMaskArrays)):
        pred = myImageArrays[i]
        label = myMaskArrays[i]
        loss = lossFunc(pred, label)
        print(f"loss shape ={loss.size()}")
        print(f" loss = {loss} for iteration No = {i}")
        totalTrainLoss += loss
        print(f" total_Loss ={totalTrainLoss} at iteration No = {i}")

    '''
    #The torch.linspace function allows creating a new Tensor where each step is evenly spaced between a start and end value.
    input_2 = torch.linspace(1., 9., steps=12)
    print(f"input_2 = {input_2}")
    #torch.arange() simply creates the values with the correct number of steps between a start and end point.
    input_3 = torch.arange(12)
    print(f"input_3 = {input_3}")

    # range
    max = 8
    min = 4
    # create tensor with random values in range (min, max)
    input_4 = (max - min) * torch.rand((2, 5)) + min
    # print tensor
    print(f"input_4 = {input_4}")

    input_5 = torch.rand((7, 7), out=None, dtype=torch.float32, layout=torch.strided, device=config.DEVICE, requires_grad=True)
    print(f"input_5 = {input_5}")
    '''
