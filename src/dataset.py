# import the necessary packages
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np # crem when to be deployed

import src.utils_asim
from src import config
import pandas as pd
import os
from sklearn.model_selection import train_test_split

class SegmentationDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, transforms):
        # store the image and mask filepaths, and augmentation transforms
        self.numClasses = config.NUM_CLASSES
        self.classes = config.CLASSES
        self.classLabels = config.CLASS_LABELS
        self.weighting = config.WEIGHTING
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms
        self.suspiciousTiles = []

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imagePaths)

    def __getitem__(self, idx):
        # grab the image path from the current index
        imagePath = self.imagePaths[idx]

        # load the image from disk, swap its channels from BGR to RGB,
        # By default, OpenCV loads an image in the BGR format, which we convert to the RGB format
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print("Shape of the image array on reading from Hard Drive =", np.asarray(image).shape)

        # and read the corresponding ground-truth segmentation associated mask from disk in grayscale mode
        # Syntax: cv2.imread(path, flag), cv2.IMREAD_GRAYSCALE: It specifies to load an image in grayscale mode.
        # Alternatively, we can pass integer value 0 for this flag.
        #mask = cv2.imread(self.maskPaths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.maskPaths[idx])
        #print("[dataset.py]: Shape of the mask array on reading from Hard Drive =", np.asarray(mask).shape)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        #print("[dataset.py]: Shape of the mask array after swaping its channels from BGR to RGB=", np.asarray(mask).shape)
        ###############################################################################
        elementVals = np.unique(mask[:, :, 0])
        # All Channels of mask contain same data
        #print("[dataset.py]: Unique elements present in mask array ch 0 =", elementVals)
        #print("[dataset.py]: Unique elements present in mask array ch 1 =", np.unique(mask[:, :, 1]))
        #print("[dataset.py]: Unique elements present in mask array ch 2 =", np.unique(mask[:, :, 2]))
        from src import utils_asim
        '''
        This snippet just saves mask filenames whic have only class 0 pixels/elements
        if elementVals.all()==0:
            #utils_asim.dispArray(mask, 1)
            self.suspiciousTiles.append(self.maskPaths[idx])
            myFileName = 'suspiciousMasks.csv'
            myFilePathName = os.path.join(config.BASE_OUTPUT, myFileName)
            src.utils_asim.listToCSV(self.suspiciousTiles, myFilePathName)
        '''
        ##############################################################################
        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            # This is important since we want our image and ground-truth mask to correspond and have the same dimension
            image = self.transforms(image)
            mask = self.transforms(mask)

            #print("Shape of the image array after transform ", np.asarray(image).shape)
            #Shape of the mask array after transform is e.g (3, W, H)
            #print("[dataset.py, __getitem__()]: Shape of the mask array after transform ", np.asarray(mask).shape)

        # return a tuple of the image and its mask
        #print("Shape of img returned by DS through DL = ",image.size() )
        #print("Shape of mask returned by DS through DL = ", mask.size())
        #return torch.tensor(image).float(), torch.tensor(cls_mask, dtype=torch.int64)

        # Set datatypes here
        image = image.float()
        mask = torch.tensor(mask, dtype=torch.int64) # generates warning

        # Masks are 3 ch every ch contain same mask info so only first ch is extracted
        #sameVals = torch.equal(mask[0, :, :], mask[1, :, :])
        #print(f"sameVals {sameVals}")
        mask = mask[0, :, :]
        #print("Shape of the mask array after taking only one channel =", mask.shape)

        #print(f"[dataset.py, __getitem__()]: We are returning image {type(image)} and mask {type(mask)}")
        # We are returning image of shape torch.Size([3, W, H]) and mask torch.Size([W, H])
        #print(f"[dataset.py, __getitem__()]: We are returning image of shape {image.size()} and mask {mask.size()}")
        return (image, mask)

    def splitDS(self): # work on it
        df = pd.read_csv('/kaggle/input/bluebook-for-bulldozers/TrainAndValid.csv', parse_dates=['saledate'],
                         low_memory=False)
        # Let's say we want to split the data in 80:10:10 for train:valid:test dataset
        train_size = config.TRAIN_SIZE

        X = df.drop(columns=['SalePrice']).copy()
        y = df['SalePrice']

        # In the first step we will split the data in training and remaining dataset
        X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.8)

        # Now since we want the valid and test size to be equal (10% each of overall data).
        # we have to define valid_size=0.5 (that is 50% of remaining data)
        test_size = config.TEST_SIZE
        X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

        print(X_train.shape), print(y_train.shape)
        print(X_valid.shape), print(y_valid.shape)
        print(X_test.shape), print(y_test.shape)

        return X_train, X_valid, X_test


############### Lets test the module #####################
'''
if __name__ == '__main__':
	from imutils import paths
	from src import config
	from sklearn.model_selection import train_test_split
	from torchvision import transforms
	import numpy as np

	# load the image and mask filepaths in a sorted list
	imagePaths = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
	# print("imagePaths = ", imagePaths)
	maskPaths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))

	split = train_test_split(imagePaths,
							 maskPaths,
							 test_size=config.TEST_SPLIT,
							 random_state=42
							 )
	
	row = len(split)
	col0 = len(split[0])
	col1 = len(split[1])
	col2 = len(split[2])
	col3 = len(split[3])
	
	print(f'No of lists in this list  = {row}')
	print(f'Length of first list trainImages:{col0}')
	print(f'Length of second list testImages:", {col1}')
	print(f'Length of third list trainMasks:", {col2}')
	print(f'Length of fourth list testMasks:", {col3}')
	
	# unpack the data split
	(trainImages, testImages) = split[:2] # list_name[start:stop:steps], here 0 t0 1 not including 2
	(trainMasks, testMasks) = split[2:]	#2 to end
	#print("trainImages =", trainImages)
	#print("testImages =", testImages)


	# define transformations
	transforms = transforms.Compose([transforms.ToPILImage(),
									 transforms.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)),
									 transforms.ToTensor()
									 ]
									)

	# create the train and test datasets, pass list of paths and transforms
	trainDS = SegmentationDataset(imagePaths = trainImages,
								  maskPaths = trainMasks,
								  transforms = transforms
								  )
	# We can now print the number of samples in trainDS and testDS with the help of the len() method.
	print(f"[INFO] found {len(trainDS)} examples in the training set...")

'''
'''
#show image from tensor
# squeeze image to make it 3D
img = img.squeeze(0) # now image is again 3D
print("Output image size:",img.size())

# convert image to PIL image
img = T.ToPILImage()(img)

# display the image after convolution
img.show()'''