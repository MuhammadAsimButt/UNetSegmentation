'''
The config.py file in the src folder stores our code’s parameters, initial settings, and configurations.
'''
# import the necessary packages
import torch
from torchvision import transforms
import os

#NUM_WORKERS=os.cpu_count()
NUM_WORKERS=0
print("num of workers = ", NUM_WORKERS)

#Set the default torch.Tensor type to floating point tensor type t
#torch.set_default_tensor_type(torch.float32)

# base path of the dataset

DATASET_PATH = os.path.join("/home/asimbutt/DATASETS/", "SICAPv2")
print(f"DATASET_PATH = {DATASET_PATH}")
# define the path to the images and masks dataset
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "images")
#IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "Working_images")
print(f"IMAGE_DATASET_PATH = {IMAGE_DATASET_PATH}")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "masks")
#MASK_DATASET_PATH = os.path.join(DATASET_PATH, "masks_Transformed")
print(f"MASK_DATASET_PATH = {MASK_DATASET_PATH}")

# For class weigh in dataset calculation
TRAIN_CSV_PATH = os.path.join(DATASET_PATH, 'partition', 'Test', 'Train.xlsx')
TEST_CSV_PATH = os.path.join(DATASET_PATH, 'partition', 'Test', 'Test.xlsx')

TRAINCRIBFRIFORM_CSV_PATH = os.path.join(DATASET_PATH, 'partition','Test', 'TrainCribfriform.xlsx')
TESTCRIBFRIFORM_CSV_PATH = os.path.join(DATASET_PATH, 'partition', 'Test', 'TestCribfriform.xlsx')


# define the Dataset split
# the fraction of the dataset that we will keep aside for the test set.
TEST_SPLIT = 0.15 # later should be removed when splitDS() is implemented in dataset class
TRAIN_SIZE = 0.85 # between train and remaining
TEST_SIZE = 0.5 # between test and validation sets

# For DEBUGGING use small portion of dataset
PERCENT_DS = 0.95


# determine the device to be used for training and evaluation
# we define the DEVICE parameter, which determines based on availability, whether we will be using a GPU or CPU for
# training our segmentation model.

# In this case, we are using a CUDA-enabled GPU device
#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Device = {DEVICE}")
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes, and number of levels in the U-Net model
NUM_CHANNELS = 3
NUM_LEVELS = 1#3 # UNet levels asim , check

# initialize learning rate, number of epochs to train for, and the batch size
INIT_LR = 0.001
NUM_EPOCHS = 1
BATCH_SIZE = 28

CONV_KERNEL_SIZE = 3
POOL_KERNEL_SIZE = 2
# define the input image dimensions
INPUT_IMAGE_WIDTH = 512 #50
INPUT_IMAGE_HEIGHT = 512 #50

# define threshold to filter weak predictions
THRESHOLD = 0.5

# define the path to the base output directory
BASE_OUTPUT = "/home/asimbutt/Projects/UNetSegmentation/output"

# define the path to the output serialized model, model training plot, and testing image paths
MODEL_NAME = "unet_"+str(PERCENT_DS)+"_"+str(BATCH_SIZE)+"_"+str(NUM_EPOCHS)+"_"+str(INIT_LR)+".pth"
MODEL_PATH = os.path.join(BASE_OUTPUT, MODEL_NAME)# Do something to dynamical naming the model.......
PLOT_NAME = "plot_"+str(PERCENT_DS)+"_"+str(BATCH_SIZE)+"_"+str(NUM_EPOCHS)+"_"+str(INIT_LR)+".png"
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, PLOT_NAME])
TEST_NAME = "testPaths_"+str(PERCENT_DS)+"_"+str(BATCH_SIZE)+"_"+str(NUM_EPOCHS)+"_"+str(INIT_LR)+".txt"
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, TEST_NAME])


#https://github.com/cvblab/gleason_grading_cmpb/blob/main/code/main.py
NUM_CLASSES = 4 # benign, 3, 4, 5
CLASSES = ['NC', 'G3', 'G4', 'G5']
#CLASS_LABELS = [0, 3, 4, 5]
CLASS_LABELS = [0, 1, 2, 3]
WEIGHTING = True
'''
##### Use at appropriate place
# Define weights for class imbalance
if weighting:
    class_weights = compute_class_weight('balanced', [0, 1, 2, 3], np.argmax(train.labels, 1))
    class_weights = {0: class_weights[0], 1: class_weights[1], 2: class_weights[2], 3: class_weights[3]}
else:
    class_weights = {0: 1, 1: 1, 2: 1, 3: 1}
'''

# Now, we are ready to set up our data loading pipeline.
# define transformations
TRANSFORMS = transforms.Compose([#transforms.ToPILImage(),
                                 #transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)),
                                 transforms.ToTensor()#,
                                 # normalize images by using the mean and standard deviation of each band in the
                                 # input images. You can calculate these
                                 #transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 #                     std=[0.2023, 0.1994, 0.2010]) # added by me
                                 # THIS CREATS CONFLICT WITH MASK DIMS
                                 ]
                                )
'''
    Our transformations include:

    ToPILImage(): it enables us to convert our input images to PIL image format. Note that this is necessary since we 
    used OpenCV to load images in our custom dataset, but PyTorch expects the input image samples to be in PIL format.
    Resize(): allows us to resize our images to a particular input dimension (i.e., config.INPUT_IMAGE_HEIGHT, 
    config.INPUT_IMAGE_WIDTH) that our model can accept
    ToTensor(): enables us to convert input images to PyTorch tensors and convert the input PIL Image, which is 
        originally in the range from [0, 255], to [0, 1].
'''
############### Lets test the module #####################
'''
if __name__ == '__main__':
    print("IMAGE_DATASET_PATH = ", IMAGE_DATASET_PATH)
    print("MODEL_PATH = ", MODEL_PATH)
    print("TEST_PATHS = ", TEST_PATHS)
'''