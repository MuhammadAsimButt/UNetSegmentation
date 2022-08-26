# USAGE
# python predict.py
# import the necessary packages
import numpy

from src import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os

'''
To use our segmentation model for prediction, we will need a function that can take our trained model and test 
images, predict the output segmentation mask and finally, visualize the output predictions.

To this end, we start by defining the prepare_plot function to help us to visualize our model predictions.
'''
def prepare_plot(origImage, origMask, predMask):
	'''
	This function takes as input an image, its ground-truth mask, and the segmentation output predicted by our
	model, that is, origImage, origMask, and predMask and creates a grid with a single row and three columns
	to display them.
	'''
	# initialize our figure
	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))

	# plot the original image, its mask, and the predicted mask
	print(f"[predict.py]: Shape of orignal image = {origImage.shape}")
	ax[0].imshow(origImage)
	print(f"[predict.py]: Shape of orignal mask = {origMask.shape}")
	ax[1].imshow(origMask)
	print(f"[predict.py]: Shape of predMask image = {predMask.shape}")
	ax[2].imshow(predMask)

	# set the titles of the subplots
	ax[0].set_title("Image")
	ax[1].set_title("Original Mask")
	ax[2].set_title("Predicted Mask")

	# set the layout of the figure and display it
	figure.tight_layout()

	figure.show()

'''
Next, we define our make_prediction function, which will take as input the path to a test image and our trained 
segmentation model and plot the predicted output.
'''
def make_predictions(model, imagePath):
	# Since we are only using our trained model for prediction, we start by setting our model to eval mode and
	# switching off PyTorch gradient computation on following two statements, respectively.

	# set model to evaluation mode
	model.eval()

	# turn off gradient tracking
	with torch.no_grad():
		'''
		we load the test image (i.e., image) from imagePath using OpenCV, convert it to RGB format, and 
		normalize its pixel values from the standard [0-255] to the range [0, 1], which our model is trained 
		to process (3rd Line)...........myQuestion isn't this done automatically when array converted to tensor??????
		'''
		# load the image from disk, swap its color channels, cast it to float data type, and scale its pixel values
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.astype("float32") / 255.0 #.myQuestion isn't this done automatically when array converted to tensor??????

		'''
		The image is then resized to the standard image dimension that our model can accept. Since we will have to 
		modify and process the image variable before passing it through the model, we make an additional copy of it 
		and store it in the orig variable, which we will use later.
		'''
		# resize the image and make a copy of it for visualization
		image = cv2.resize(image, (config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT))
		orig = image.copy()

		'''
		we get the path to the ground-truth mask for our test image and load the mask. Note that we resize the mask to 
		the same dimensions as the input image.
		'''
		# find the filename and generate the path to ground truth mask
		root_ext = os.path.splitext(imagePath) # get a tuple containing root path and file extension
		root = root_ext[0] # get root path
		filename = (root.split(os.path.sep)[-1])+'.png'# get file name and add required extension
		print(f"[predict.py]: filename = {filename}")
		groundTruthPath = os.path.join(config.MASK_DATASET_PATH, filename)
		print(f"[predict.py]: groundTruthPath = {groundTruthPath}")
		################################################### color???????????
		# load the ground-truth segmentation mask in grayscale mode and resize it. The flag cv2.IMREAD_GRAYSCALE (we
		# can pass integer value 0 for this flag) specifies to load an image in grayscale mode.
		gtMask = cv2.imread(groundTruthPath, 0)
		gtMask = cv2.resize(gtMask, (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_HEIGHT))
		print(f"[predict.py]: gtMask shape after reading from the disk = {gtMask.shape}")

		'''
		Now we process our image to a format that our model can process. Note that currently, our image has the shape 
		[height, width, Channels]. However, our segmentation model accepts four-dimensional inputs of the format 
		[batch_dimension, channel_dimension, height, width].
		'''
		'''
		we transpose the image to convert it to channel-first format, that is, [ch, h, w], and on next Line, we add an 
		extra dimension using the expand_dims function of numpy to convert our image into a four-dimensional array 
		(i.e., [1, 3, h,R]). Note that the first dimension here represents the batch dimension equal to one since we are
		processing one test image at a time. We then convert our image to a PyTorch tensor with the help of the 
		torch.from_numpy() function and move it to the device our model is on with the help of 3rd Line.
		'''
		# make the channel axis to be the leading one, add a batch dimension, create a PyTorch tensor,
		# and flash it to the current device
		image = np.transpose(image, (2, 0, 1))
		print(f"input Image Dimensions after transpose = {image.shape}, max element is = {np.amax(image)}")
		image = np.expand_dims(image, 0)
		print(f"input Image Dimensions after dimension expansion = {image.shape}")
		image = torch.from_numpy(image).to(config.DEVICE)
		print(f"input Image Dimensions after converting to tensor {image.size()}")

		'''
		Finally, on following three Lines, we process our test image by passing it through our model and saving the 
		output prediction as predMask. We then apply the sigmoid activation to get our predictions in the range [0, 1]. 
		As discussed earlier, the segmentation task is a classification problem where we have to classify the pixels 
		in one of the two discrete classes. 
		'''
		# make the prediction, pass the results through the sigmoid function, and convert the result to a NumPy array
		#predMask = model(image).squeeze()
		predMask = model(image)
		print(f"predMask Dimensions={predMask.size()}")#batch,classes, W,H(classes num of arrays each for single class)
		print(f"mask for  class  0 = {predMask[0,3,:,:]}")
		predMask = predMask.squeeze()
		print(f"predMask Dimensions after squeez = {predMask.size()}")
		predMask = torch.sigmoid(predMask)
		print(f"predMask Dimensions after sigmoid {predMask.size()}")
		predMask = predMask.cpu().numpy()
		print(f"predMask Dimensions cpu().numpy() {type(predMask)}")

		'''
		Since sigmoid outputs continuous values in the range [0, 1], we use our config.THRESHOLD to binarize our output 
		and assign the pixels, values equal to 0 or 1. This implies that anything greater than the threshold will be 
		assigned the value 1, and others will be assigned 0.
		
		Since the thresholded output (i.e., (predMask > config.THRESHOLD)), now comprises of values 0 or 1, 
		multiplying it with 255 makes the final pixel values in our predMask either 0 (i.e., pixel value for black color) 
		or 255 (i.e., pixel value for white color). As discussed earlier, the white pixels will correspond to the 
		region where our model has detected salt deposits, and the black pixels correspond to regions where salt is 
		not present.
		'''
		# filter out the weak predictions and convert them to integers
		predMask = (predMask > config.THRESHOLD) * 255
		predMask = predMask.astype(np.uint8)

		'''
		We plot our original image (i.e., orig), ground-truth mask (i.e., gtMask), and our predicted output 
		(i.e., predMask) with the help of our prepare_plot function. 
		This completes the definition of our make_prediction function.
		'''
		# prepare a plot for visualization
		prepare_plot(orig, gtMask, predMask)


# We are ready to see our model in action now.

'''
we open the folder where our test image paths are stored and randomly grab 10 image paths
'''
# load the image paths in our testing file and randomly select 10 image paths
print("[INFO] loading up test image paths...")
imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
imagePaths = np.random.choice(imagePaths, size=10)

'''
following Line loads the trained weights of our U-Net from the saved checkpoint at config.MODEL_PATH.
'''
# load our model from disk and flash it to the current device
print("[INFO] load up model...")
unet = torch.load(config.MODEL_PATH).to(config.DEVICE)

'''
We finally iterate over our randomly chosen test imagePaths and predict the outputs with the help of our 
make_prediction function on Lines 90-92.
'''
# iterate over the randomly selected test image paths
for path in imagePaths:
	# make predictions and visualize the results
	make_predictions(unet, path)

############### Lets test the module #####################
if __name__ == '__main__':
	print("AOA")