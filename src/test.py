# Python program to explain os.path.splitext() method

# importing os module
import os
import numpy as np
# path
path = '/home/asim_butt/DATASETS/SICAPv2/masks/'

fileName = '16B0001851_Block_Region_1_0_1_xini_7827_yini_59786.png'
path = '/home/asim_butt/DATASETS/SICAPv2/masks/'+fileName

# Split the path in root and ext pair
root_ext = os.path.splitext(path)

# print root and ext of the specified path
print("root part of '% s':" % path, root_ext[0])
print("ext part of '% s':" % path, root_ext[1], "\n")

# Split the path in root and ext pair
root_ext = os.path.splitext(path)

# print root and ext of the specified path
print("root part of '% s':" % path, root_ext[0])
print("ext part of '% s':" % path, root_ext[1])

# Python code to read image
import cv2

# To read image from disk, we use  cv2.imread function, in below method,
img = cv2.imread(path, cv2.IMREAD_COLOR)
print(f"type of image {type(img)}")
print(f"Unique contents of img ={np.unique(img)}")

#new_img_array = img[img==3] = 1

new_img_array = np.array([[3, 3, 3], [4, 4, 4],[5,5,5]])

new_img_array = np.where(new_img_array == 3,1, new_img_array)
new_img_array = np.where(new_img_array == 4,2, new_img_array)
new_img_array = np.where(new_img_array == 5,3, new_img_array)
print(f"new_img_array={new_img_array}")
print(f"Unique contents of new_img_array ={np.unique(new_img_array)}")
# Creating GUI window to display an image on screen, first Parameter is windows title (should be in string format)
# Second Parameter is image array
#cv2.imshow("image", img)
#cv2.waitKey(0)

# It is for removing/deleting created GUI window from screen and memory
#cv2.destroyAllWindows()