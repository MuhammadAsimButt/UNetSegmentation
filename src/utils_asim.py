
import os
import pandas as pd
import cv2
import numpy as np
from numpy import asarray
from numpy import savetxt
from os import listdir
from os.path import isfile, join

########################################################################################################################

# Display a 2D array as figure
def dispArray(array_2D, channel):
    myArray = array_2D[:,:,channel]

def listToCSV(list_obj, fileName, dimenssions = None, headings = None):
    '''

    :param list_obj:
    :param fileName:
    :param dimenssions:
    :param headings:
    :return:
    '''

    mylist = list_obj
    myHeadings = headings # This may be a list of length equal to headings
    #dictionary = {'Headings': myHeadings, 'Data': mylist}
    dictionary = {'Data': mylist}
    # I already 
    dataframe = pd.DataFrame(dictionary)
    dataframe.to_csv(fileName)

def changeLabel(maskSourcePath, maskDestinationPath, changeDict, NoOfMasks=0, maskFileNames=None):
    '''
    This function changes labels of the given mask with new ones

    '''
    if maskFileNames == None:
        # Using path to mask directory get all files withOUT path
        mask_Names = giveListOfFiles(maskSourcePath)
        print(f" No of files in the folder {len(mask_Names)}")
    else:
        mask_Names = maskFileNames

    # List of paths to new transformed images
    transformedMasks = []

    if NoOfMasks == 0:
        NoOfMasks = int(len(mask_Names))
    print(f" NoOfMasks = {NoOfMasks}")

    #dict keys are orignal labels and values are new labels
    for i in range(0,NoOfMasks):
        #print(f" {i}th mask_paths = {os.path.join(maskSourcePath,mask_Names[i])}")
        img = cv2.imread(os.path.join(maskSourcePath,mask_Names[i]), cv2.IMREAD_COLOR)
        #print(f"data type of image {img.dtype}")
        unique_O, counts_O = np.unique(img, return_counts=True)
        print(f"Unique contents of original mask ={unique_O} and their counts are {counts_O}, Total entries ={np.sum(counts_O)}")
        new_img_array = img # Just a view in memory this is same object i.e img is also altered
        new_img_array = np.select([new_img_array == 3, new_img_array == 4, new_img_array == 5], [1, 2, 3], new_img_array)
        unique, counts = np.unique(new_img_array, return_counts=True)
        print(f"Unique contents of new_img_array ={unique} and their counts are {counts}")
        newMaskNamePath = os.path.join(maskDestinationPath, mask_Names[i])
        #print(f"newMaskNamePath ={newMaskNamePath}")
        transformedMasks.append(newMaskNamePath)
        cv2.imwrite(newMaskNamePath, new_img_array)

    return transformedMasks

def giveListOfFiles(dirPath):
    '''
    This function returns a list of file in the directory provided by path
    :return:
    '''

    mypath = dirPath
    # os.listdir() returns everything inside a directory -- including both files and directories.
    # os.path's isfile() can be used to only list files:
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    return onlyfiles

def changeExtensions(fileNames, newExtension):
    filesWithChangedExtension = []
    for i in range(0, len(fileNames)):
        fileName = fileNames[i]

        # Split the path in root and ext pair
        root_ext = os.path.splitext(fileName)
        # print root and ext of the specified path
        #print("root part of '% s':" % fileName, root_ext[0])
        #print("ext part of '% s':" % fileName, root_ext[1], "\n")
        fileName = root_ext[0]
        newFileName = fileName+newExtension
        #print(f"New file name = {newFileName}")
        filesWithChangedExtension.append(newFileName)

    return filesWithChangedExtension
#####NOT USED
def newMaskNames(oldMaskName, newMaskpath):
    '''
    This function takes a list of full path wih old mask name and change the mask name and path to new directory
    :param oldMaskName:
            File name with extension
        newMaskpath:
            path to new folder
    :return: list
    '''
    '''newMaskNamesLst = []
    for i in range(0, len(oldMaskNames)):
        fileNameWithPath = oldMaskNames[i]

        # Split the path in root and ext pair
        root_ext = os.path.splitext(fileNameWithPath)
        # print root and ext of the specified path
        print("root part of '% s':" % fileNameWithPath, root_ext[0])
        print("ext part of '% s':" % fileNameWithPath, root_ext[1], "\n")
        maskName = root_ext[1]
        newMaskNamePath = os.path.join(newMaskpath,maskName)

        newMaskNamesLst.append(newMaskNamePath)

    return newMaskNamesLst
    '''

def saveNdarrayOnDisk(ndArray, path):
    '''

    :param ndArray:
    :param path: full path with .csv file extension
    :return:
    '''
    # save numpy array as csv file

    # define data
    data = ndArray
    # save to csv file
    savetxt(path, data, delimiter=',')


########################################################################################################################

if __name__ == '__main__':
    '''
    mylist = ['abc', 'bcd', 'cde']
    myFilePath = '/home/asimbutt/Projects/UNetSegmentation/output'
    myFileName = 'suspiciousMasks.csv'
    myFilePathName = os.path.join(myFilePath, myFileName)
    #print(f"myFilePathName = {myFilePathName}")
    #listToCSV(mylist, myFilePathName)

    df = pd.read_csv(myFilePathName)
    print(df)
    '''
    ################################################################################
    imgPath_source = '/home/asimbutt/DATASETS/SICAPv2/images'
    maskPath_source = '/home/asimbutt/DATASETS/SICAPv2/masks'
    maskPath_destination = '/home/asimbutt/DATASETS/SICAPv2/masks_Transformed'

    thisdict = {
        "3": 1,
        "4": 2,
        "5": 3
    }
    imgFileNames = giveListOfFiles(imgPath_source)
    maskFileNames = changeExtensions(imgFileNames, ".png")
    changeLabel(maskPath_source, maskPath_destination, thisdict,len(maskFileNames), maskFileNames)