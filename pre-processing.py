import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import os
import cv2

def resizeImage(imagePath,img):

    #resize image using bicubic
    resize_image = img.resize((512,512), Image.BICUBIC)

    resize_image.save(imagePath)

def countFiles(folder):
    fileCount = 0
    uniqueHeight= []
    uniqueWidth = []

    #os.listdir creates a list of the files and folders in a specific folder
    #entry will be the filename or folder name
    for entry in os.listdir(folder):
        filepath = os.path.join(folder,entry)
        #isfile checks if the filepath given is a file
        #path.join concatenates 2 paths
        if os.path.isfile(filepath):
            try:
                #open file as image
                with Image.open(filepath) as img:
                    width,height = img.size # get the dimensions
                    if width != 512 or height != 512:
                        print("gg")
                        resizeImage(filepath,img)

                    uniqueHeight.append(height)
                    uniqueWidth.append(width)

                    uniqueWidth = list(set(uniqueWidth))
                    uniqueHeight = list(set(uniqueHeight))


            except Exception as e:
                print(f"Could not process file '{entry}': {e}")
            fileCount+=1

    print(uniqueWidth)
    print(uniqueHeight)
    return fileCount


#print(countFiles("./archive/Testing/pituitary"))

trainArray = []
trainArrayY = []
testArray = []
testArrayY = []
def saveTrainingSet(folderPath, label):

    #go through each image in a directory
    for image_name in os.listdir(folderPath):
        #get the full image path
        image_path = os.path.join(folderPath,image_name)

        #open the image
        img = Image.open(image_path)
        if img.mode == "RGB":
            img = img.convert('L')

        img = img.resize((512,512))

        #convert to numpy array then append to the image_array
        imgnump = np.array(img)

        trainArray.append(imgnump)
        trainArrayY.append(label)


def saveTestSet(folderPath,label):

    #go through each image in a directory
    for image_name in os.listdir(folderPath):
        #get the full image path
        image_path = os.path.join(folderPath,image_name)

        #open the image
        img = Image.open(image_path)
        if img.mode == "RGB":
            img = img.convert('L')

        img = img.resize((512, 512))

        #convert to numpy array then append to the image_array
        imgnump = np.array(img)

        testArray.append(imgnump)
        testArrayY.append(label)



saveTrainingSet("./archive/Training/glioma",0)
saveTrainingSet("./archive/Training/meningioma",1)
saveTrainingSet("./archive/Training/notumor",2)
saveTrainingSet("./archive/Training/pituitary",3)

saveTestSet("./archive/Testing/glioma",0)
saveTestSet("./archive/Testing/meningioma",1)
saveTestSet("./archive/Testing/notumor",2)
saveTestSet("./archive/Testing/pituitary",3)


print(len(trainArray))
print(len(testArray))
print(len(trainArrayY))
print(len(testArrayY))

saveTrain= np.array(trainArray)
saveTest = np.array(testArray)
saveTrainY= np.array(trainArrayY)
saveTestY = np.array(testArrayY)

#saving as npy
np.save('./traindata/trainX.npy',saveTrain)
np.save('./traindata/trainY.npy',saveTrainY)
np.save('./testdata/testX.npy',saveTest)
np.save('./testdata/testY.npy',saveTestY)






