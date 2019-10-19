# Randomize the data and save it into different numpy array

import os
from tqdm import tqdm
import cv2
import numpy as np
import random

folderTrain = "asl-alphabet/asl_alphabet_train"
folderTest = "asl-alphabet/asl_alphabet_test"


xTrain = []
yTrain = []
xTest = []
yTest = []
imageLabel =[]

imageName =[]
index =0
for folderName in os.listdir(folderTrain):
    if not folderName.startswith("."):
        for Name in tqdm(os.listdir(folderTrain + '/' + folderName)):
            if not Name.startswith('.') and index <=1500:
                index +=1
                imageName.append(folderTrain+'/'+folderName+'/'+Name)
        index = 0

random.shuffle(imageName)
print('------------------------')
print(imageName)
for image in tqdm(imageName):
    img = cv2.imread(image)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100, 100))
    xTrain.append(img)
    try:
        yTrain.append(ord(image[image.find("/",30)+1:image.find("/",32)])-65)
    except:
        if image[image.find("/",30)+1:image.find("/",32)] == "space":
            yTrain.append(26)
        elif image[image.find("/",30)+1:image.find("/",32)] == "nothing":
            yTrain.append(27)
        elif image[image.find("/",30)+1:image.find("/",32)] == "del":
            yTrain.append(28)

xTrain = np.array(xTrain)
yTrain = np.array(yTrain)

# print(xTrain)
# print(yTrain)

np.save("data_greyScale/xtrainRandom100.npy", xTrain)
np.save("data_greyScale/ytrainRandom100.npy", yTrain)



for testName in os.listdir(folderTest):
    if not testName.startswith("."):
        if testName[1] =="_":
            label = ord(testName[0]) - 65
        else:
            if testName == "space_test.jpg":
                label = 26
            elif testName == "nothing_test.jpg":
                label = 27
            elif testName == 'del595.jpg':
                label = 28
        img = cv2.imread(folderTest+'/'+testName)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (100, 100))
        xTest.append(img)
        yTest.append(label)
        print(folderTest+'/'+testName)


xTest = np.array(xTest)
yTest = np.array(yTest)
np.save("data_greyScale/xTestRandom100.npy",xTest)
np.save("data_greyScale/yTestRandom100.npy",yTest)