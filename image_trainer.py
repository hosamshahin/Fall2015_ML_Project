import os
import cv2
import numpy as np
from feature_extractor import FeatureExtractor
import fnmatch
from scipy.cluster.vq import kmeans, vq
from q_learning import QLearn

numWords = 10
featureExtractor = FeatureExtractor(type='orb', parent= None)

trainImageDir = os.path.join(os.curdir,"original_images")

gestureIDlist = ['A','B','C','D','E','F']
trainImages = []
for i in range(len(gestureIDlist)):
	trainImages.append([])

desList = []
for filename in os.listdir(trainImageDir):
	firstchar = filename[0]
	for i,gestureID in enumerate(gestureIDlist):
		if firstchar == gestureID:
			trainImages[i].append(filename)
			im = cv2.imread(os.path.join(trainImageDir, filename))
			imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
			kp,des = featureExtractor.get_keypoints_and_descriptors(imgray)
			desList.append(des)
			

descriptors = desList[0]
for des in desList:
	descriptors = np.vstack((descriptors, des))

if descriptors.dtype != "float32":
	descriptors = np.float32(descriptors)
voc,variance = kmeans(descriptors, numWords, 30)
sumNum = []
for i in range(len(gestureIDlist)):
	if i == 0:
		sumNum.append(0)
	else:
		val = sumNum[i-1] + len(trainImages[i-1]);
		sumNum.append(val)

trainData = np.zeros((sumNum[-1]+len(trainImages[-1]), numWords), "float32")
trainLabels = np.zeros((sumNum[-1]+len(trainImages[-1])), "uint32")

for gestureID in range(len(gestureIDlist)):
	for numFrame in range(len(trainImages[gestureID])):
		words, distance = vq(desList[sumNum[gestureID]+numFrame], voc)
		for w in words:
			trainData[sumNum[gestureID]+numFrame][w] += 1
		trainLabels[sumNum[gestureID]+numFrame] = gestureID


