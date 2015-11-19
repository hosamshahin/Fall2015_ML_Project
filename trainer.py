import sys
import os
import cv2
import numpy as np
from scipy.cluster.vq import kmeans, vq
from sklearn import svm

class Trainer(object):
	def __init__(self, numGestures, numFramesPerGesture, numWords, handTracker, featureExtractor):
		self.numGestures = numGestures
		self.numFramesPerGesture = numFramesPerGesture
		self.numWords = numWords
		self.handTracker = handTracker
		self.featureExtractor = featureExtractor
		self.desList = []
		self.trainData = np.zeros((numGestures*numFramesPerGesture, numWords), "float32")
		self.trainLabels = np.zeros((numGestures*numFramesPerGesture), "uint32")
		self.voc = None
		self.classifier = None

	def extract_descriptors_from_video(self, vc, windowName, handWindowName, numFramesForMedian, minDescriptorsPerFrame, imWidth, imHeight, imsave=False):
		if imsave:
			medianImDir = os.path.join(os.curdir,"medianImages")
			if not os.path.exists(medianImDir):
				os.makedirs(medianImDir)
		gestureFrames = np.zeros((numFramesForMedian,imHeight,imWidth,3), "uint8")
		gestureID = 1
		frameNum = 0
		gestureFrameID = 0
		captureFlag = False
		#while(vc.isOpened()):
		while(vc.isOpened()):
			ret,im = vc.read()
			im = cv2.flip(im, 1)
			imhsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
			imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
			binaryIm = self.handTracker.get_binary_image(imhsv)
			cnt,hull,centroid,defects = self.handTracker.get_contour(binaryIm)
			imCopy = 1*im
			if cnt is not None:
				cropImage = self.handTracker.get_cropped_image(im, cnt)
				cropImageGray = self.handTracker.get_cropped_image(imgray, cnt)
				kp,des = self.featureExtractor.get_keypoints_and_descriptors(cropImageGray)
				if des is not None and des.shape[0] >= 0:
					self.featureExtractor.draw_keypoints(cropImage, kp)
				if captureFlag:
					#gestureFrames[gestureFrameID] = im
					#gestureFrameID += 1
					#if gestureFrameID < numFramesForMedian:
					#	continue
					#gestureFrameID = 0
					#medianIm = np.uint8(np.median(gestureFrames, axis=0))
					#if imsave:
					#	filename = "image_{0}_{1}.jpg".format(gestureID, frameNum)
					#	cv2.imwrite(os.path.join(medianImDir,filename), medianIm)
					#imhsv = cv2.cvtColor(medianIm, cv2.COLOR_BGR2HSV)
					#cnt,hull,centroid,defects = self.handTracker.get_contour(imhsv)
					#if cnt is not None:
					#if gestureID > 0 and frameNum < self.numFramesPerGesture:
					if des is not None and des.shape[0] >= minDescriptorsPerFrame:
						self.desList.append(des)
						#self.featureExtractor.draw_keypoints(cropImage, kp)
						self.handTracker.draw_on_image(imCopy, cnt=False, hullColor=(0,255,0))
						frameNum += 1
					else:
						self.handTracker.draw_on_image(imCopy, cnt=False, hullColor=(0,0,255))
					if frameNum >= self.numFramesPerGesture:
						if gestureID >= self.numGestures:
							break
						else:
							captureFlag = False
							gestureID += 1
							frameNum = 0
				else:
					self.handTracker.draw_on_image(imCopy, cnt=False)
				cv2.imshow(handWindowName, cropImage)
			cv2.imshow(windowName,imCopy)
			k = cv2.waitKey(1)
			if not captureFlag:
				print "Press <space> for new gesture <{0}>!".format(gestureID)
				if k == 32:
					captureFlag = True
				elif k == 27:
					sys.exit(0)
			else:
				if k == 27:
					sys.exit(0)
		descriptors = self.desList[0]
		for des in self.desList:
			descriptors = np.vstack((descriptors, des))
		return descriptors

	def kmeans(self, descriptors, numIters):
		if descriptors.dtype != "float32":
			descriptors = np.float32(descriptors)
		self.voc,variance = kmeans(descriptors, self.numWords, numIters)
		return variance

	def bow(self):
		for gestureID in range(self.numGestures):
			for numFrame in range(self.numFramesPerGesture):
				words, distance = vq(self.desList[gestureID*self.numFramesPerGesture+numFrame], self.voc)
				for w in words:
					self.trainData[gestureID*self.numFramesPerGesture+numFrame][w] += 1
				self.trainLabels[gestureID*self.numFramesPerGesture+numFrame] = gestureID
		print "Training data extracted!"

	def linear_svm(self):
		lin_clf = svm.LinearSVC()
		valScore = self.leave_one_out_validate(lin_clf)
		lin_clf.fit(self.trainData, self.trainLabels)
		self.classifier = lin_clf
		return valScore

	def leave_one_out_validate(self, clf):
		fullTrainData = self.trainData
		fullTrainLabels = self.trainLabels
		accuracy = np.zeros(fullTrainLabels.shape[0])
		for i in range(fullTrainLabels.shape[0]):
			testData = fullTrainData[i]
			testLabels = fullTrainLabels[i]
			trainData = np.append(fullTrainData[:i], fullTrainData[i+1:], axis=0)
			trainLabels = np.append(fullTrainLabels[:i], fullTrainLabels[i+1:])
			#clf = svm.LinearSVC()
			clf.fit(trainData, trainLabels)
			prediction = clf.predict(testData.reshape(1,-1))
			#score = clf.decision_function(testData.reshape(1,-1))
			if prediction != testLabels:
				accuracy[i] = 0
			else:
				accuracy[i] = 1
		return np.mean(accuracy)

	def predict(self, testData):
		prediction = self.classifier.predict(testData.reshape(1,-1))
		score = self.classifier.decision_function(testData.reshape(1,-1))
		return prediction[0], score[0]
