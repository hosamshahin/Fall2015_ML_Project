import cv2
import numpy as np
from scipy.cluster.vq import kmeans, vq

class Trainer(object):
	def __init__(self, numGestures, numFramesPerGesture, numWords, handTracker, featureExtractor):
		self.numGestures = numGestures
		self.numFramesPerGesture = numFramesPerGesture
		self.numWords = numWords
		self.handTracker = handTracker
		self.featureExtractor = featureExtractor
		self.desList = []
		self.trainData = np.zeros((numGestures*numFramesPerGesture, numWords), "float32")
		self.trainLabels = np.zeros((numGestures*numFramesPerGesture, 1), "uint32")

	def extract_descriptors_from_video(self, vc, windowName):
		gestureID = 0
		frameNum = self.numFramesPerGesture
		#while(vc.isOpened()):
		while(vc.isOpened()):
			ret,im = vc.read()
			im = cv2.flip(im, 1)
			imhsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
			cnt,hull,centroid,defects = self.handTracker.get_contour(imhsv)
			imCopy = 1*im
			if cnt is not None:
				self.handTracker.draw_on_image(imCopy)
				if gestureID > 0 and frameNum < self.numFramesPerGesture:
					cropImage = self.handTracker.get_cropped_image(im, cnt)
					kp,des = self.featureExtractor.get_keypoints_and_descriptors(cropImage)
					self.desList.append(des)
					self.featureExtractor.draw_keypoints(cropImage, kp)
					frameNum += 1
					cv2.imshow(windowName,imCopy)
			cv2.imshow(windowName,imCopy)
			k = cv2.waitKey(1)
			if frameNum >= self.numFramesPerGesture:
				if gestureID >= self.numGestures:
					print "Training data extracted!"
					break
				else:
					print "Press <space> for new gesture <{0}>!".format(gestureID)
					if k == 32:
						gestureID += 1
						frameNum = 0
		descriptors = self.desList[0]
		for des in self.desList:
			descriptors = np.vstack((descriptors, des))
		return descriptors

	def kmeans(self, descriptors):
		if descriptors.dtype != "float32":
			descriptors = np.float32(descriptors)
		voc,variance = kmeans(descriptors, self.numWords, 1)
		return voc,variance

	def bow(self, voc):
		for gestureID in range(self.numGestures):
			for numFrame in range(self.numFramesPerGesture):
				words, distance = vq(self.desList[gestureID*self.numFramesPerGesture+numFrame], voc)
				for w in words:
					self.trainData[gestureID*self.numFramesPerGesture+numFrame][w] += 1
				self.trainLabels[gestureID*self.numFramesPerGesture+numFrame][0] = gestureID
