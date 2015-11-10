import cv2
import numpy as np
from color_profiler import ColorProfiler
from hand_tracker import HandTracker
from feature_extractor import FeatureExtractor
from trainer import Trainer

def add_centers(centers, coord_width, coord_height, im_width, im_height):
	if centers is None:
		return np.array([[int(im_width/2),int(im_height/2)],[int(im_width/2+coord_width),int(im_height/2)],[int(im_width/2),int(im_height/2+coord_height)],[int(im_width/2-coord_width),int(im_height/2)],[int(im_width/2),int(im_height/2-coord_height)],[int(im_width/2-coord_width/2),int(im_height/2+coord_height/2)],[int(im_width/2+coord_width/2),int(im_height/2-coord_height/2)],[int(im_width/2+coord_width/2),int(im_height/2+coord_height/2)],[int(im_width/2-coord_width/2),int(im_height/2-coord_height/2)]])
	else:
		return np.append(centers, np.array([[int(im_width/2),int(im_height/2)],[int(im_width/2+coord_width),int(im_height/2)],[int(im_width/2),int(im_height/2+coord_height)],[int(im_width/2-coord_width),int(im_height/2)],[int(im_width/2),int(im_height/2-coord_height)],[int(im_width/2-coord_width/2),int(im_height/2+coord_height/2)],[int(im_width/2+coord_width/2),int(im_height/2-coord_height/2)],[int(im_width/2+coord_width/2),int(im_height/2+coord_height/2)],[int(im_width/2-coord_width/2),int(im_height/2-coord_height/2)]]), axis=0)

numGestures = 5
numFramesPerGesture = 40
numWords = 100
windowName = 'preview'

vc = cv2.VideoCapture(0)
vc.set(cv2.CAP_PROP_FPS,15)
cv2.namedWindow(windowName)

ret,prevIm = vc.read()
imHeight,imWidth,channels = prevIm.shape
centers = add_centers(None, 60, 120, imWidth, imHeight)
centers = add_centers(centers, 30, 60, imWidth, imHeight)
centers = add_centers(centers, 45, 90, imWidth, imHeight)
hsvRange = np.array([1,160,160])
colorProfiler = ColorProfiler(centers=centers, windowSize=15, hsvRange=hsvRange)
handTracker = HandTracker(colorProfiler=colorProfiler, kernelSize=7, thresholdAngle=0.4, defectDistFromHull=30)
featureExtractor = FeatureExtractor(type='orb')
#featureExtractor.model = cv2.ORB_create()

while(vc.isOpened()):
	ret,im = vc.read()
	im = cv2.flip(im, 1)
	imhsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
	colorProfiler.draw_color_windows(im)
	cv2.imshow(windowName, im)
	k = cv2.waitKey(1)
	if k == 32: # space
		break

colorProfiler.run(imhsv)
cnt,hull,centroid,defects = handTracker.initialize_contour(imhsv)
trainer = Trainer(numGestures=numGestures, numFramesPerGesture=numFramesPerGesture, numWords=numWords, handTracker=handTracker, featureExtractor=featureExtractor)

choice = raw_input("Have you extracted the features already (y/n)?")
if choice != 'y':
	descriptors = trainer.extract_descriptors_from_video(vc, windowName)
	voc,variance = trainer.kmeans(descriptors)
	trainer.bow(voc)

	np.save("train_data.npy", trainer.trainData)
	np.save("train_labels.npy", trainer.trainLabels)
else:
	trainer.trainData = np.load("train_data.npy")
	trainer.trainLabels = np.load("train_labels.npy")

cv2.destroyAllWindows()