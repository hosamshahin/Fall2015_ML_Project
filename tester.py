import sys
import cv2
import numpy as np
from scipy.cluster.vq import kmeans, vq
from sklearn import svm
from feature_extractor import FeatureExtractor
from hand_tracker import HandTracker

class Tester(object):
    def __init__(self, numGestures, minDescriptorsPerFrame, numWords, descType, numPredictions, parent):
        self.numGestures = numGestures
        self.numWords = numWords
        self.minDescriptorsPerFrame = minDescriptorsPerFrame
        self.parent = parent
        self.classifier = None
        self.windowName = "Testing preview"
        self.handWindowName = "Cropped hand"
        self.binaryWindowName = "Binary frames"
        self.predictionList = [-1]*numPredictions;
        self.handTracker = HandTracker(kernelSize=7, thresholdAngle=0.4, defectDistFromHull=30, parent=self)
        self.featureExtractor = FeatureExtractor(type=descType, parent=self)

    def initialize(self, clf):
        self.classifier = clf

    def test_on_video(self):
        vc = self.parent.vc
        while(vc.isOpened()):
            ret,im = vc.read()
            im = cv2.flip(im, 1)
            imhsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            self.handTracker.colorProfiler.draw_color_windows(im)
            cv2.imshow(self.windowName, im)
            k = cv2.waitKey(1)
            if k == 32: # space
                break
            elif k == 27:
                sys.exit(0)

        self.handTracker.colorProfiler.run(imhsv)
        binaryIm = self.handTracker.get_binary_image(imhsv)
        cnt,hull,centroid,defects = self.handTracker.initialize_contour(binaryIm)

        while(vc.isOpened()):
            ret,im = vc.read()
            im = cv2.flip(im, 1)
            imhsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            binaryIm = self.handTracker.get_binary_image(imhsv)
            cnt,hull,centroid,defects = self.handTracker.get_contour(binaryIm)
            imCopy = 1*im
            if cnt is not None:
                numDefects = defects.shape[0]
                cropImage = self.handTracker.get_cropped_image(im, cnt)
                cropImageGray = self.handTracker.get_cropped_image(imgray, cnt)
                kp,des = self.featureExtractor.get_keypoints_and_descriptors(cropImageGray)
                if des is not None and des.shape[0] >= 0:
                    self.featureExtractor.draw_keypoints(cropImage, kp)
                if des is not None and des.shape[0] >= self.minDescriptorsPerFrame and self.is_hand(defects):
                    words, distance = vq(des, self.classifier.voc)
                    testData = np.zeros(self.numWords, "float32")
                    for w in words:
                        testData[w] += 1
                    prediction,score = self.predict(testData)
                    if max(score) > 0:
                        self.handTracker.draw_on_image(imCopy, cnt=False, hullColor=(0,255,0))
                    else:
                        self.handTracker.draw_on_image(imCopy, cnt=False, hullColor=(0,0,255))
                        prediction = -1
                else:
                    self.handTracker.draw_on_image(imCopy, cnt=False, hullColor=(0,0,255))
                    prediction = -1
                cv2.imshow(self.handWindowName,cropImage)
            else:
                prediction = -1
            #self.insert_to_prediction_list(prediction)
            #prediction,predictionCount = self.most_common(self.predictionList)
            #if prediction>=0:
            if prediction>=0 and numDefects==self.classifier.medianDefects[prediction]:
                #print modePrediction, predictionList
                print prediction
            cv2.imshow(self.binaryWindowName, binaryIm)
            cv2.imshow(self.windowName,imCopy)
            k = cv2.waitKey(1)
            if k == 27: # space
                break

    def predict(self, testData):
        prediction = self.classifier.predict(testData.reshape(1,-1))
        score = self.classifier.decision_function(testData.reshape(1,-1))
        return prediction[0], score[0]

    def insert_to_prediction_list(self, prediction):
        self.predictionList.append(prediction)
        self.predictionList = self.predictionList[1:]

    def most_common(self, lst):
        for i in range(1,len(lst)-1):
            if lst[i] != lst[i-1] and lst[i] != lst[i+1]:
                lst[i] = -1
        e = max(set(lst), key=lst.count)
        return e,lst.count(e)

    def is_hand(self, defects):
        if defects.shape[0] > 4:
            return False
        else:
            return True