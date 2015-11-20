import sys
import cv2
import numpy as np
from scipy.cluster.vq import kmeans, vq
from sklearn import svm
from feature_extractor import FeatureExtractor
from hand_tracker import HandTracker

class Trainer(object):
    def __init__(self, numGestures, numFramesPerGesture, minDescriptorsPerFrame, numWords, descType, parent):
        self.numGestures = numGestures
        self.numFramesPerGesture = numFramesPerGesture
        self.numWords = numWords
        self.minDescriptorsPerFrame = minDescriptorsPerFrame
        self.parent = parent
        self.desList = []
        self.voc = None
        self.classifier = None
        self.trainData = np.zeros((numGestures*numFramesPerGesture, numWords), "float32")
        self.trainLabels = np.zeros((numGestures*numFramesPerGesture), "uint32")
        self.windowName = "Training preview"
        self.handWindowName = "Cropped hand"
        self.binaryWindowName = "Binary frames"
        self.handTracker = HandTracker(kernelSize=7, thresholdAngle=0.4, defectDistFromHull=30, parent=self)
        self.featureExtractor = FeatureExtractor(type=descType, parent=self)
        self.numDefects = np.zeros((self.numGestures,self.numFramesPerGesture), "uint8")
        self.firstFrameList = []

    def extract_descriptors_from_video(self):
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

        gestureID = 1
        frameNum = 0
        gestureFrameID = 0
        captureFlag = False
        while(vc.isOpened()):
            ret,im = vc.read()
            im = cv2.flip(im, 1)
            imhsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            binaryIm = self.handTracker.get_binary_image(imhsv)
            cnt,hull,centroid,defects = self.handTracker.get_contour(binaryIm)
            imCopy = 1*im
            if cnt is not None:
                cropImage,cropPoints = self.handTracker.get_cropped_image_from_cnt(im, cnt, 0.05)
                cropImageGray = self.handTracker.get_cropped_image_from_points(imgray, cropPoints)
                kp,des = self.featureExtractor.get_keypoints_and_descriptors(cropImageGray)
                if des is not None and des.shape[0] >= 0:
                    self.featureExtractor.draw_keypoints(cropImage, kp)
                if captureFlag:
                    if frameNum == 0:
                        self.firstFrameList.append(im)
                    if des is not None and des.shape[0] >= self.minDescriptorsPerFrame and self.is_hand(defects):
                        self.desList.append(des)
                        self.handTracker.draw_on_image(imCopy, cnt=False, hullColor=(0,255,0))
                        self.numDefects[gestureID-1][frameNum] = defects.shape[0]
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
                cv2.imshow(self.handWindowName, cropImage)
            cv2.imshow(self.binaryWindowName, binaryIm)
            cv2.imshow(self.windowName,imCopy)
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
        cv2.destroyAllWindows()

    def kmeans(self, numIters):
        print "Running k-means clustering with {0} iterations...".format(numIters)
        descriptors = self.desList[0]
        for des in self.desList:
            descriptors = np.vstack((descriptors, des))
        if descriptors.dtype != "float32":
            descriptors = np.float32(descriptors)
        self.voc,variance = kmeans(descriptors, self.numWords, numIters)
        return variance

    def bow(self):
        print "Extracting bag-of-words features..."
        for gestureID in range(self.numGestures):
            for numFrame in range(self.numFramesPerGesture):
                words, distance = vq(self.desList[gestureID*self.numFramesPerGesture+numFrame], self.voc)
                for w in words:
                    self.trainData[gestureID*self.numFramesPerGesture+numFrame][w] += 1
                self.trainLabels[gestureID*self.numFramesPerGesture+numFrame] = gestureID+1

    def linear_svm(self):
        print "Training linear SVM classifier..."
        lin_clf = svm.LinearSVC()
        valScore = self.leave_one_out_validate(lin_clf)
        lin_clf.fit(self.trainData, self.trainLabels)
        self.classifier = lin_clf
        self.classifier.voc = self.voc
        self.classifier.medianDefects = np.median(self.numDefects, axis=1)
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

    def is_hand(self, defects):
        if defects.shape[0] > 4:
            return False
        else:
            return True
