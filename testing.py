import sys
import os
import cv2
import numpy as np
from recognizer import Recognizer
import cPickle as pickle
from feature_extractor import FeatureExtractor
import fnmatch
from scipy.cluster.vq import kmeans, vq
from q_learning import QLearn
import imutils
from hand_tracker import HandTracker
from sklearn.svm import SVC

def get_new_directory(numGestures, descType):
    i = 0
    while(1):
        trainDirName = "{0}_{1}_train{2}".format(numGestures, descType, i)
        trainDirPath = get_traindir_path(trainDirName)
        if not os.path.exists(trainDirPath):
            return trainDirName
        i += 1

def get_traindir_path(dirName):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "TrainData", dirName)

def get_gesture_parentdir_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "GestureImages")

def get_gesture_mask_parentdir_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "GestureMasks")

def pickle_files(outTrainDir, trainer):
    outTrainDirPath = get_traindir_path(outTrainDir)
    if not os.path.exists(outTrainDirPath):
        os.makedirs(outTrainDirPath)
    descriptorFile = os.path.join(outTrainDirPath, "descriptors.pkl")
    classifierFile = os.path.join(outTrainDirPath, "classifier.pkl")
    with open(descriptorFile, 'wb') as output:
        desList = trainer.desList
        pickle.dump(desList, output, pickle.HIGHEST_PROTOCOL)
        trainLabels = trainer.trainLabels
        pickle.dump(trainLabels, output, pickle.HIGHEST_PROTOCOL)
    with open(classifierFile, 'wb') as output:
        clf = trainer.classifier
        pickle.dump(clf, output, pickle.HIGHEST_PROTOCOL)
    save_first_frames(outTrainDirPath, recognizer.trainer.firstFrameList)
    return clf

def get_full_descriptors_and_labels(dirNamelist):
    desList = []
    trainLabelsList = []
    for dirName in dirNamelist:
        dirPath = get_traindir_path(dirName)
        descFile = os.path.join(dirPath, "descriptors.pkl")
        if not os.path.exists(descFile):
            print "Descriptor file not present for directory {0}".format(dirName)
            continue
        with open(descFile, 'rb') as input:
            des = pickle.load(input)
            trainLabels = pickle.load(input)
        desList += des
        trainLabelsList += trainLabels
    return desList,trainLabelsList

def save_first_frames(dirPath, frameList):
    for i,im in enumerate(frameList):
        filepath = os.path.join(dirPath, "gesture_{0}.jpg".format(i+1))
        cv2.imwrite(filepath, im)

def cmd_parser():
        """
        Parse user's command line and return opts object and args
        """
        from optparse import OptionParser
        parser = OptionParser()
        parser.add_option("-n", "--num",        help="Number of gestures", default=2, type="int")
        parser.add_option("-f", "--frames",     help="Number of frames to train on per gesture", default=100, type="int")
        parser.add_option("-w", "--words",      help="Number of visual words", default=100, type="int")
        parser.add_option("-d", "--desc",       help="Minimum number of descriptors per frame", default=100, type="int")
        parser.add_option("-t", "--type",       help="Descriptor type", action="store", type="string", default="surf")
        parser.add_option("-k", "--kernel",     help="Kernel type", action="store", type="string", default="linear")
        parser.add_option("-i", "--iter",       help="Number of iterations for k-means clustering", type="int", default=30)
        parser.add_option(      "--doc",        help="Print the docstring", action="store_true", default=False)
        parser.add_option(      "--notrain",    help="Whether to train the system", action="store_true", default=False)
        parser.add_option(      "--nocollect",  help="Whether to collect train descriptors", action="store_true", default=False)
        parser.add_option(      "--notest",     help="Whether to run the system in test mode", action="store_true", default=False)  
        parser.add_option(      "--traindir",   help="Training directory(ies)", action="store", type="string")
        parser.add_option(      "--testdir",    help="Test directory", action="store", type="string")
        parser.add_option(      "--trainmask",  help="Type of masking for image data", default=0, type="int")
        return parser.parse_args()

def process_opts(opts):
        if opts.notrain:
                opts.nocollect = True
        if opts.notrain:
                if opts.traindir is None:
                        print "Specify training directory with --traindir"
                        exit(0)
                inTrainDirs = opts.traindir.split(',')[0]
                inputMode = None
        elif opts.nocollect:
                if opts.traindir is None:
                        print "Specify training directory with --traindir"
                        exit(0)
                inTrainDirs = opts.traindir.split(',')
                inputMode = "descriptors"
        else:
                if opts.traindir is None:
                        inTrainDirs = None
                        inputMode = "video"
                else:
                        inTrainDirs = opts.traindir.split(',')
                        inputMode = "images"
        return inputMode,inTrainDirs





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

clf = SVC(probability=True)
clf.fit(trainData, trainLabels)

def is_hand(defects):
    if defects.shape[0] > 5:
        return False
    else:
        return True

def learningFrames(vc):
    frameNum = 1
    while(vc.isOpened()):
        ret,im = vc.read()
        im = cv2.flip(im, 1)
        imhsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        handTracker.colorProfiler.draw_color_windows(im, imhsv)
        cv2.imshow("color", im)
        k = cv2.waitKey(1)
        if k == 32: # space
            break
        elif k == 27:
            sys.exit(0)
    
    handTracker.colorProfiler.run()
    print "1"
    ret, frame = vc.read()
    frame = cv2.flip(frame, 1)
    imhsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    binaryIM = handTracker.get_binary_image(imhsv)
    print "2"
    cnt,hull,centroids, defects = handTracker.initialize_contour(binaryIM)
    print "3"

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        ret, frame = vc.read()
        frame = cv2.flip(frame, 1)
        imhsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binaryIm = handTracker.get_binary_image(imhsv)
        cnt,hull,centroids, defects = handTracker.get_contour(binaryIm)
        print "4"
        frameCopy = 1*frame
        testData = None
        prediction = -1
        score = -1
        if cnt is not None:
            print "5"
            numDefects = defects.shape[0]
            cropImage,cropPoints = handTracker.get_cropped_image_from_cnt(frame, cnt, 0.05)
            cropImageGray = handTracker.get_cropped_image_from_points(imgray, cropPoints)
            kp = featureExtractor.get_keypoints(cropImageGray)
            cropCnt = handTracker.get_cropped_contour(cnt, cropPoints)
            kp = featureExtractor.get_keypoints_in_contour(kp, cropCnt)
            kp, des = featureExtractor.compute_descriptors(cropImageGray, kp)

            if des is not None and des.shape[0] >= 0:
                featureExtractor.draw_keypoints(cropImage, kp)
            
            des = np.float32(des)
            if des is not None and des.shape[0] >= 10 and is_hand(defects):

                voc, variance = kmeans(des, numWords, 30)
                words, distance = vq(des, voc)
                testData = np.zeros(numWords, "float32")
                for w in words:
                    testData[w] += 1
                normTestData = np.linalg.norm(testData, ord=2) * np.ones(numWords)
                testData = np.divide(testData, normTestData)
                class_probabilities = clf.predict_proba(testData)
                print frameNum
                print class_probabilities
                cv2.imshow("frames", cropImage)
                frameNum = frameNum + 1
            else:
                handTracker.draw_on_image(frameCopy, cnt=False, hullColor=(0,0,255))
        else:
            prediction = -1

        cv2.imshow('testing', frameCopy)

def no_hand(vc):
    firstFrame = None
    hand = False
    while True:
        ret, frame = vc.read()
        frame = cv2.flip(frame, 1)
        text = "Unoccupied"
        hand = False
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if firstFrame is None:
            firstFrame = gray
            continue

        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        
        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < 20000:
                continue
 
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Found hand"
            hand = True

        cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow("Hand Search", frame)
        key = cv2.waitKey(1) & 0xFF
        if hand:
            break
        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break

    if hand:
        learningFrames(vc)


handTracker = None
opts,args = cmd_parser()
inputMode,inTrainDirs = process_opts(opts)
vc = cv2.VideoCapture(0)
try:
    #recognizer = Recognizer(vc=vc, opts=opts)
    #score = recognizer.train_from_video()
    #print "Training score = {0}".format(score)
    #outTrainDir = get_new_directory(opts.num, opts.type)
    # ret,frame1 = vc.read()
    # ret,frame2 = vc.read()
    # while True:
    #     cv2.imshow('img1', frame1)
    #     cv2.imshow('img2', frame2)
            
    #     if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y' 
    #         #cv2.imwrite('images/c1.png',frame)
    #         cv2.destroyAllWindows()
    #         break
    r,f = vc.read()
    handTracker = HandTracker(kernelSize=7, thresholdAngle=0.4, defectDistFromHull=30, parent=None, frame=f)
    no_hand(vc)

    vc.release()
    cv2.destroyAllWindows()
except:
    vc.release()
    cv2.destroyAllWindows()
    import traceback
    traceback.print_exc(file=sys.stdout)
