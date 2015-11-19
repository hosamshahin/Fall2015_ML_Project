import sys
import cv2
import numpy as np
from color_profiler import ColorProfiler
from hand_tracker import HandTracker
from feature_extractor import FeatureExtractor
from trainer import Trainer
from scipy.cluster.vq import vq

def add_centers(centers, coord_width, coord_height, im_width, im_height):
    if centers is None:
        return np.array([[int(im_width/2),int(im_height/2)],[int(im_width/2+coord_width),int(im_height/2)],[int(im_width/2),int(im_height/2+coord_height)],[int(im_width/2-coord_width),int(im_height/2)],[int(im_width/2),int(im_height/2-coord_height)],[int(im_width/2-coord_width/2),int(im_height/2+coord_height/2)],[int(im_width/2+coord_width/2),int(im_height/2-coord_height/2)],[int(im_width/2+coord_width/2),int(im_height/2+coord_height/2)],[int(im_width/2-coord_width/2),int(im_height/2-coord_height/2)]])
    else:
        return np.append(centers, np.array([[int(im_width/2),int(im_height/2)],[int(im_width/2+coord_width),int(im_height/2)],[int(im_width/2),int(im_height/2+coord_height)],[int(im_width/2-coord_width),int(im_height/2)],[int(im_width/2),int(im_height/2-coord_height)],[int(im_width/2-coord_width/2),int(im_height/2+coord_height/2)],[int(im_width/2+coord_width/2),int(im_height/2-coord_height/2)],[int(im_width/2+coord_width/2),int(im_height/2+coord_height/2)],[int(im_width/2-coord_width/2),int(im_height/2-coord_height/2)]]), axis=0)

def insert_to_prediction_list(predictionList, prediction):
    predictionList.append(prediction)
    predictionList = predictionList[1:]
    return predictionList

def most_common(lst):
    for i in range(1,len(lst)-1):
        if lst[i] != lst[i-1] and lst[i] != lst[i+1]:
            lst[i] = -1
        e = max(set(lst), key=lst.count)
        return e,lst.count(e)

def cmd_parser():
    """
    Parse user's command line and return opts object and args
    """
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-n", "--num",        help="Number of gestures", default=3, type="int")
    parser.add_option("-f", "--frames",     help="Number of frames to train on per gesture", default=100, type="int")
    parser.add_option("-w", "--words",      help="Number of visual words", default=100, type="int")
    parser.add_option("-d", "--desc",       help="Minimum number of descriptors per frame", default=100, type="int")
    parser.add_option("-s", "--seed",       help="Seed for reproducibility", action="store", type="int")
    parser.add_option(      "--doc",        help="Print the docstring", action="store_true", default=False)
    parser.add_option(      "--train",      help="Whether to train the system", action="store_true", default=False)
    parser.add_option(      "--notest",     help="Whether to run the system in test mode", action="store_false", default=True)

    return parser.parse_args()

#########################
### Main script entry ###
#########################
if __name__ == "__main__":
    opts,args = cmd_parser()
    numGestures = opts.num
    numFramesPerGesture = opts.frames
    numWords = opts.words
    minDescriptorsPerFrame = opts.desc
    if opts.seed is None:
        import random
        rseed = random.randint(0x00000000, 0xffffffff)
    else:
        rseed = opts.seed
    windowName = 'preview'
    handWindowName = 'hand window'
    vc = None
    try:
        vc = cv2.VideoCapture(0);
        #vc.set(cv2.CAP_PROP_FPS,15)
        cv2.namedWindow(windowName)
        cv2.namedWindow(handWindowName)


        ret,prevIm = vc.read()
        imHeight,imWidth,channels = prevIm.shape
        centers = add_centers(None, 60, 120, imWidth, imHeight)
        centers = add_centers(centers, 30, 60, imWidth, imHeight)
        centers = add_centers(centers, 45, 90, imWidth, imHeight)
        hsvRange = np.array([1,160,160])
        colorProfiler = ColorProfiler(centers=centers, windowSize=15, hsvRange=hsvRange)
        handTracker = HandTracker(colorProfiler=colorProfiler, kernelSize=7, thresholdAngle=0.4, defectDistFromHull=30)
        featureExtractor = FeatureExtractor(type='surf')
        #featureExtractor.model = cv2.ORB_create()
        trainer = Trainer(numGestures=numGestures, numFramesPerGesture=numFramesPerGesture, numWords=numWords, handTracker=handTracker, featureExtractor=featureExtractor)

        choice = 'y'
        if choice != 'y':
            ## Training phase
            while(vc.isOpened()):
                ret,im = vc.read()
                im = cv2.flip(im, 1)
                imhsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
                trainer.handTracker.colorProfiler.draw_color_windows(im)
                cv2.imshow(windowName, im)
                k = cv2.waitKey(1)
                if k == 32: # space
                    break
                elif k == 27:
                    vc.release();
                    sys.exit(0)

            trainer.handTracker.colorProfiler.run(imhsv)
            binaryIm = trainer.handTracker.get_binary_image(imhsv)
            cnt,hull,centroid,defects = trainer.handTracker.initialize_contour(binaryIm)
            descriptors = trainer.extract_descriptors_from_video(vc, windowName, handWindowName, 1, minDescriptorsPerFrame, imWidth, imHeight, imsave=False)
            cv2.destroyAllWindows()
            variance = trainer.kmeans(descriptors, 30)
            trainer.bow()
            np.save("train_data_surf3.npy", trainer.trainData)
            np.save("train_labels_surf3.npy", trainer.trainLabels)
            np.save("train_voc_surf3.npy", trainer.voc)
        else:
            trainer.trainData = np.load("train_data_surf3.npy")
            trainer.trainLabels = np.load("train_labels_surf3.npy")
            trainer.voc = np.load("train_voc_surf3.npy")
        
        score = trainer.linear_svm()
        print "Training score = {0}".format(score)

        ## Testing phase
        while(vc.isOpened()):
            ret,im = vc.read()
            im = cv2.flip(im, 1)
            imhsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            trainer.handTracker.colorProfiler.draw_color_windows(im)
            cv2.imshow(windowName, im)
            k = cv2.waitKey(1)
            if k == 32: # space
                break
            elif k == 27:
                sys.exit(0)

            trainer.handTracker.colorProfiler.run(imhsv)
            binaryIm = trainer.handTracker.get_binary_image(imhsv)
            cnt,hull,centroid,defects = trainer.handTracker.initialize_contour(binaryIm)
            predictionList = [-1]*7;

            while(vc.isOpened()):
                ret,im = vc.read()
                im = cv2.flip(im, 1)
                imhsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
                imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                binaryIm = trainer.handTracker.get_binary_image(imhsv)
                cnt,hull,centroid,defects = trainer.handTracker.get_contour(binaryIm)
                imCopy = 1*im
                if cnt is not None:
                    cropImage = trainer.handTracker.get_cropped_image(im, cnt)
                    cropImageGray = trainer.handTracker.get_cropped_image(imgray, cnt)
                    kp,des = trainer.featureExtractor.get_keypoints_and_descriptors(cropImageGray)
                    if des is not None and des.shape[0] >= 0:
                        trainer.featureExtractor.draw_keypoints(cropImage, kp)
                    if des is not None and des.shape[0] >= minDescriptorsPerFrame:
                        words, distance = vq(des, trainer.voc)
                        testData = np.zeros(numWords, "float32")
                        for w in words:
                            testData[w] += 1

                        prediction,score = trainer.predict(testData)
                        if max(score) > 0:
                            trainer.handTracker.draw_on_image(imCopy, cnt=False, hullColor=(0,255,0))
                            #print prediction, score
                        else:
                            trainer.handTracker.draw_on_image(imCopy, cnt=False, hullColor=(0,0,255))
                            prediction = -1

                    else:
                        trainer.handTracker.draw_on_image(imCopy, cnt=False, hullColor=(0,0,255))
                        prediction = -1
                        cv2.imshow(handWindowName,cropImage)

                else:
                    prediction = -1

                predictionList = insert_to_prediction_list(predictionList, prediction)
                modePrediction,predictionCount = most_common(predictionList)
                if modePrediction>=0:
                    #print modePrediction, predictionList
                    print modePrediction

                cv2.imshow(windowName,imCopy)
                k = cv2.waitKey(1)
                if k == 27: # space
                    break

    except:
        vc.release()
