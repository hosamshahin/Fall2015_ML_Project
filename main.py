import sys
import os
import cv2
import numpy as np
from scipy.cluster.vq import vq
from recognizer import Recognizer
import cPickle as pickle

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

def get_full_descriptors(dirNamelist):
    desList = []
    for dirName in dirNamelist:
        dirPath = get_traindir_path(dirName)
        descFile = os.path.join(dirPath, "descriptors.pkl")
        if not os.path.exists(descFile):
            print "Descriptor file not present for directory {0}".format(dirName)
            continue
        with open(descFile, 'rb') as input:
            des = pickle.load(input)
        desList += des
    return desList

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
    parser.add_option("-n", "--num",        help="Number of gestures", default=3, type="int")
    parser.add_option("-f", "--frames",     help="Number of frames to train on per gesture", default=100, type="int")
    parser.add_option("-w", "--words",      help="Number of visual words", default=100, type="int")
    parser.add_option("-d", "--desc",       help="Minimum number of descriptors per frame", default=100, type="int")
    parser.add_option("-t", "--type",       help="Descriptor type", action="store", type="string", default="surf")
    parser.add_option(      "--doc",        help="Print the docstring", action="store_true", default=False)
    parser.add_option(      "--notrain",    help="Whether to train the system", action="store_true", default=False)
    parser.add_option(      "--nocollect",  help="Whether to collect train descriptors", action="store_true", default=False)
    parser.add_option(      "--notest",     help="Whether to run the system in test mode", action="store_true", default=False)  
    parser.add_option(      "--traindir",   help="Training directory(ies)", action="store", type="string")
    return parser.parse_args()

#########################
### Main script entry ###
#########################
if __name__ == "__main__":
    opts,args = cmd_parser()
    #if opts.seed is None:
    #    import random
    #    rseed = random.randint(0x00000000, 0xffffffff)
    #else:
    #    rseed = opts.seed
    #trainDirName = "Train_{0}".format(rseed)
    #trainDirPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TrainData", trainDirName)
    #classifierFile = os.path.join(trainDirPath, 'classifier.pkl')
    if opts.notrain:
        opts.nocollect = True

    if opts.notrain or opts.nocollect:
        if opts.traindir is None:
            print "Specify training directory with --traindir"
            exit(0)
        inTrainDirs = opts.traindir.split(',')
        if opts.notrain:
            inTrainDirs = [inTrainDirs[0]]
    else:
        inTrainDirs = None
    
    if not opts.notrain and opts.nocollect:
        if len(inTrainDirs) == 1:
            outTrainDir = inTrainDirs[0]
        else:
            outTrainDir = get_new_directory(opts.num, opts.type)
    elif not opts.notrain and not opts.nocollect:
        outTrainDir = get_new_directory(opts.num, opts.type)
    else:
        outTrainDir = inTrainDirs[0]

    vc = cv2.VideoCapture(0)
    try:
        recognizer = Recognizer(vc=vc, opts=opts)

        if not opts.notrain:
            if not opts.nocollect:
                score = recognizer.train_from_video()
            else:
                descList = get_full_descriptors(inTrainDirs)
                score = recognizer.train_from_descriptors(descList)
            print "Training score = {0}".format(score)
            outTrainDirPath = get_traindir_path(outTrainDir)
            if not os.path.exists(outTrainDirPath):
                os.makedirs(outTrainDirPath)
                #if opts.notrain or opts.nocollect:
                #    print "Training data not already present for seed={0}".format(rseed)
                #    exit(0)
            descriptorFile = os.path.join(outTrainDirPath, "descriptors.pkl")
            classifierFile = os.path.join(outTrainDirPath, "classifier.pkl")
            with open(classifierFile, 'wb') as output:
                clf = recognizer.trainer.classifier
                pickle.dump(clf, output, pickle.HIGHEST_PROTOCOL)
            with open(descriptorFile, 'wb') as output:
                desList = recognizer.trainer.desList
                pickle.dump(desList, output, pickle.HIGHEST_PROTOCOL)
            save_first_frames(outTrainDirPath, recognizer.trainer.firstFrameList)
        elif not opts.notest:
            inTrainDirPath = get_traindir_path(inTrainDirs[0])
            classifierFile = os.path.join(inTrainDirPath, "classifier.pkl")
            if not os.path.exists(classifierFile):
                print "Trained classifier not present for directory {0}".format(inTrainDirs[0])
                exit(0)
            with open(classifierFile, 'rb') as input:
                clf = pickle.load(input)

        if not opts.notest:
            recognizer.test_on_video(clf)
    except:
        vc.release()
        import traceback
        traceback.print_exc(file=sys.stdout)
    print "Train directory = {0}".format(outTrainDir)