import numpy as np
from trainer import Trainer
from tester import Tester

class Recognizer(object):
    def __init__(self, vc, descriptor, opts):
        self.vc = vc
        ret,im = vc.read()
        self.imHeight,self.imWidth,self.channels = im.shape
        self.trainer = Trainer(numGestures=opts.num, numFramesPerGesture=opts.frames, minDescriptorsPerFrame=opts.desc, numWords=opts.words, descType=opts.type, parent=self)
        self.tester = Tester(numGestures=opts.num, minDescriptorsPerFrame=opts.desc, numWords=opts.words, descType=opts.type, numPredictions=7, parent=self)

    def train_from_video(self):
        self.trainer.extract_descriptors_from_video()
        variance = self.trainer.kmeans(30)
        self.trainer.bow()
        score = self.trainer.linear_svm()
        return score

    def test_on_video(self, clf):
        self.tester.initialize(clf)
        self.tester.test_on_video()

