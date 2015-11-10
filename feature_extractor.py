import cv2

class FeatureExtractor(object):
	def __init__(self, type):
		self.type = type
		if type == 'orb':
			self.model = cv2.ORB_create()
		else:
			self.model = None

	def get_keypoints_and_descriptors(self, image):
		kp = self.model.detect(image, None)
		kp, des = self.model.compute(image, kp)
		return kp,des

	def draw_keypoints(self, image, kp):
		cv2.drawKeypoints(image, kp, image, color=(0,255,0), flags=0)