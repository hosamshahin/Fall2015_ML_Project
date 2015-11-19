import cv2
import numpy as np
import math

class HandTracker(object):
	def __init__(self, colorProfiler, kernelSize, thresholdAngle, defectDistFromHull):
		self.colorProfiler = colorProfiler
		self.kernelSize = kernelSize
		self.kernel = np.ones((self.kernelSize,self.kernelSize),np.uint8)
		self.thresholdAngle = thresholdAngle
		self.defectDistFromHull = defectDistFromHull

	def get_binary_image(self, imhsv):
		imHeight,imWidth,channels = imhsv.shape
		finalBinIm = np.zeros((imHeight,imWidth), dtype=np.uint8)
		for i in range(self.colorProfiler.centers.shape[0]):
			hsvMint = self.colorProfiler.hsvRanges[i][0]
			hsvMaxt = self.colorProfiler.hsvRanges[i][1]
			binHSV = cv2.inRange(imhsv, hsvMint, hsvMaxt)
			binIm = cv2.morphologyEx(binHSV, cv2.MORPH_CLOSE, self.kernel)
			#binIm, final_hsv_range = utils.get_binary_image_hsv(imhsv, points_hsv[i], hsv_range, kernel_size)
			finalBinIm = np.add(finalBinIm, binIm)
		closing = cv2.morphologyEx(finalBinIm, cv2.MORPH_CLOSE, self.kernel)
		median = cv2.medianBlur(closing, self.kernelSize)
		return median

	def get_contour(self, binaryIm):
		#binaryIm = self.get_binary_image(imhsv)
		contours, hierarchy = cv2.findContours(binaryIm,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		maxArea = 0
		ci = -1
		for i,c in enumerate(contours):
			h = cv2.convexHull(c)
			if cv2.pointPolygonTest(h, tuple(self.prevCentroid), False) == 1:
				#epsilon = 0.001*cv2.arcLength(c,True)
				#c = cv2.approxPolyDP(c,epsilon,True)
				area = cv2.contourArea(c)
				if area > maxArea:# and area > area_threshold:
					maxArea = area
					ci = i
					cnt = c
					hull = h
		if ci != -1:
			self.prevCnt = cnt
			self.prevHull = hull
			M = cv2.moments(cnt)
			#warped_cnt = deskew(cnt, 10, M)
			self.prevCentroid = np.array([int(M['m10']/M['m00']), int(M['m01']/M['m00'])])
			hullPoints = cv2.convexHull(cnt, returnPoints=False)
			defects = cv2.convexityDefects(cnt,hullPoints)
			#prev_end = tuple(np.zeros(2, dtype=np.int32))
			finalDefects = []
			for i in range(defects.shape[0]):
				s,e,f,d = defects[i,0]
				start = cnt[s][0]
				end = cnt[e][0]
				far = cnt[f][0]
				if self.threshold_angle_diff(tuple(start), tuple(end), tuple(far), self.thresholdAngle) and cv2.pointPolygonTest(hull, tuple(far), True) > self.defectDistFromHull:
					finalDefects.append(far)
			self.prevDefects = np.array(finalDefects)
			return self.prevCnt, self.prevHull, self.prevCentroid, self.prevDefects
		else:
			return None, None, None, None

	def initialize_contour(self, binaryIm):
		#binaryIm = self.get_binary_image(imhsv)
		contours, hierarchy = cv2.findContours(binaryIm,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		maxArea = 0
		ci = -1
		for i,c in enumerate(contours):
			h = cv2.convexHull(c)
			if cv2.pointPolygonTest(c, tuple(self.colorProfiler.centers[0]), False) == 1:
				#epsilon = 0.001*cv2.arcLength(c,True)
				#c = cv2.approxPolyDP(c,epsilon,True)
				area = cv2.contourArea(c)
				if area > maxArea:# and area > area_threshold:
					maxArea = area
					ci = i
					cnt = c
					hull = h
		if ci != -1:
			self.prevCnt = cnt
			self.prevHull = hull
			M = cv2.moments(cnt)
			self.prevCentroid = np.array([int(M['m10']/M['m00']), int(M['m01']/M['m00'])])
			hullPoints = cv2.convexHull(cnt, returnPoints=False)
			defects = cv2.convexityDefects(cnt,hullPoints)
			#prev_end = tuple(np.zeros(2, dtype=np.int32))
			finalDefects = []
			for i in range(defects.shape[0]):
				s,e,f,d = defects[i,0]
				start = cnt[s][0]
				end = cnt[e][0]
				far = cnt[f][0]
				if self.threshold_angle_diff(tuple(start), tuple(end), tuple(far), self.thresholdAngle) and cv2.pointPolygonTest(hull, tuple(far), True) > self.defectDistFromHull:
					finalDefects.append(far)
			self.prevDefects = np.array(finalDefects)
			return self.prevCnt, self.prevHull, self.prevCentroid, self.prevDefects
		else:
			return None, None, None, None

	def get_cropped_image(self, image, cnt):
		x,y,w,h = cv2.boundingRect(cnt)
		#cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
		cropIm = image[y:y+h,x:x+w]
		return cropIm

	def draw_on_image(self, image, cnt=True, hull=True, centroid=True, defects=True, cntColor=(0,255,0), hullColor=(255,0,0)):
		if cnt is True:
			cv2.drawContours(image, [self.prevCnt], 0, cntColor, 3)
		elif cnt is not False:
			cv2.drawContours(image, [cnt], 0, cntColor, 3)
		if hull is True:
			cv2.drawContours(image, [self.prevHull], 0, hullColor, 3)
		elif hull is not False:
			cv2.drawContours(image, [hull], 0, hullColor, 3)
		if centroid is True:
			cv2.circle(image,tuple(self.prevCentroid),5,[0,0,255],-1)
		elif centroid is not False:
			cv2.circle(image,tuple(centroid),5,[0,0,255],-1)
		if defects is True:
			for i in range(self.prevDefects.shape[0]):
				cv2.circle(image,tuple(self.prevDefects[i]),5,[0,0,255],-1)
		elif defects is not False:
			for i in range(defects.shape[0]):
				cv2.circle(image,tuple(defects[i]),5,[0,0,255],-1)

	def find_angle(self, pt1, pt2):
		x1 = float(pt1[0])
		x2 = float(pt2[0])
		y1 = float(pt1[1])
		y2 = float(pt2[1])
		if (x2-x1) == 0:
			if (y2-y1) > 0:
				return math.pi/2
			elif (y2-y1) < 0:
				return -math.pi/2
			else:
				return None
		theta = math.atan((y2-y1)/(x2-x1))
		if (x2-x1) < 0:
			if theta > 0:
				theta -= math.pi
			else:
				theta += math.pi
		return theta	

	def threshold_angle_diff(self, start, end, far, threshold):
		theta1 = self.find_angle(far, start)
		theta2 = self.find_angle(far, end)
		if theta1 is None or theta2 is None:
			return False
		if abs(theta1-theta2)/math.pi <= threshold:
			return True
		else:
			return False