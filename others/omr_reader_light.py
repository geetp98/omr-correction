from imutils import contours
import numpy as np
from imutils.perspective import four_point_transform
import cv2
import imutils
from imutils import *
import argparse

def logImage(image, contour, color, image_name):
	img = image.copy()
	cv2.drawContours(img, contour, -1, color, 7)
	cv2.imwrite(image_name, img)

def getContours(edged, box):
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	docCnt = None

	if len(cnts) > 0:
		cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
		contour = []
		i = 0
		   
		for c in cnts:
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.02 * peri, True)

			if len(approx) == 4:
				docCnt = approx
				contour.append(docCnt)
				#logImage(image, c, 'log_contoured_image' + str(i))
				i = i+1
				if box==False or i == 2:
					contour = contours.sort_contours(contour, method='left-to-right')[0]
					return contour


def detectBubbles(questions):
	questions = cv2.cvtColor(questions, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(questions, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	questionCnts = []

	for c in cnts:
		(x, y, w, h) = cv2.boundingRect(c)
		ar = w / float(h)

		if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
			questionCnts.append(c)

	return questionCnts

def grade(questionTuple, answer_key):

	answers = []
	correct = []
	questionCnts = questionTuple[1]
	questionCnts = contours.sort_contours(questionCnts, method='top-to-bottom')[0]
	questionImg = questionTuple[0]
	questionImg = cv2.cvtColor(questionImg, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(questionImg, 150, 255, cv2.THRESH_BINARY_INV)[1]

	for (q, i) in enumerate(np.arange(0, len(questionCnts), 4)):
		cnts = contours.sort_contours(questionCnts[i:i + 4])[0]
		bubbled = None
		count = 0
		for (j, c) in enumerate(cnts):
			mask = np.zeros(thresh.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)

			mask = cv2.bitwise_and(thresh, thresh, mask=mask)
			total = cv2.countNonZero(mask)

			if total > 3000:
				count = count + 1
				bubbled = (total, j)
		if(bubbled != None and count == 1):
			ans = chr(bubbled[1]+65)
			answers.append(ans)
		else:
			answers.append('None')

	for i in range(len(answers)):
		if(answers[i] == answer_key[i]):
			correct.append(True)
		else:
			correct.append(False)

	return answers, correct
	
if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--value", type=str, default=100, required=True, help="threshold value")
	args = vars(ap.parse_args())
	# image_name = input('Enter the name of Image: ')
	# key_path = input('Enter the name of the Key: ')
	image_name = 'omr_filled_light.jpg'
	key_path = 'key.txt'
	print('\n')
	answer_key1 = []
	answer_key2 = []
	answer_key = []
	f = open(key_path)
	string = f.read()

	for i in range(30):
		char = str(string[i])
		if char.isupper() and char.isalpha():
			answer_key1.append(char)
	answer_key.append(answer_key1)
	# print('Answer Key (Block 1): '+str(answer_key1)+' :length '+str(len(answer_key1)) )

	for i in range(30):
		char = str(string[30+i])
		if char.isupper() and char.isalpha():
			answer_key2.append(char)
	answer_key.append(answer_key2)
	# print('Answer Key (Block 2): '+str(answer_key2)+' :length '+str(len(answer_key2)) )

	image = cv2.imread(image_name)
	# img_shape = image.shape
	# newshape = (500, int(500*img_shape[0]/float(img_shape[1])) )
	# image = cv2.resize(image, newshape, cv2.INTER_LINEAR)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# cv2.imshow('gray', gray)
	# cv2.waitKey(0)
	cv2.imwrite('1 log_full_gray.png', gray)
	blurred = cv2.GaussianBlur(gray, (9, 9), 0)
	cv2.imwrite('2 log_full_blurred.png', blurred)
	thresh = cv2.threshold(blurred.copy(), int(args["value"]), 255, cv2.THRESH_BINARY)[1]
	cv2.imwrite('3 log_full_thresh.png', thresh)
	# edged = cv2.Canny(thresh, 75, 200)
	# cv2.imwrite('4 log_full_edged.png', edged)

	border = getContours(thresh, False)
	logImage(image, border, (255,255,0), '5 log_full_contours.png')
	paper = four_point_transform(image, border[0].reshape(4, 2))
	warped = four_point_transform(gray, border[0].reshape(4, 2))
	cv2.imwrite('6 log_paper.png', paper)

	image = warped
	thresh = cv2.threshold(image.copy(), 150, 255, cv2.THRESH_BINARY_INV)[1]
	cv2.imwrite('7 log_paper_threshold.png', thresh)

	border = getContours(thresh, True)
	logImage(paper, border, (0,255,255), '8 log_paper_contours.png')
	questionTuples = []
	answers = []
	correct =[]
	total = 0
	for i in range(2): # since we're gonna have two blocks 
		# cropping and appending image to the list

		warped = four_point_transform(paper.copy(), border[i].reshape(4, 2))
		dimen = warped.shape
		x = round(0.03*dimen[0])
		y = round(0.03*dimen[1])
		cropped = warped[x:dimen[0]-x, y:dimen[1]-y]
		qc = detectBubbles(cropped)

		questionTuples.append((cropped, qc))
		answers_, correct_ = grade(questionTuples[i], answer_key[i])
		
		for j in range(15):
			if correct_[j]:
				point = 1
			else:
				point = 0
			print('Answer in the sheet: '+str(answers_[j]))
			print('Correst answer: '+str(answer_key[i][j]))
			print('Point: '+str(point))
			print('\n')
			total = total+point

		# answers.append(answers_)
		# correct.append(correct_)
		logImage(questionTuples[i][0], questionTuples[i][1], (255,0,255), str(i+9)+' log_questions' + str(i) + '.png')

	print('Final Score: '+str(total))
	# print(answers)
	# print(correct)