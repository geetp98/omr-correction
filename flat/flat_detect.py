import cv2
import imutils
from imutils import contours
from imutils.perspective import four_point_transform

def detectsquare(c):
	shape = "None"
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.04*peri, True)

	if len(approx == 4):
		(x,y,w,h) = cv2.boundingRect(approx)
		aspect = w/float(h)

		if(aspect>=0.90 and aspect<=1.10):
			shape = "square"
			position = [x,y,h,w]
			# print("[INFO] Square appears at: "+str(position))
			return position

image = cv2.imread("omr_flat_new.png")
print("[WORKFLOW] Reading omr_flat.py and resizing..")
# cv2.imshow("Sample", image)
# cv2.waitKey(0)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("[WORKFLOW] Converting omr_sample.py to grayscale image..")
# cv2.imshow("Gray Image", gray)
# cv2.waitKey(0)
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# cv2.imshow("Blurred Image", blurred)
# cv2.waitKey(0)
retval, thresh = cv2.threshold(gray.copy(), 220, 255, cv2.THRESH_BINARY_INV)
print("[WORKFLOW] Applying threshold and converting to a binary image..")
cv2.imwrite("fd_threshold.png", thresh)
print("[WORKFLOW] fd_threshold.png saved.")
# thresh_inv = cv2.bitwise_not(thresh)
# cv2.imshow("Threshold Image", thresh)
# cv2.waitKey(0)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print("[INFO] Number of contours detected: "+str(len(cnts)))

img = image.copy()
cv2.drawContours(img, cnts, -1, (0,255,0), 3)
# cv2.imshow("Contours", img)
# cv2.waitKey(0)
cv2.imwrite("fd_countours.png", img)
print("[WORKFLOW] fd_contours.png saved.")

print("[WORKFLOW] Starting to look for squares..")
i = 0
img_2 = image.copy()
pos = []
for c in cnts:
	pos1 = detectsquare(c)
	if(pos1 != None):
		cv2.rectangle(img_2, (pos1[0],pos1[1]), (pos1[0]+pos1[2],pos1[1]+pos1[3]), (255,0,0), 3)
		pos.append(pos1)

print("[INFO] Squares detected at positions:")
print("       "+str(pos))

cv2.imwrite("fd_squares.png", img_2)
print("[WORKFLOW] fd_squares.png saved.")
