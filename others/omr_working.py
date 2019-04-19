import cv2
import imutils
from imutils import contours
from imutils.perspective import four_point_transform
	
def findpaper(cnts):
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)

		if len(approx) == 4:
			return approx
	
	print("[INFO] No paper found!")

def findboxes(cnts):
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	i = 0
	contours = []
	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.04*peri, True)

		if len(approx) == 4:
			contours.append(approx)
			i += 1
			if i == 2:
				return contours


imgname = "omr_irl2.jpeg"
image = cv2.imread(imgname)
width = int(image.shape[1]*50/float(100))
height = int(image.shape[0]*50/float(100))
dim = (width, height)
image = cv2.resize(image, dim, interpolation = cv2.INTER_LINEAR)
print("[WORKFLOW] Reading "+imgname+" and resizing..")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("[WORKFLOW] Converting "+imgname+" to grayscale image..")
retval, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
print("[WORKFLOW] Applying threshold and converting to a binary image..")
cv2.imwrite("1 ow_whole_thresh.png", thresh)
print("[WORKFLOW] ow_whole_thresh.png saved.")

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print("[INFO] Number of contours detected: "+str(len(cnts)))
img = image.copy()
cv2.drawContours(img, cnts, -1, (0,255,0), 3)
cv2.imwrite("2 ow_whole_contours.png", img)
print("[WORKFLOW] ow_whole_contours.png saved.")

if len(cnts) > 0:
	docCnt = findpaper(cnts)

print("[INFO] Paper found at position:")
print("      "+str(docCnt[1])+str(docCnt[0])+str(docCnt[3])+str(docCnt[2]))

paper = four_point_transform(image.copy(), docCnt.reshape(4, 2))
width_2 = int(paper.shape[1]*50/float(100))
height_2 = int(paper.shape[0]*50/float(100))
warped = four_point_transform(gray, docCnt.reshape(4, 2))
print("[WORKFLOW] Applying four point transform to crop the paper..")
cv2.imwrite("3 ow_paper_detected.png", warped)
print("[WORKFLOW] ow_paper_detected.png saved.")

retval, thresh_2 = cv2.threshold(warped, 150, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite("4 ow_paper_threshold.png", thresh_2)
print("[WORKFLOW] ow_paper_threshold.png saved.")

cnts_2 = cv2.findContours(thresh_2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts_2 = imutils.grab_contours(cnts_2)
print("[INFO] Number of contours detected: "+str(len(cnts_2)))
img_2 = paper.copy()
cv2.drawContours(img_2, cnts_2, -1, (0,255,0), 3)
cv2.imwrite("5 ow_paper_contours.png", img_2)
print("[WORKFLOW] ow_paper_contours.png saved.")

boxes = findboxes(cnts_2)
img_3 = paper.copy()
cv2.drawContours(img_3, boxes, -1, (255,0,0), 3)
print("[INFO] Number of boxes detected: "+str(len(boxes)))
print("[INFO] Boxes detected at position:")
print("1.\n"+str(boxes[0][0]))
print("2.\n"+str(boxes[1][0]))
cv2.imwrite("6 ow_paper_boxes.png", img_3)
print("[WORKFLOW] ow_paper_boxes.png saved.")

newboxes = []
if boxes[0][0].item(0) > boxes[1][0].item(0):
	newboxes.append(boxes[1])
	newboxes.append(boxes[2])
else:
	newboxes = boxes

seg0 = four_point_transform(paper, newboxes[0].reshape(4, 2))
seg1 = four_point_transform(paper, newboxes[1].reshape(4, 2))

cv2.imwrite("7 ow_seg0.png", seg0)
print("[WORKFLOW] 7 ow_seg0.png saved.")
cv2.imwrite("8 ow_seg1.png", seg1)
print("[WORKFLOW] 8 ow_seg1.png saved.")
