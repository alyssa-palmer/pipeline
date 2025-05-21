import cv2

image = cv2.imread(r"output\frames\frame_0000.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("grayscale", gray)
# cv2.waitKey(0)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply thresholding (tune this!)
_, thresh = cv2.threshold(blurred, 75, 255, cv2.THRESH_BINARY)
# thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,21,2)
cv2.imshow("thresholded", thresh)
cv2.waitKey(0)