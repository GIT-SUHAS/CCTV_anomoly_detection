import cv2

img = cv2.imread("CCTV_anomoly_detection/dataset/train/001.tif")
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()