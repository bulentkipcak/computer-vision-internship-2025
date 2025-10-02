import cv2, os

img = cv2.imread("inputs/landscape.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("inputs/landscape.jpg bulunamadı.")

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)
sobel = cv2.convertScaleAbs(sobel)

canny = cv2.Canny(img, 100, 200)

cv2.imwrite("outputs/day3/edges_sobel.png", sobel)
cv2.imwrite("outputs/day3/edges_canny.png", canny)
