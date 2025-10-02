import cv2

img = cv2.imread("inputs/landscape.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("inputs/landscape.jpg bulunamadı.")

_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite("outputs/day4/threshold_basic.png", binary)
