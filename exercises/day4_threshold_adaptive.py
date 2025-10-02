import cv2

img = cv2.imread("inputs/landscape.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("inputs/landscape.jpg bulunamadı.")

adaptive = cv2.adaptiveThreshold(img, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY,
                                 11, 2)

cv2.imwrite("outputs/day4/threshold_adaptive.png", adaptive)
