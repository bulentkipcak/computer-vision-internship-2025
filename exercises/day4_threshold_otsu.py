import cv2

img = cv2.imread("inputs/landscape.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("inputs/landscape.jpg bulunamadı.")

_, otsu = cv2.threshold(img, 0, 255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imwrite("outputs/day4/threshold_otsu.png", otsu)
