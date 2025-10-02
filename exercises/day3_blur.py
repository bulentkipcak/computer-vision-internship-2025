import cv2

img = cv2.imread("inputs/landscape.jpg")
if img is None:
    raise FileNotFoundError("inputs/landscape.jpg bulunamadı.")

avg = cv2.blur(img, (5,5))
gauss = cv2.GaussianBlur(img, (5,5), 0)

cv2.imwrite("outputs/day3/blur_average.png", avg)
cv2.imwrite("outputs/day3/blur_gaussian.png", gauss)
