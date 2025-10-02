import cv2, matplotlib.pyplot as plt, os

img = cv2.imread("inputs/landscape.jpg")
if img is None:
    raise FileNotFoundError("inputs/landscape.jpg bulunamadı.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.hist(gray.ravel(), bins=256, range=[0,256])
plt.savefig("outputs/day2/histogram_landscape.png")
