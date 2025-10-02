import cv2, numpy as np

img = cv2.imread("inputs/landscape.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("inputs/landscape.jpg bulunamadı.")

hist = cv2.calcHist([img], [0], None, [256], [0,256])
np.save("outputs/day7/histogram.npy", hist)

import matplotlib.pyplot as plt
plt.plot(hist)
plt.title("Grayscale Histogram")
plt.savefig("outputs/day7/histogram.png")
