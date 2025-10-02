import sys, cv2, numpy as np, os

print("Python:", sys.version)
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)

img = cv2.imread("inputs/landscape.jpg")
if img is None:
    raise FileNotFoundError("inputs/landscape.jpg bulunamadı.")

print("Input shape:", img.shape, "dtype:", img.dtype)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("outputs/day2/gray_check.png", gray)
