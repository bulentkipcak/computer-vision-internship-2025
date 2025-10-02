import cv2

img = cv2.imread("inputs/shapes.png", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("inputs/shapes.png bulunamadı.")

hog = cv2.HOGDescriptor()
h = hog.compute(img)

print(f"HOG özellik vektör boyutu: {h.shape}")
