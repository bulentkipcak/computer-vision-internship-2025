import cv2, os
os.makedirs("outputs/day6", exist_ok=True)

img = cv2.imread("inputs/landscape.jpg")
if img is None:
    raise FileNotFoundError("inputs/landscape.jpg bulunamadı.")

flip = cv2.flip(img, 1)
rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
bright = cv2.convertScaleAbs(img, alpha=1.2, beta=30)

cv2.imwrite("outputs/day6/flip.jpg", flip)
cv2.imwrite("outputs/day6/rotated.jpg", rotated)
cv2.imwrite("outputs/day6/bright.jpg", bright)
