import cv2, numpy as np, os
img = np.zeros((512, 512, 3), dtype=np.uint8)
cv2.rectangle(img, (60,60), (220,220), (255,255,255), -1)
cv2.circle(img, (350,160), 80, (255,255,255), -1)
pts = np.array([[300,360],[420,460],[180,460]], np.int32)
cv2.fillPoly(img, [pts], (255,255,255))
cv2.imwrite("inputs/shapes.png", img)