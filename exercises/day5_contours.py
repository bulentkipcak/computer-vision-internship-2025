import cv2

img = cv2.imread("inputs/shapes.png")
if img is None:
    raise FileNotFoundError("inputs/shapes.png bulunamadı.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output = img.copy()

for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(output, (x,y), (x+w, y+h), (255,0,0), 2)

    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    center = (int(cx), int(cy))
    cv2.circle(output, center, int(radius), (0,0,255), 2)

    print(f"Kontur {i+1}: Alan={area:.2f}, Çevre={perimeter:.2f}")

cv2.drawContours(output, contours, -1, (0,255,0), 2)
cv2.imwrite("outputs/day5/contours.png", output)
