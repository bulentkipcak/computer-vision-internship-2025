import cv2

images = ["inputs/landscape.jpg", "inputs/shapes.png"]
for path in images:
    img = cv2.imread(path)
    if img is None:
        print(f"{path} bulunamadı.")
        continue
    h, w, c = img.shape
    print(f"{path}: {w}x{h}, {c} kanal")
