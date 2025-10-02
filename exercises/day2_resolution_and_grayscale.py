import cv2, time, os

img = cv2.imread("inputs/landscape.jpg")
if img is None:
    raise FileNotFoundError("inputs/landscape.jpg bulunamadı.")

resolutions = [(640,480), (1280,720), (1920,1080)]
timings = []

for w,h in resolutions:
    resized = cv2.resize(img, (w,h))
    t0 = time.time()
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    dt = (time.time() - t0)*1000
    timings.append((w,h,dt))
    cv2.imwrite(f"outputs/day2/gray_{w}x{h}.png", gray)
    print(f"{w}x{h} -> {dt:.2f} ms")

with open("outputs/day2/timings_day2.csv","w") as f:
    f.write("width,height,elapsed_ms\n")
    for w,h,t in timings:
        f.write(f"{w},{h},{t:.2f}\n")
