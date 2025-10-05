import os, cv2, numpy as np, random

random.seed(42)
os.makedirs("dataset/detect/train/images", exist_ok=True)
os.makedirs("dataset/detect/train/labels", exist_ok=True)
os.makedirs("dataset/detect/val/images",   exist_ok=True)
os.makedirs("dataset/detect/val/labels",   exist_ok=True)

W = H = 640
classes = {"square":0, "circle":1, "triangle":2}

def rand_color():
    return int(random.randint(80,255))

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def yolo_line(cls_id, cx, cy, w, h):
    return f"{cls_id} {cx/W:.6f} {cy/H:.6f} {w/W:.6f} {h/H:.6f}\n"

def draw_square(img):
    size = random.randint(60,160)
    x = random.randint(0, W - size)
    y = random.randint(0, H - size)
    c = rand_color()
    cv2.rectangle(img, (x,y), (x+size, y+size), (c,c,c), -1)
    cx, cy = x + size/2, y + size/2
    return classes["square"], cx, cy, size, size

def draw_circle(img):
    r = random.randint(30,100)
    cx = random.randint(r, W - r)
    cy = random.randint(r, H - r)
    c = rand_color()
    cv2.circle(img, (cx,cy), r, (c,c,c), -1)
    return classes["circle"], cx, cy, 2*r, 2*r

def draw_triangle(img):
    s = random.randint(70,180)
    x = random.randint(0, W - s)
    y = random.randint(0, H - s)
    p1 = (x, y+s)
    p2 = (x+s, y+s)
    p3 = (x+s//2, y)
    pts = np.array([p1,p2,p3], dtype=np.int32)
    c = rand_color()
    cv2.fillPoly(img, [pts], (c,c,c))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    cx = min(xs) + w/2
    cy = min(ys) + h/2
    return classes["triangle"], cx, cy, w, h

def draw_one(img):
    fn = random.choice([draw_square, draw_circle, draw_triangle])
    return fn(img)

def add_noise_affine(img):
    M = np.float32([[1,0,random.randint(-15,15)],[0,1,random.randint(-15,15)]])
    img = cv2.warpAffine(img, M, (W,H), borderMode=cv2.BORDER_REFLECT101)
    g = np.random.normal(0, 6, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + g, 0, 255).astype(np.uint8)
    return img

def make_split(split, n_images):
    for i in range(n_images):
        img = np.zeros((H,W,3), dtype=np.uint8) + 20
        n_objs = random.randint(1,3)
        labels = []
        for _ in range(n_objs):
            cls_id, cx, cy, bw, bh = draw_one(img)
            cx = clamp(cx, 0, W); cy = clamp(cy, 0, H)
            bw = clamp(bw, 4, W);  bh = clamp(bh, 4, H)
            labels.append((cls_id, cx, cy, bw, bh))
        if split == "train":
            img = add_noise_affine(img)
        img_name = f"{split}_{i:05d}.jpg"
        lbl_name = f"{split}_{i:05d}.txt"
        cv2.imwrite(f"dataset/detect/{split}/images/{img_name}", img)
        with open(f"dataset/detect/{split}/labels/{lbl_name}", "w") as f:
            for cls_id, cx, cy, bw, bh in labels:
                f.write(yolo_line(cls_id, cx, cy, bw, bh))

make_split("train", 1200)
make_split("val",   300)
print("Detection dataset generated.")
