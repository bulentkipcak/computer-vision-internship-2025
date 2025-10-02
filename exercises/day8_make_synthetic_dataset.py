import cv2, numpy as np, os, random

def make_canvas():
    return np.zeros((128, 128, 3), dtype=np.uint8)

def draw_square(img):
    s = random.randint(30, 70)
    x = random.randint(10, 128 - s - 10)
    y = random.randint(10, 128 - s - 10)
    angle = random.choice([0, 15, 30, 45])
    rect = ((x + s//2, y + s//2), (s, s), angle)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(img, [box], (255, 255, 255))

def draw_circle(img):
    r = random.randint(15, 40)
    cx = random.randint(20, 108)
    cy = random.randint(20, 108)
    cv2.circle(img, (cx, cy), r, (255, 255, 255), -1)

def draw_triangle(img):
    r = random.randint(25, 55)
    cx = random.randint(30, 98)
    cy = random.randint(30, 98)
    pts = np.array([
        [cx, cy - r],
        [cx - int(0.866*r), cy + int(0.5*r)],
        [cx + int(0.866*r), cy + int(0.5*r)]
    ], dtype=np.int32)
    M = cv2.getRotationMatrix2D((cx, cy), random.choice([0, 20, 40, 60]), 1.0)
    pts = cv2.transform(pts[None, :, :], M)[0].astype(np.int32)
    cv2.fillPoly(img, [pts], (255, 255, 255))

def jitter(img):
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    alpha = random.uniform(0.9, 1.2)
    beta = random.randint(-15, 25)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img

def save_sample(path, draw_fn, n):
    os.makedirs(path, exist_ok=True)
    for i in range(n):
        img = make_canvas()
        draw_fn(img)
        img = jitter(img)
        cv2.imwrite(os.path.join(path, f"{i:04d}.png"), img)

def main():
    save_sample("dataset/train/images/square", draw_square, 200)
    save_sample("dataset/train/images/circle", draw_circle, 200)
    save_sample("dataset/train/images/triangle", draw_triangle, 200)
    save_sample("dataset/val/images/square", draw_square, 60)
    save_sample("dataset/val/images/circle", draw_circle, 60)
    save_sample("dataset/val/images/triangle", draw_triangle, 60)

if __name__ == "__main__":
    main()
