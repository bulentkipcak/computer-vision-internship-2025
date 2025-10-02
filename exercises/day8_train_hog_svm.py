import cv2, os, numpy as np
from glob import glob
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import joblib

def iter_images(root, classes):
    X, y, paths = [], [], []
    hog = cv2.HOGDescriptor()
    for label, cls in enumerate(classes):
        for p in sorted(glob(os.path.join(root, "images", cls, "*.png"))):
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            h = hog.compute(img).reshape(-1)
            X.append(h)
            y.append(label)
            paths.append(p)
    return np.array(X), np.array(y), paths

def save_sample_preds(model, classes, paths, X, y, out_png):
    idx = np.random.choice(len(X), size=min(16, len(X)), replace=False)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for ax, i in zip(axes.ravel(), idx):
        p = model.predict(X[i].reshape(1, -1))[0]
        img = cv2.imread(paths[i])
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"P:{classes[p]}  T:{classes[y[i]]}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png)

def main():
    classes = ["square", "circle", "triangle"]
    Xtr, ytr, _ = iter_images("dataset/train", classes)
    Xva, yva, paths_val = iter_images("dataset/val", classes)

    model = make_pipeline(StandardScaler(with_mean=False), LinearSVC(max_iter=10000))
    model.fit(Xtr, ytr)
    ypred = model.predict(Xva)

    report = classification_report(yva, ypred, target_names=classes, digits=4)
    with open("outputs/day8/report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(yva, ypred)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot(values_format="d")
    plt.tight_layout()
    plt.savefig("outputs/day8/confusion_matrix.png")

    save_sample_preds(model, classes, paths_val, Xva, yva, "outputs/day8/sample_predictions.png")
    joblib.dump(model, "outputs/day8/hog_svm_model.joblib")

if __name__ == "__main__":
    os.makedirs("outputs/day8", exist_ok=True)
    main()
