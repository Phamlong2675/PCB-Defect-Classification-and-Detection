import json
import sys
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = Path("Data/processed/classification")
OUT_DIR = Path("runs/simple_cnn")
BATCH_SIZE = 64
IMG_SIZE = (224, 224)

tf.get_logger().setLevel("ERROR")

manifest = pd.read_csv(DATA_DIR / "classification_manifest.csv")
class_names = json.loads((OUT_DIR / "class_names.json").read_text("utf-8"))["class_names"]
model = tf.keras.models.load_model(str(OUT_DIR / "simple_cnn_model.h5"), compile=False)


def read_image(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=1, expand_animations=False)
    image.set_shape([None, None, 1])
    image = tf.image.resize(image, IMG_SIZE) 
    return image


def predict(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths).map(read_image).batch(BATCH_SIZE)
    probs = model.predict(ds, verbose=0)
    return probs.argmax(axis=1), probs.max(axis=1)


# predict ảnh truyền vào
if len(sys.argv) > 1:
    paths = sys.argv[1:]
    for path, label, conf in zip(paths, *predict(paths)):
        print(f"{path}: {class_names[label]} ({conf:.1%})")
    sys.exit()

test = manifest[manifest["split"] == "test"].reset_index(drop=True)
name_to_idx = {n: i for i, n in enumerate(class_names)}
y_true = test["class_name"].map(name_to_idx).to_numpy()
y_pred, conf = predict([str(DATA_DIR / p) for p in test["crop_path"]])

samples = pd.DataFrame({
    "image": test["crop_name"],
    "true": [class_names[i] for i in y_true],
    "predicted": [class_names[i] for i in y_pred],
    "confidence": conf.round(4),
})
print(samples.head(10).to_string(index=False))

labels = list(range(len(class_names)))
kw = dict(labels=labels, target_names=class_names, zero_division=0)
report = classification_report(y_true, y_pred, output_dict=True, **kw)
report_str = classification_report(y_true, y_pred, digits=4, **kw)
print("\n" + report_str)

cm = confusion_matrix(y_true, y_pred, labels=labels)
print(pd.DataFrame(cm, index=class_names, columns=class_names))

acc = report["accuracy"]
macro = report["macro avg"]
print(f"\nAccuracy: {acc:.4f} | Precision: {macro['precision']:.4f} | "
      f"Recall: {macro['recall']:.4f} | F1: {macro['f1-score']:.4f}")
print(f"Wrong predictions: {np.sum(y_pred != y_true)} of {len(y_true)}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
results_path = OUT_DIR / "results.json"
results = json.loads(results_path.read_text("utf-8")) if results_path.exists() else {}
results["test"] = {
    "accuracy": acc,
    "macro_precision": macro["precision"],
    "macro_recall": macro["recall"],
    "macro_f1": macro["f1-score"],
}
results_path.write_text(json.dumps(results, indent=2), "utf-8")

cm_pct = confusion_matrix(y_true, y_pred, labels=labels, normalize="true") * 100
fig, (ax_cm, ax_txt) = plt.subplots(1, 2, figsize=(16, 7))

sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax_cm)
ax_cm.set(xlabel="Predicted", ylabel="Actual", title="Confusion Matrix (%)")

ax_txt.axis("off")
ax_txt.text(0, 1, report_str, family="monospace", fontsize=9, va="top")
ax_txt.set_title("Metrics Report")

fig.suptitle(f"Simple CNN — Test Accuracy: {acc:.2%}", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT_DIR / "test_result.png", dpi=150)
plt.show()