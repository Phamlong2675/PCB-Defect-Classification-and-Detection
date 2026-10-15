import tensorflow as tf
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = Path("Data/processed/classification")
OUT_DIR = Path("runs/simple_cnn")
IMG_SIZE = (224, 224)
BATCH_SIZE = 64

tf.get_logger().setLevel("ERROR")

manifest = pd.read_csv(DATA_DIR / "classification_manifest.csv")
class_names = json.loads(
    (OUT_DIR / "class_names.json").read_text(encoding="utf-8"))["class_names"]
model = tf.keras.models.load_model(
    OUT_DIR / "simple_cnn_model.h5", compile=False)

# Load testset
test = manifest[manifest["split"] == "test"].reset_index(drop=True)
paths = [str(DATA_DIR / p) for p in test["crop_path"]]

def load_image(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=1, expand_animations=False)
    image.set_shape([None, None, 1])
    return tf.image.resize(image, IMG_SIZE)

test_ds = tf.data.Dataset.from_tensor_slices(paths)
test_ds = test_ds.map(load_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Predict
probs = model.predict(test_ds, verbose=0)
y_pred = probs.argmax(axis=1)
y_true = test["class_name"].map({name: i for i, name in enumerate(class_names)}).to_numpy()

print("Results simple CNN:")
print(classification_report(
    y_true, y_pred, labels=range(len(class_names)),
    target_names=class_names, digits=4, zero_division=0
))

print("Confusion Matrix:")
print(pd.DataFrame(
    confusion_matrix(y_true, y_pred, labels=range(len(class_names))),
    index=class_names, columns=class_names
))

report = classification_report(
    y_true, y_pred, labels=range(len(class_names)),
    target_names=class_names, output_dict=True, zero_division=0
)
macro = report["macro avg"]

print(f"\nAccuracy:  {report['accuracy']:.4f}")
print(f"Precision: {macro['precision']:.4f}")
print(f"Recall:    {macro['recall']:.4f}")
print(f"F1-score:  {macro['f1-score']:.4f}")
print(f"Wrong predictions: {(y_pred != y_true).sum()} / {len(y_true)}")
