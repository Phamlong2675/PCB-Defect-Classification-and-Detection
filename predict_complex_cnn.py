import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# Configuration 
DATA_DIR = Path("Data/processed/classification")
MODEL_FILE = Path("runs/complex_cnn/complex_cnn_model.h5")   
BATCH_SIZE = 64

tf.get_logger().setLevel("ERROR")

# Class names in the same order as during training
manifest = pd.read_csv(DATA_DIR / "classification_manifest.csv")
class_names = manifest.drop_duplicates("class_index")
class_names = class_names.sort_values("class_index")["class_name"].tolist()
num_classes = len(class_names)

def read_image(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=1, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.ensure_shape(image, (224, 224, 1))
    return image

# Load the trained model
model = tf.keras.models.load_model(str(MODEL_FILE), compile=False)
print("Loaded model:", MODEL_FILE)

def predict(paths):
    dataset = tf.data.Dataset.from_tensor_slices(paths).map(read_image).batch(BATCH_SIZE)
    logits = model.predict(dataset, verbose=0)
    probabilities = tf.nn.softmax(logits, axis=1).numpy()
    return np.argmax(probabilities, axis=1), np.max(probabilities, axis=1)

if len(sys.argv) > 1:
    predictions, confidence = predict(sys.argv[1:])
    for path, label, conf in zip(sys.argv[1:], predictions, confidence):
        print(f"{path}: {class_names[label]} ({conf:.1%})")
    sys.exit()

rows = manifest[manifest["split"] == "test"].reset_index(drop=True)
paths = [str(DATA_DIR / path) for path in rows["crop_path"]]
true_labels = rows["class_index"].to_numpy()

start = time.time()
predictions, confidence = predict(paths)
print(f"Predicted {len(paths)} test images in {time.time() - start:.0f}s")

examples = pd.DataFrame({
    "image": rows["crop_name"],
    "true": [class_names[i] for i in true_labels],
    "predicted": [class_names[i] for i in predictions],
    "confidence": np.round(confidence, 4)})
print(examples.head(10).to_string(index=False))

# Evaluate the predictions
report = classification_report(
    true_labels, predictions,
    labels=list(range(num_classes)), target_names=class_names,
    output_dict=True, zero_division=0)
print("\nTEST RESULT: Complex CNN")
print(classification_report(
    true_labels, predictions,
    labels=list(range(num_classes)), target_names=class_names,
    digits=4, zero_division=0))
print("Confusion matrix (rows = true, columns = predicted):")
print(pd.DataFrame(confusion_matrix(true_labels, predictions, labels=list(range(num_classes))),
                   index=class_names, columns=class_names))
print(f"\nAccuracy: {report['accuracy']:.4f} | Precision: {report['macro avg']['precision']:.4f} | "
      f"Recall: {report['macro avg']['recall']:.4f} | F1-score: {report['macro avg']['f1-score']:.4f} (macro)")
print("Wrong predictions:", int(np.sum(predictions != true_labels)), "of", len(true_labels))