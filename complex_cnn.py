import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score

# Configuration
DATA_DIR = Path("Data/processed/classification")
OUT_DIR = Path("runs/complex_cnn")
BATCH_SIZE = 64
EPOCHS = 20
PATIENCE = 8
LEARNING_RATE = 1e-3
SEED = 42

OUT_DIR.mkdir(parents=True, exist_ok=True)
tf.keras.utils.set_random_seed(SEED)   
tf.get_logger().setLevel("ERROR")

# Load data
manifest = pd.read_csv(DATA_DIR / "classification_manifest.csv")
class_names = manifest.drop_duplicates("class_index").sort_values("class_index")["class_name"].tolist()
num_classes = len(class_names)
train_rows = manifest[manifest["split"] == "train"].reset_index(drop=True)
val_rows = manifest[manifest["split"] == "valid"].reset_index(drop=True)
test_rows = manifest[manifest["split"] == "test"].reset_index(drop=True)
print("Images per split:", manifest["split"].value_counts().to_dict())

def read_image(path, label):
    image = tf.io.decode_image(tf.io.read_file(path), channels=1, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)   # pixel values 0..255 -> 0..1
    return tf.ensure_shape(image, (224, 224, 1)), tf.one_hot(label, num_classes)

def make_dataset(rows, shuffle=False):
    paths = [str(DATA_DIR / path) for path in rows["crop_path"]]
    dataset = tf.data.Dataset.from_tensor_slices((paths, rows["class_index"].to_numpy(dtype=np.int32)))
    if shuffle:
        dataset = dataset.shuffle(len(rows), seed=SEED)
    # read the images in parallel -> group into batches -> prepare the next batch during training
    return dataset.map(read_image, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(train_rows, shuffle=True)
val_ds = make_dataset(val_rows)
test_ds = make_dataset(test_rows)

# Mean and variance of the training pixels, used to normalize every image
normalizer = tf.keras.layers.Normalization(axis=None)
normalizer.adapt(train_ds.map(lambda image, label: image))

# Augmentation
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomTranslation(0.05, 0.05)])

# Build the Complex CNN
def conv_block(x, filters, kernel=3, stride=1):
    x = tf.keras.layers.Conv2D(filters, kernel, strides=stride, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU()(x)

def se_block(x, channels):
    # Squeeze-and-Excitation
    weights = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(x)
    weights = tf.keras.layers.Conv2D(max(channels // 16, 8), 1, activation="relu")(weights)
    weights = tf.keras.layers.Conv2D(channels, 1, activation="sigmoid")(weights)
    return tf.keras.layers.Multiply()([x, weights])

def multi_branch_block(x, channels):
    filters = channels // 4
    branch1 = conv_block(x, filters, kernel=1)

    branch2 = conv_block(x, filters, kernel=1)
    branch2 = conv_block(branch2, filters)

    branch3 = conv_block(x, filters, kernel=1)
    branch3 = conv_block(branch3, filters)
    branch3 = conv_block(branch3, filters)

    branch4 = tf.keras.layers.MaxPooling2D(3, strides=1, padding="same")(x)
    branch4 = conv_block(branch4, filters, kernel=1)

    merged = tf.keras.layers.Concatenate()([branch1, branch2, branch3, branch4])
    merged = tf.keras.layers.Conv2D(channels, 1, use_bias=False)(merged)
    merged = tf.keras.layers.BatchNormalization(gamma_initializer="zeros")(merged)
    merged = se_block(merged, channels)
    x = tf.keras.layers.Add()([x, merged])
    return tf.keras.layers.ReLU()(x)

def reduction_block(x, channels):
    branch1 = conv_block(x, channels // 2, stride=2)

    branch2 = conv_block(x, channels // 4, kernel=1)
    branch2 = tf.keras.layers.MaxPooling2D(pool_size=1, strides=2)(branch2)
    branch2 = conv_block(branch2, channels // 4)

    branch3 = tf.keras.layers.MaxPooling2D(2)(x)
    branch3 = conv_block(branch3, channels // 4, kernel=1)
    return tf.keras.layers.Concatenate()([branch1, branch2, branch3])

inputs = tf.keras.Input(shape=(224, 224, 1))
x = normalizer(augmentation(inputs))

# Stem: 224x224 -> 112x112
x = conv_block(x, 32, kernel=5, stride=2)
x = conv_block(x, 64)

# Stage 1: 128 channels, 56x56, 2 MultiBranch blocks
x = reduction_block(x, 128)
x = multi_branch_block(x, 128)
x = multi_branch_block(x, 128)

# Stage 2: 256 channels, 28x28, 3 MultiBranch blocks
x = reduction_block(x, 256)
x = multi_branch_block(x, 256)
x = multi_branch_block(x, 256)
x = multi_branch_block(x, 256)

# Stage 3: 384 channels, 14x14, 2 MultiBranch blocks
x = reduction_block(x, 384)
x = multi_branch_block(x, 384)
x = multi_branch_block(x, 384)

# Head: average pooling + max pooling -> Dropout -> Dense -> Dropout -> Dense
avg = tf.keras.layers.GlobalAveragePooling2D()(x)
maximum = tf.keras.layers.GlobalMaxPooling2D()(x)
x = tf.keras.layers.Concatenate()([avg, maximum])
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.25)(x)
outputs = tf.keras.layers.Dense(num_classes)(x)
model = tf.keras.Model(inputs, outputs)

if hasattr(tf.keras.optimizers, "AdamW"):   
    AdamW = tf.keras.optimizers.AdamW
else:                                       
    AdamW = tf.keras.optimizers.experimental.AdamW
schedule = tf.keras.optimizers.schedules.CosineDecay(LEARNING_RATE, EPOCHS * len(train_ds))
model.compile(optimizer=AdamW(learning_rate=schedule, weight_decay=1e-4),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.05),
              metrics=["accuracy"])

history_rows = []
best_f1, best_epoch, wait = -1.0, 0, 0
for epoch in range(1, EPOCHS + 1):
    print(f"Epoch {epoch}/{EPOCHS}")
    result = model.fit(train_ds, validation_data=val_ds, epochs=1)

    val_pred = model.predict(val_ds, verbose=0).argmax(axis=1)
    val_f1 = f1_score(val_rows["class_index"], val_pred, average="macro", zero_division=0)
    print(f"Val Macro-F1: {val_f1:.4f}")
    history_rows.append({
        "epoch": epoch,
        "train_loss": result.history["loss"][0],
        "train_acc": result.history["accuracy"][0],
        "val_loss": result.history["val_loss"][0],
        "val_acc": result.history["val_accuracy"][0],
        "val_f1": val_f1})

    if val_f1 > best_f1:
        best_f1, best_epoch, wait = val_f1, epoch, 0
        best_weights = model.get_weights()
    else:
        wait += 1
        if wait >= PATIENCE:
            print("Early stopping: Macro-F1 has not improved.")
            break

model.set_weights(best_weights)
model.save(str(OUT_DIR / "complex_cnn_model.h5"), include_optimizer=False)
history = pd.DataFrame(history_rows)
history.to_csv(OUT_DIR / "history.csv", index=False)

# Evaluate the best model on the validation and test sets
results = {"model": "complex_cnn", "img_size": 224, "parameters": model.count_params(), "best_epoch": best_epoch}
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, split, dataset, rows in [(axes[0], "valid", val_ds, val_rows), (axes[1], "test", test_ds, test_rows)]:
    probabilities = tf.nn.softmax(model.predict(dataset, verbose=0), axis=1).numpy()
    predictions = probabilities.argmax(axis=1)
    true_labels = rows["class_index"].to_numpy()

    print("\nResult", split.upper())
    print(classification_report(true_labels, predictions, target_names=class_names,
                                digits=4, zero_division=0))
    report = classification_report(true_labels, predictions, target_names=class_names,
                                   output_dict=True, zero_division=0)
    results[split] = {
        "accuracy": report["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"]}

    ConfusionMatrixDisplay.from_predictions(true_labels, predictions, display_labels=class_names,
                                            ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title(split)

    if split == "test":
        output = rows.copy()
        output["predicted"] = [class_names[i] for i in predictions]
        output["confidence"] = probabilities.max(axis=1)
        output["correct"] = predictions == true_labels
        output.to_csv(OUT_DIR / "predictions_test.csv", index=False)

fig.tight_layout()
fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=130)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
history.plot(x="epoch", y=["train_loss", "val_loss"], ax=axes[0], title="Loss")
history.plot(x="epoch", y=["train_acc", "val_acc"], ax=axes[1], title="Accuracy")
history.plot(x="epoch", y="val_f1", ax=axes[2], title="Macro-F1")
for ax in axes:
    ax.axvline(best_epoch, linestyle="--", color="gray")
fig.tight_layout()
fig.savefig(OUT_DIR / "learning_curves.png", dpi=130)

with open(OUT_DIR / "results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Best epoch: {best_epoch} | Val Macro-F1: {best_f1:.4f}")
print("Results saved to:", OUT_DIR)