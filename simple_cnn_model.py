import json
import csv
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

DATA_DIR = Path("Data/processed/classification")
OUT_DIR = Path("runs/simple_cnn")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 40
BEST_MODEL = OUT_DIR / "best_model.keras"

def load_split(name, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / name,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        color_mode="grayscale",
        shuffle=shuffle,
    )

train_ds = load_split("train", shuffle=True)
val_ds = load_split("valid", shuffle=False)
test_ds = load_split("test", shuffle=False)

# lấy class_names lấy ngay từ dataset để chắc chắn khớp thứ tự label
class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# model simple CNN 
augment = tf.keras.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(0.05, 0.05),
], name="augment")

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    return layers.ReLU()(x)

def build_model():
    inputs = layers.Input(shape=(*IMG_SIZE, 1))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = augment(x)

    x = conv_block(x, 32)
    x = layers.MaxPooling2D()(x)   # 224 -> 112
    x = conv_block(x, 64)
    x = layers.MaxPooling2D()(x)   # 112 -> 56
    x = conv_block(x, 64)
    x = layers.MaxPooling2D()(x)   # 56 -> 28
    x = conv_block(x, 128)
    x = layers.MaxPooling2D()(x)   # 28 -> 14

    x = layers.Concatenate()([
        layers.GlobalAveragePooling2D()(x),
        layers.GlobalMaxPooling2D()(x),
    ])
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="simple_cnn")

model = build_model()
model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# train model 
callbacks = [
    EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6, verbose=1),
    ModelCheckpoint(BEST_MODEL, monitor="val_loss", save_best_only=True, verbose=1),
]

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# save model tốt nhất
model = tf.keras.models.load_model(BEST_MODEL)
model.save(OUT_DIR / "simple_cnn_model.h5")

# class 6 loại lỗi .json 
with open(OUT_DIR / "class_names.json", "w", encoding="utf-8") as f:
    json.dump({"class_names": class_names, "img_size": list(IMG_SIZE)}, f,
              ensure_ascii=False, indent=2)

# kết quả train model 
train_loss, train_acc = model.evaluate(train_ds, verbose=0)
val_loss, val_acc = model.evaluate(val_ds, verbose=0)
print(f"\nTrain acc: {train_acc:.2%} | Val acc: {val_acc:.2%}")

import csv
keys = list(history.history.keys())
with open(OUT_DIR / "history.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch"] + keys)
    for i in range(len(history.history["loss"])):
        writer.writerow([i + 1] + [history.history[k][i] for k in keys])

def ev(ds):
    loss, acc = model.evaluate(ds, verbose=0)
    return {"loss": float(loss), "accuracy": float(acc)}

results = {
    "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
    "batch_size": BATCH_SIZE,
    "img_size": list(IMG_SIZE),
    "train": ev(load_split("train", shuffle=False)),
    "val": ev(val_ds),
}
(OUT_DIR / "results.json").write_text(json.dumps(results, indent=2), "utf-8")

# đường accuracy / loss 
plt.figure(figsize=(12, 4))
 
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()
 
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Val")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()
 
plt.savefig(OUT_DIR / "training_curves.png", dpi=200)
plt.close()
