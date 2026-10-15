from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

DATA_DIR = Path("Data/processed/classification")
OUT_DIR = Path("runs/simple_cnn")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE, BATCH_SIZE, EPOCHS = (224, 224), 64, 40
BEST_MODEL = OUT_DIR / "best_model.keras"

def load_data(split, shuffle = False):
    return tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / split, image_size=IMG_SIZE, batch_size=BATCH_SIZE, 
        label_mode="int", color_mode="grayscale", shuffle=shuffle)

train_ds = load_data("train", shuffle=True)
val_ds = load_data("valid")
test_ds = load_data("test")

# lấy class_names lấy ngay từ dataset để chắc chắn khớp thứ tự label
class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print("Classes:", class_names)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

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

inputs = layers.Input(shape=(*IMG_SIZE, 1))
x = augment(layers.Rescaling(1.0 / 255)(inputs))

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
x = layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs, name="simple_cnn")
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# train 
callbacks = [
    EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6, verbose=1),
    ModelCheckpoint(BEST_MODEL, monitor="val_loss", save_best_only=True, verbose=1),
]

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# save model tốt nhất
model = tf.keras.models.load_model(BEST_MODEL)
model.save(OUT_DIR / "simple_cnn_model.h5")



