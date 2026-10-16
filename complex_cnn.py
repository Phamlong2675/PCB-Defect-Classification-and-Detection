from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

DATA_DIR = Path("Data/processed/classification")
OUT_DIR = Path("runs/complex_cnn")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE, BATCH_SIZE, EPOCHS = (224, 224), 64, 20
BEST_MODEL = OUT_DIR / "best_model.keras"

tf.keras.utils.set_random_seed(42)

def load_data(split, shuffle=False):
    return tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / split, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="categorical", color_mode="grayscale", shuffle=shuffle)

train_ds = load_data("train", shuffle=True)
val_ds = load_data("valid")

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print("Classes:", class_names)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# model Complex CNN
augment = tf.keras.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(0.05, 0.05),
], name="augment")

normalizer = layers.Normalization(axis=None)
normalizer.adapt(train_ds.map(lambda x, y: x / 255.0))

def conv_block(x, filters, kernel=3, stride=1):
    x = layers.Conv2D(filters, kernel, strides=stride, padding="same",
                      use_bias=False, kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)

def se_block(x, channels):
    weights = layers.GlobalAveragePooling2D(keepdims=True)(x)
    weights = layers.Conv2D(max(channels // 16, 8), 1, activation="relu")(weights)
    weights = layers.Conv2D(channels, 1, activation="sigmoid")(weights)
    return layers.Multiply()([x, weights])

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

inputs = layers.Input(shape=(*IMG_SIZE, 1))
x = normalizer(augment(layers.Rescaling(1.0 / 255)(inputs)))

# Stem: 224 -> 112
x = conv_block(x, 32, kernel=5, stride=2)
x = conv_block(x, 64)

# Stage 1: 112 -> 56
x = reduction_block(x, 128)
x = multi_branch_block(x, 128)
x = multi_branch_block(x, 128)

# Stage 2: 56 -> 28
x = reduction_block(x, 256)
x = multi_branch_block(x, 256)
x = multi_branch_block(x, 256)
x = multi_branch_block(x, 256)

# Stage 3: 28 -> 14
x = reduction_block(x, 384)
x = multi_branch_block(x, 384)
x = multi_branch_block(x, 384)

x = layers.Concatenate()([
    layers.GlobalAveragePooling2D()(x),
    layers.GlobalMaxPooling2D()(x),
])
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(NUM_CLASSES)(x)

model = tf.keras.Model(inputs, outputs, name="complex_cnn")

AdamW = (tf.keras.optimizers.AdamW if hasattr(tf.keras.optimizers, "AdamW")
         else tf.keras.optimizers.experimental.AdamW)

model.compile(
    optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.05),
    metrics=["accuracy"],
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6, verbose=1),
    ModelCheckpoint(BEST_MODEL, monitor="val_loss", save_best_only=True, verbose=1),
]

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

model = tf.keras.models.load_model(BEST_MODEL)
model.save(OUT_DIR / "complex_cnn_model.h5")