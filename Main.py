import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Disable oneDNN optimisations if you need exact reproducibility
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Paths
train_dir = r"C:\Users\Jeeva\Documents\Fishimageclassification\Fish_image_classification\Data\train"
val_dir   = r"C:\Users\Jeeva\Documents\Fishimageclassification\Fish_image_classification\Data\val"

# Verify paths exist
for d in [train_dir, val_dir]:
    p = Path(d)
    if not p.is_dir():
        raise FileNotFoundError(f"Directory not found: {p}")

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.0  # we are using separate val folder
)
val_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)
print("Train classes:", train_generator.class_indices)
num_classes = len(train_generator.class_indices)
print("Number of classes:", num_classes)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)
print("Validation classes:", validation_generator.class_indices)
if len(validation_generator.class_indices) != num_classes:
    raise ValueError(f"Training and validation folder differ in classes: train={num_classes}, val={len(validation_generator.class_indices)}")

# Model definition
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(num_classes, activation='softmax')   # <-- match number of classes
])

model.summary()

# Compile
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train
print("Starting training...")
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)
print("Training completed.")

# After training your model
model.save(r"C:\Users\Jeeva\Documents\Fishimageclassification\Fish_image_classification\fish_model.keras")
print("Model saved to fish_model.keras")
