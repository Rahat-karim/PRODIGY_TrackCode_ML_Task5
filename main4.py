import os
import zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Unzip the dataset
with zipfile.ZipFile('food-101.zip', 'r') as zip_ref:
    zip_ref.extractall('food-101')

# Set data directories
data_dir = 'food-101/food-101/images'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

# Prepare the data generator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_directory(data_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', subset='training')
val_generator = datagen.flow_from_directory(data_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', subset='validation')
# Load the pre-trained VGG16 model without the top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add new top layers
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
predictions = Dense(101, activation='softmax')(x)

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, validation_data=val_generator, epochs=10, steps_per_epoch=train_generator.samples // 32, validation_steps=val_generator.samples // 32)

# Save the model
model.save('food_recognition_model.h5')

# Example calorie mapping (replace with actual values)
calorie_mapping = {
    'apple_pie': 300, 'pizza': 266, 'burger': 295, # Add all 101 categories
    # ...
}

# Function to estimate calories based on model prediction
def estimate_calories(prediction):
    predicted_class = np.argmax(prediction)
    class_label = train_generator.class_indices
    reverse_class_label = {v: k for k, v in class_label.items()}
    food_item = reverse_class_label[predicted_class]
    return calorie_mapping.get(food_item, 'Unknown')

# Example usage
image_path = 'path_to_image.jpg'
image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
image = tf.keras.preprocessing.image.img_to_array(image)
image = np.expand_dims(image, axis=0) / 255.0
prediction = model.predict(image)
calories = estimate_calories(prediction)
print(f"Estimated Calories: {calories}")