#evaluate_movement_classifier
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Paths
test_dir = "cnn_test_data"
img_size = (64, 64)
batch_size = 32
model_path = "movement_classifier.h5"

# Load the saved model
model = load_model(model_path)

# Data generator for test data (only rescaling, no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False  # Important for evaluation to not shuffle
)

# Evaluate model on test data
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
