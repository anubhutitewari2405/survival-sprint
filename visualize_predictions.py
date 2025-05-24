import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import random

# Paths
test_dir = "cnn_test_data"
model_path = "movement_classifier.h5"
img_size = (64, 64)
batch_size = 32

# Load model
model = load_model(model_path)

# Prepare test data
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Predict on test data
predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype("int32").flatten()
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

# Save confusion matrix figure and close it to avoid backend issues
plt.savefig('confusion_matrix.png')
plt.close()

print("Confusion matrix saved as confusion_matrix.png")

# Visualize a few predictions
filenames = test_generator.filenames
plt.figure(figsize=(12, 8))
for i in range(6):
    idx = random.randint(0, len(filenames) - 1)
    img_path = os.path.join(test_dir, filenames[idx])
    img = load_img(img_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    pred = model.predict(np.expand_dims(img_array, axis=0))[0][0]
    pred_label = class_labels[int(pred > 0.5)]
    true_label = class_labels[true_classes[idx]]

    plt.subplot(2, 3, i + 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Predicted: {pred_label}\nActual: {true_label}")

plt.tight_layout()
plt.suptitle("Sample Predictions", fontsize=16)
plt.subplots_adjust(top=0.85)

# Save sample predictions figure and close it to avoid backend issues
plt.savefig('sample_predictions.png')
plt.close()

print("Sample predictions saved as sample_predictions.png")
