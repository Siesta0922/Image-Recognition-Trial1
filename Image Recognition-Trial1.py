import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as leras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image

# Define the path to the dataset
data_dir = r'F:\Siesta\Professional Stuff\College\Extras\Reference\dataset'  # Replace with your dataset path

# Initialize lists to store images and labels
images = []
labels = []
class_names = []

# Load the images and their labels
for label, fruit in enumerate(os.listdir(data_dir)):
    fruit_dir = os.path.join(data_dir, fruit)
    class_names.append(fruit)
    for img_file in os.listdir(fruit_dir):
        img_path = os.path.join(fruit_dir, img_file)
        image = cv2.imread(img_path)
        if image is not None:
            image = cv2.resize(image, (100, 100))  # Resize images to a fixed size
            images.append(image)
            labels.append(label)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Normalize the images
images = images / 255.0

# Convert labels to categorical format
num_classes = len(np.unique(labels))
labels = to_categorical(labels, num_classes)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Print dataset information
print(f"Total images: {len(images)}")
print(f"Number of classes: {num_classes}")
print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),  # First convolutional layer
    MaxPooling2D((2, 2)),  # First max pooling layer
    Conv2D(64, (3, 3), activation='relu'),  # Second convolutional layer
    MaxPooling2D((2, 2)),  # Second max pooling layer
    Conv2D(128, (3, 3), activation='relu'),  # Third convolutional layer
    MaxPooling2D((2, 2)),  # Third max pooling layer
    Flatten(),  # Flatten the output
    Dense(128, activation='relu'),  # Fully connected layer
    Dropout(0.5),  # Dropout for regularization
    Dense(num_classes, activation='softmax')  # Output layer with softmax activation
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model with data augmentation
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=20, validation_data=(X_test, y_test))

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.2f}")

# Plot training & validation accuracy and loss values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.show()

# Function to preprocess new images for prediction
def preprocess_image(img_path):
    try:
        # Attempt to read the image using OpenCV
        image = cv2.imread(img_path)
        if image is not None:
            image = cv2.resize(image, (100, 100))
            image = image / 255.0  # Normalize
            image = np.expand_dims(image, axis=0)  # Add batch dimension
            print("Image loaded and processed with OpenCV.")
        else:
            # Fallback to PIL if OpenCV fails
            print("OpenCV failed to load the image. Trying PIL...")
            image = Image.open(img_path)
            image = image.resize((100, 100))
            image = np.array(image) / 255.0  # Normalize
            image = np.expand_dims(image, axis=0)  # Add batch dimension
            print("Image loaded and processed with PIL.")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    return image

def predict_image(img_path):
    image = preprocess_image(img_path)
    if image is not None:
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)
        class_name = class_names[predicted_class[0]]
        print(f"Predicted class: {class_name}")
        return class_name, prediction
    else:
        print("Error processing image.")
        return None, None

# Function to predict and display multiple images from a directory
def predict_multiple_images(directory):
    fig, axes = plt.subplots(1, 9, figsize=(20, 20))  # Adjust number of columns based on your preference
    img_count = 0
    for img_file in os.listdir(directory):
        img_path = os.path.join(directory, img_file)
        class_name, _ = predict_image(img_path)
        if class_name:
            # Display the image
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[img_count].imshow(img_rgb)
            else:
                # Use PIL to display the image if OpenCV fails
                img = Image.open(img_path)
                axes[img_count].imshow(img)

            axes[img_count].set_title(f"Predicted: {class_name}")
            axes[img_count].axis('off')
            img_count += 1
            
            # Stop after displaying 5 images to avoid cluttering (adjust this as needed)
            if img_count == 9:
                break
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Example: Predict multiple images from a specific folder
predict_multiple_images(r'F:\Siesta\Professional Stuff\College\Extras\Reference\Images')
