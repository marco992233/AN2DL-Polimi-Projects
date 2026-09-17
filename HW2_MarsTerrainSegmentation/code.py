# -*- coding: utf-8 -*-
#**1. ⚙️ Setup Enviroment**

# Connect Colab to Google Drive
"""

# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive
drive.mount('/content/gdrive', force_remount = True)
# %cd /content/gdrive/My Drive/Second_Challenge

"""# Import Libraries"""

# Commented out IPython magic to ensure Python compatibility.
!pip install tfa-nightly
import tensorflow_addons as tfa

# Set seed for reproducibility
seed = 42

# Import necessary libraries
import os

# Set environment variables before importing modules
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['MPLCONFIGDIR'] = os.getcwd() + '/configs/'

# Suppress warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

# Import necessary modules
import logging
import random
import numpy as np

# Set seeds for random number generators in NumPy and Python
np.random.seed(seed)
random.seed(seed)

# Import TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl

# Set seed for TensorFlow
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

# Reduce TensorFlow verbosity
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Print TensorFlow version
print(tf.__version__)

# Import other libraries
import os
import math
from PIL import Image
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

# %matplotlib inline

"""#**2. ⏳ Inspect and preprocess data**

# Load the Dataset
"""

data = np.load("Dataset/mars_for_students.npz")
training_set = data["training_set"]

X_train_val = training_set[:, 0]
Y_train_val = training_set[:, 1]

X_test = data["test_set"]

"""#Inspect Data

##Data inspection
"""

# Print the shapes of the loaded datasets
#Labels: 0: 'Basophil', 1:'Eosinophil', 2:'Erythroblas', 3:'Immature granulocytes', 4:'Lymphocyte', 5:'Monocyte', 6:'Neutrophil', 7:'Platelets'
print("Training-Validation Data Shape:", X_train_val.shape)
print("Training-Validation Label Shape:", Y_train_val.shape)
print("Number of unique labels:", np.unique(Y_train_val).shape)

unique_labels = np.unique(Y_train_val)
n_elements = []
ratio_elements = []
labels = []
tot_elements = Y_train_val.shape[0]
for i, label in enumerate(np.unique(Y_train_val)): # Use enumerate to get the index
  labels.append(label)
  n_elements.append(np.count_nonzero(Y_train_val == label))
  print(label,": ",n_elements[i]) # Use the index 'i' to access n_elements
# labels for bars
tick_label = [0, 1, 2, 3, 4]

ratio_elements = 100*np.array(n_elements)/tot_elements

# plotting a bar chart
plt.bar(labels, ratio_elements, tick_label = tick_label,width = 0.8, color = ['red', 'green'])

# naming the x-axis
plt.xlabel('Class Labels', )
# naming the y-axis
plt.ylabel('Percentage of elements')
# plot title
plt.title('Original data Distribution')

# function to show the plot
plt.show()

"""##Print some images"""

# Select 50 random indices
num_images = 50
random_indices = np.random.choice(X_train_val.shape[0], num_images, replace=False)

# Display the images and their corresponding label masks
plt.figure(figsize=(30, 30))
for i, idx in enumerate(random_indices):
    # Display the image
    plt.subplot(10, 10, 2 * i + 1)  # 10x10 grid for images
    plt.imshow(X_train_val[idx], cmap='gray')  # Use cmap='gray' for grayscale images
    plt.axis('off')  # Disable axes for a cleaner visualization
    plt.title(f"Image {i + 1} (Index {idx})")  # Add the index to the title

    # Display the label mask
    plt.subplot(10, 10, 2 * i + 2)  # 10x10 grid for label masks
    plt.imshow(Y_train_val[idx], cmap='tab20b')  # Use a colormap to display labels
    plt.axis('off')  # Disable axes for a cleaner visualization
    plt.title(f"Mask {i + 1} (Index {idx})")  # Add the index to the title

plt.tight_layout()
plt.show()

num_images = 100
random_indices = np.random.choice(X_test.shape[0], num_images, replace=False)

# Visualizza le immagini e le loro maschere di etichetta
plt.figure(figsize=(30, 30))
for i, idx in enumerate(random_indices):
    # Visualizza l'immagine
    plt.subplot(20, 20, 2 * i + 1)  # Griglia 10x10 per le immagini
    plt.imshow(X_test[idx], cmap='gray')  # Usa cmap='gray' per immagini in scala di grigio
    plt.axis('off')  # Disabilita gli assi per una visualizzazione pulita
    plt.title(f"Image {i + 1} (Index {idx})")  # Aggiungi l'indice nel titolo
plt.tight_layout()
plt.show()

"""##Eliminate contaminated images"""

#Take the ground mask of one of the contaminated images (with index 1324)
paul_image = Y_train_val[1324]
plt.imshow(paul_image, cmap='tab20b')
delete_indices = []
for index in range(Y_train_val.shape[0]):
    # Check if the image is contaminated (compare with 'paul_image')
    if np.array_equal(paul_image, Y_train_val[index]):
        delete_indices.append(index)

if delete_indices:
    delete_indices = np.array(delete_indices)
    print(delete_indices)
    X_train_val = np.delete(X_train_val, delete_indices, axis=0)
    Y_train_val = np.delete(Y_train_val, delete_indices, axis=0)

# Print updated shapes of X_train_val and Y_train_val
print(f"Updated X_train_val shape: {X_train_val.shape}")
print(f"Updated Y_train_val shape: {Y_train_val.shape}")

"""##Save new dataset"""

np.savez('Dataset/new_mars_for_student.npz', data = X_train_val, label = Y_train_val)

print("Dataset cleaned and saved as 'new_mars_for_student.npz'")

"""# New Dataset Processing

##Load Data
"""

# Load the .npz file
data = np.load('Dataset/new_mars_for_student.npz')

# Explore the contents of the .npz file
print(data.files)  # Display the names of the arrays in the file

# Assume the data is organized into two arrays: "images" and "labels"
images = data['data']  # Replace with the actual name of the image array
labels = data['label']  # Replace with the actual name of the label array

# Create paths for the images and labels
image_paths = [f"/content/gdrive/My Drive/Second_Challenge/Dataset/Dataset_right/images/{i}.png" for i in range(len(images))]
label_paths = [f"/content/gdrive/My Drive/Second_Challenge/Dataset/Dataset_right/labels/{i}.png" for i in range(len(labels))]

# Check the first few paths for images and labels
print("First 5 image paths:", image_paths[:5])
print("First 5 label paths:", label_paths[:5])

"""##Set Parameters"""

# Set batch size for training
BATCH_SIZE = 64

# Set learning rate for the optimiser
LEARNING_RATE = 1e-4

# Set early stopping patience threshold
PATIENCE = 15

# Set maximum number of training epochs
EPOCHS = 50

# Set data split size for training and validation
SPLITS_SIZE = 0.2

"""##Fix image shape"""

images = images[..., np.newaxis] / 255.0

print(f"images shape: {images.shape}")

"""##Split dataset"""

# Split the paths (not the data) into training, validation, and test sets
print("Splitting data...")
train_img, val_img, train_lbl, val_lbl= train_test_split(
    images, labels, test_size=SPLIT_SIZE, random_state=seed
)
print("Data splitted!")

print(f"\nNumber of images:")
print(f"Train: {len(train_img)}")
print(f"Validation: {len(val_img)}")

print(f"\nNumber of images:")
print(f"Train img: {train_img.shape}")
print(f"Train lbl: {train_lbl.shape}")
print(f"Validation: {len(val_img)}")

print(f"Train shape: {train_img.shape}")
print(f"Validation shape: {val_img.shape}")

"""##Define category mapping (not really useful for this dataset)"""

# Define whether to use the simple or detailed version
simple_version = True

# Define the category mapping
if simple_version:
    category_map = {
        0: 0,  # unlabelled
        1: 1,  # static
        2: 2,  # ground
        3: 3,  # road
        4: 4,  # sidewalk

    }
else:
    category_map = {
        0: 0,  # unlabelled
        1: 1,  # static
        2: 2,  # ground
        3: 3,  # road
        4: 4,  # sidewalk
    }


# Calculate the correct number of classes after mapping
NUM_CLASSES = len(set(category_map.values()))
print(f"Number of original categories: {len(category_map)}")
print(f"Number of classes after mapping: {NUM_CLASSES}")

"""##Augmentations

###Flip Orizontal
"""

@tf.function
def random_flip(image, label, seed=None):
    """Consistent random horizontal flip."""
    if seed is None:
        seed = np.random.randint(0, 1000000)
    flip_prob = tf.random.uniform([], seed=seed)
    image = tf.cond(
        flip_prob > -0.5,
        lambda: tf.image.flip_left_right(image),
        lambda: image
    )
    # Add a channel dimension to the label before flipping
    label = tf.expand_dims(label, axis=-1)
    label = tf.cond(
        flip_prob > -0.5,
        lambda: tf.image.flip_left_right(label),
        lambda: label
    )
    # Remove the channel dimension after flipping
    label = tf.squeeze(label, axis=-1)
    return image, label

"""###Flip Vertical"""

@tf.function
def random_vertical_flip(image, label, seed=None):
    """Consistent random vertical flip."""
    if seed is None:
        seed = np.random.randint(0, 1000000)
    flip_prob = tf.random.uniform([], seed=seed)
    image = tf.cond(
        flip_prob > -0.5,
        lambda: tf.image.flip_up_down(image),
        lambda: image
    )
    # Add a channel dimension to the label before flipping
    label = tf.expand_dims(label, axis=-1)
    label = tf.cond(
        flip_prob > -0.5,
        lambda: tf.image.flip_up_down(label),
        lambda: label
    )
    # Remove the channel dimension after flipping
    label = tf.squeeze(label, axis=-1)
    return image, label

"""###Rotate"""

@tf.function
def random_rotate(image, label, max_angle=30, seed=None):
    """Randomly rotate an image and its corresponding label."""
    if seed is None:
        seed = np.random.randint(0, 1000000)
    angle = tf.random.uniform([], -max_angle, max_angle, seed=seed)
    image = tfa.image.rotate(image, angle * np.pi / 180)  # Rotate the image
    label = tfa.image.rotate(label, angle * np.pi / 180)  # Rotate the label
    return image, label

# Select a sample image from the training set
test_image = train_img[0]
test_label = train_lbl[0]

# Display the original image and label
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(test_image.squeeze(), cmap='gray')
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(test_label.squeeze(), cmap='tab20')
plt.title("Original Label")
plt.show()

# Apply the random rotation
rotated_image, rotated_label = random_rotate(test_image, test_label, max_angle=30)

# Display the rotated image and label
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(rotated_image.numpy().squeeze(), cmap='gray')
plt.title("Rotated Image")

plt.subplot(1, 2, 2)
plt.imshow(rotated_label.numpy().squeeze(), cmap='tab20')
plt.title("Rotated Label")
plt.show()

"""###Translate"""

@tf.function
def random_translate(image, label, max_translation=0.1, seed=None):
    """Random translation of the image and label using tfa.image.translate."""
    if seed is None:
        seed = np.random.randint(0, 1000000)

    # Calculate random translation along x and y axes
    translate_x = tf.random.uniform([], -max_translation, max_translation, seed=seed) * tf.cast(image.shape[1], tf.float32)
    translate_y = tf.random.uniform([], -max_translation, max_translation, seed=seed) * tf.cast(image.shape[0], tf.float32)

    # Apply the translation to both the image and the label using tfa.image.translate
    image = tfa.image.translate(image, [translate_x, translate_y])
    label = tfa.image.translate(label, [translate_x, translate_y])

    return image, label

# Assume we have a sample image and label
test_image = train_img[0]  # Use one of the images from your dataset
test_label = train_lbl[0]  # Use the corresponding label

# Display the original image and label
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(test_image.squeeze(), cmap='gray')
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(test_label.squeeze(), cmap='tab20')
plt.title("Original Label")
plt.show()

# Apply the random translation
translated_image, translated_label = random_translate(test_image, test_label, max_translation=0.2)

# Display the image and label after translation
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(translated_image.numpy().squeeze(), cmap='gray')
plt.title("Translated Image")

plt.subplot(1, 2, 2)
plt.imshow(translated_label.numpy().squeeze(), cmap='tab20')
plt.title("Translated Label")
plt.show()

"""###Augmentation array"""

#The array is composed by (augmentation_function, probability of the augmentation) pairs
augmentations = [
    (random_flip, 1),
    (random_vertical_flip, 1),
    (random_rotate, 1),
    (random_translate, 1),
    ]

"""##Make Dataset

###Make Dataset function
"""

def make_dataset(images, labels, batch_size, shuffle=True, augmentations=None, seed=None):
    """
    Create a memory-efficient TensorFlow dataset.

    Args:
        images: Array of input images.
        labels: Array of corresponding labels.
        batch_size: Batch size for the dataset.
        shuffle: Whether to shuffle the dataset.
        augmentations: List of tuples [(augmentation_function, probability), ...].
        seed: Random seed for reproducibility.

    Returns:
        A tf.data.Dataset object.
    """
    def apply_augmentations(image, label):
        """Apply a sequence of augmentations based on probabilities and keep the original."""
        def augment_single(image, label, aug_fn, prob):
            return tf.cond(
                tf.random.uniform([]) < prob,
                lambda: aug_fn(image, label),
                lambda: (image, label)
            )

        augmented_images = [image]
        augmented_labels = [label]

        for aug_fn, prob in augmentations:
            aug_image, aug_label = augment_single(image, label, aug_fn, prob)
            augmented_images.append(aug_image)
            augmented_labels.append(aug_label)

        return tf.stack(augmented_images), tf.stack(augmented_labels)

    # Create dataset from file paths
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 2, seed=seed)

    if augmentations:
        dataset = dataset.map(
            lambda x, y: apply_augmentations(x, y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Flatten the augmented images and labels
        dataset = dataset.flat_map(
            lambda images, labels: tf.data.Dataset.from_tensor_slices((images, labels))
        )

    # Batch the data
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

"""###Make train and validation dataset"""

# Create the datasets
print("Creating datasets...")
train_dataset = make_dataset(
    train_img, train_lbl,
    batch_size=BATCH_SIZE,
    shuffle=True,
    augmentations=augmentations,
    seed=seed
)


val_dataset = make_dataset(
    val_img, val_lbl,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Datasets created!")

# Check the shape of the data
for images, labels in train_dataset.take(1):
    input_shape = images.shape[1:]
    print(f"\nInput shape: {input_shape}")
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Labels dtype:", labels.dtype)
    break

"""###Inspect datasets"""

def create_segmentation_colormap(num_classes):
    """
    Create a linear colormap using a predefined palette.
    Uses 'viridis' as default because it is perceptually uniform
    and works well for colorblindness.
    """
    return plt.cm.viridis(np.linspace(0, 1, num_classes))

def apply_colormap(label, colormap=None):
    """
    Apply the colormap to a label.
    """
    # Ensure label is 2D
    label = np.squeeze(label)

    if colormap is None:
        num_classes = len(np.unique(label))
        colormap = create_segmentation_colormap(num_classes)

    # Apply the colormap
    colored = colormap[label.astype(int)]

    return colored

def plot_sample_batch(dataset, num_samples=3):
    """
    Display some image and label pairs from the dataset.
    """
    plt.figure(figsize=(15, 4*num_samples))

    for images, labels in dataset.take(1):
        labels_np = labels.numpy()
        num_classes = len(np.unique(labels_np))
        colormap = create_segmentation_colormap(num_classes)

        for j in range(min(num_samples, len(images))):
            # Plot original image
            plt.subplot(num_samples, 2, j*2 + 1)
            plt.imshow(images[j])
            plt.title(f'Image {j+1}')
            plt.axis('off')

            # Plot colored label
            plt.subplot(num_samples, 2, j*2 + 2)
            colored_label = apply_colormap(labels_np[j], colormap)
            plt.imshow(colored_label)
            plt.title(f'Label {j+1}')
            plt.axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()

# Visualize examples from the training set
print("Visualizing examples from the training set:")
plot_sample_batch(train_dataset, num_samples=64)

"""#**3. Model Training**

# 🛠️ Models and Experiments

##Define model
"""

def residual_unet_block(input_tensor, filters, kernel_size=3, activation='relu', stack=2, name=''):
    # Save the original input for the residual path
    shortcut = input_tensor

    # Apply a series of Conv2D, Batch Normalization, and Activation
    x = input_tensor
    for i in range(stack):
        x = tfkl.Conv2D(filters, kernel_size=kernel_size, padding='same', name=name + 'conv' + str(i + 1))(x)
        x = tfkl.BatchNormalization(name=name + 'bn' + str(i + 1))(x)
        x = tfkl.Activation(activation, name=name + 'activation' + str(i + 1))(x)

    # If the shortcut's shape is different, use a conv layer to adapt it
    if shortcut.shape[-1] != filters:
        shortcut = tfkl.Conv2D(filters, kernel_size=1, padding='same', name=name + 'shortcut_conv')(shortcut)

    # Add the residual path
    x = tfkl.Add()([x, shortcut])
    x = tfkl.Activation(activation)(x)

    return x

def get_residual_unet_model(input_shape=(64, 128, 1), num_classes=NUM_CLASSES, seed=seed):
    tf.random.set_seed(seed)
    input_layer = tfkl.Input(shape=input_shape, name='input_layer')

    # Downsampling path with residual blocks
    down_block_1 = residual_unet_block(input_layer, 64, name='down_block1_')
    d1 = tfkl.MaxPooling2D()(down_block_1)

    down_block_2 = residual_unet_block(d1, 128, name='down_block2_')
    d2 = tfkl.MaxPooling2D()(down_block_2)

    # Bottleneck with residual block
    bottleneck = residual_unet_block(d2, 256, name='bottleneck')

    # Upsampling path with residual blocks
    u1 = tfkl.UpSampling2D()(bottleneck)
    u1 = tfkl.Concatenate()([u1, down_block_2])
    u1 = residual_unet_block(u1, 128, name='up_block1_')

    u2 = tfkl.UpSampling2D()(u1)
    u2 = tfkl.Concatenate()([u2, down_block_1])
    u2 = residual_unet_block(u2, 64, name='up_block2_')

    # Output Layer
    output_layer = tfkl.Conv2D(num_classes, kernel_size=1, padding='same', activation="softmax", name='output_layer')(u2)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name='ResidualUNet')
    return model

"""##Inspect Model"""

model = get_residual_unet_model()

# Print a detailed summary of the model with expanded nested layers and trainable parameters.
model.summary(expand_nested=True, show_trainable=True)

# Generate and display a graphical representation of the model architecture.

"""## Define custom Mean Intersection Over Union metric"""

import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
from tensorflow.keras.saving import register_keras_serializable  # Import for custom object registration

# Define custom Mean Intersection Over Union metric
@register_keras_serializable()
class MeanIntersectionOverUnion(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes, labels_to_exclude=None, name="mean_iou", dtype=None, **kwargs):  # **kwargs to absorb extra arguments
        super(MeanIntersectionOverUnion, self).__init__(num_classes=num_classes, name=name, dtype=dtype)
        if labels_to_exclude is None:
            labels_to_exclude = [0]  # Default to excluding label 0
        self.labels_to_exclude = labels_to_exclude

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions to class labels
        y_pred = tf.math.argmax(y_pred, axis=-1)

        # Flatten the tensors
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        # Apply mask to exclude specified labels
        for label in self.labels_to_exclude:
            mask = tf.not_equal(y_true, label)
            y_true = tf.boolean_mask(y_true, mask)
            y_pred = tf.boolean_mask(y_pred, mask)

        # Update the state
        return super().update_state(y_true, y_pred, sample_weight)

"""##Define Visualization callback"""

# Visualization callback
class VizCallback(tf.keras.callbacks.Callback):
    def __init__(self, image, label, frequency=5):
        super().__init__()
        self.image = image
        self.label = label
        self.frequency = frequency

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.frequency == 0:  # Visualize only every "frequency" epochs
            image = self.image
            label = self.label
            print(image.shape)
            image1 = tf.expand_dims(image, 0)
            pred = self.model.predict(image1, verbose=0)
            y_pred = tf.math.argmax(pred, axis=-1)
            y_pred = y_pred.numpy()

            # Create colormap
            num_classes = NUM_CLASSES
            colormap = create_segmentation_colormap(num_classes)

            plt.figure(figsize=(16, 4))

            # Input image
            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title("Input Image")
            plt.axis('off')

            # Ground truth
            plt.subplot(1, 3, 2)
            colored_label = apply_colormap(label, colormap)
            plt.imshow(colored_label)
            plt.title("Ground Truth Mask")
            plt.axis('off')

            # Prediction
            plt.subplot(1, 3, 3)
            colored_pred = apply_colormap(y_pred, colormap)
            plt.imshow(colored_pred)
            plt.title("Predicted Mask")
            plt.axis('off')

            plt.tight_layout()
            plt.show()
            plt.close()

"""##Define custom loss"""

import tensorflow as tf
import tensorflow.keras.backend as K

@register_keras_serializable()
class SpecializedSegmentationLoss:
    def __init__(self,
                 num_classes=5,
                 ignore_index=0,
                 alpha=0.7,  # Dice loss weight
                 beta=0.3,   # Focal loss weight
                 gamma=2.0,  # Focal loss focusing parameter
                 smooth=1e-7):
        """
        Specialized loss for grayscale semantic segmentation

        Args:
            num_classes: Total number of classes
            ignore_index: Class index to ignore (usually background)
            alpha: Weight for Dice loss
            beta: Weight for Focal loss
            gamma: Focal loss focusing parameter
            smooth: Prevent division by zero
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def weighted_dice_loss(self, y_true, y_pred):
        """Weighted Dice loss with background class handling"""
        # Create mask to ignore specific class
        mask = tf.not_equal(y_true, self.ignore_index)
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)

        # One-hot encode
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=self.num_classes)

        # Compute intersection and union
        intersection = K.sum(y_true_one_hot * y_pred, axis=[0,1])
        union = K.sum(y_true_one_hot, axis=[0,1]) + K.sum(y_pred, axis=[0,1])

        # Class-balanced Dice coefficient
        dice_coef = (2. * intersection + self.smooth) / (union + self.smooth)

        # Weight classes inversely proportional to their frequency
        class_weights = 1.0 / (K.sum(y_true_one_hot, axis=[0,1]) + self.smooth)
        weighted_dice = dice_coef * class_weights

        return 1 - K.mean(weighted_dice)

    def focal_loss(self, y_true, y_pred):
        """Focal loss with background class handling"""
        # Create mask to ignore specific class
        mask = tf.not_equal(y_true, self.ignore_index)
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)

        # One-hot encode
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=self.num_classes)

        # Compute focal loss
        pt_1 = tf.where(tf.equal(y_true_one_hot, 1), y_pred, 1 - y_pred)
        focal_loss = -K.mean(K.pow(1. - pt_1, self.gamma) * K.log(pt_1 + self.smooth))

        return focal_loss

    def __call__(self, y_true, y_pred):
        """Combine losses"""
        # Sparse Categorical Cross-Entropy (ignoring background)
        mask = tf.not_equal(y_true, self.ignore_index)
        sparse_ce = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False,
            reduction=tf.keras.losses.Reduction.NONE
        )(y_true[mask], y_pred[mask])

        # Weighted Dice loss
        dice = self.weighted_dice_loss(y_true, y_pred)

        # Focal loss
        focal = self.focal_loss(y_true, y_pred)

        # Weighted combination
        total_loss = (self.alpha * dice) + \
                     (self.beta * focal) + \
                     K.mean(sparse_ce)

        return total_loss

    def get_config(self):
      """Get configuration for serialization."""
      config = {
          "num_classes": self.num_classes,
          "ignore_index": self.ignore_index,
          "alpha": self.alpha,
          "beta": self.beta,
          "gamma": self.gamma,
          "smooth": self.smooth,
      }
      return config

    @classmethod
    def from_config(cls, config):
        """Reconstruct the loss function from its configuration."""
        return cls(**config)  # Create a new instance using the config


def get_specialized_segmentation_loss(num_classes=5, ignore_index=0):
    """Helper function to create the loss"""
    return SpecializedSegmentationLoss(num_classes=num_classes, ignore_index=ignore_index)

""" ## Compile model"""

advanced_loss = get_specialized_segmentation_loss(num_classes=NUM_CLASSES)

# Compile the model
print("Compiling model...")
model.compile(
    loss=advanced_loss,
    optimizer=tf.keras.optimizers.AdamW(LEARNING_RATE),
    metrics=["accuracy", MeanIntersectionOverUnion(num_classes=NUM_CLASSES, labels_to_exclude=[0])]
)
print("Model compiled!")

"""##Setup callbacks"""

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_mean_iou',
    mode='max',
    patience=PATIENCE,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='Models/Nick/checkpoint.keras',
    monitor='val_mean_iou',
    mode='max',
    save_best_only=True,
    verbose = True
)

reduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_mean_iou',
    factor=0.01,
    patience=10,
    mode = 'min',
    min_lr=1e-7,
    verbose=1
)

viz_callback = VizCallback(val_img[0], val_lbl[0])

"""##Load Model (if needed)"""

model = tf.keras.models.load_model('Models/Nick/checkpoint.keras', custom_objects={'MeanIntersectionOverUnion': MeanIntersectionOverUnion})

"""##Train Model"""

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=[early_stopping , viz_callback, checkpoint, reduceLROnPlateau],
    verbose=1
).history

# Calculate and print the final validation accuracy
final_val_meanIoU = round(max(history['val_mean_iou'])* 100, 2)
print(f'Final validation Mean Intersection Over Union: {final_val_meanIoU}%')

# # Save the trained model to a file with the accuracy included in the filename
model_filename = 'UNet_Residual_'+str(final_val_meanIoU)+'.keras'
model.save("Models/" + model_filename)

"""# 📊 Prepare Submission"""

model = tf.keras.models.load_model('Models/UNet_Residual_55.17.keras', custom_objects={'MeanIntersectionOverUnion': MeanIntersectionOverUnion})
data = np.load("Dataset/mars_for_students.npz")
X_test = data["test_set"]
X_test = X_test[..., np.newaxis] / 255.0
preds = model.predict(X_test)
preds = np.argmax(preds, axis=-1)
print(f"Predictions shape: {preds.shape}")

import pandas as pd

def y_to_df(y) -> pd.DataFrame:
    """Converts segmentation predictions into a DataFrame format for Kaggle."""
    n_samples = len(y)
    y_flat = y.reshape(n_samples, -1)
    df = pd.DataFrame(y_flat)
    df["id"] = np.arange(n_samples)
    cols = ["id"] + [col for col in df.columns if col != "id"]
    return df[cols]

submission_filename = f"submission1.csv"
submission_df = y_to_df(preds)
submission_df.to_csv(submission_filename, index=False)
from google.colab import files
files.download(submission_filename)

# Plot and display training and validation loss
plt.figure(figsize=(18, 3))
plt.plot(history['loss'], label='Training', alpha=0.8, color='#ff7f0e', linewidth=2)
plt.plot(history['val_loss'], label='Validation', alpha=0.9, color='#5a9aa5', linewidth=2)
plt.title('Cross Entropy')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Plot and display training and validation accuracy
plt.figure(figsize=(18, 3))
plt.plot(history['accuracy'], label='Training', alpha=0.8, color='#ff7f0e', linewidth=2)
plt.plot(history['val_accuracy'], label='Validation', alpha=0.9, color='#5a9aa5', linewidth=2)
plt.title('Accuracy')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Plot and display training and validation mean IoU
plt.figure(figsize=(18, 3))
plt.plot(history['mean_iou'], label='Training', alpha=0.8, color='#ff7f0e', linewidth=2)
plt.plot(history['val_mean_iou'], label='Validation', alpha=0.9, color='#5a9aa5', linewidth=2)
plt.title('Mean Intersection over Union')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
