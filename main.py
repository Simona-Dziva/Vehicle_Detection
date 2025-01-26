import cv2

import numpy as np
from matplotlib import pyplot as plt

import pandas as pd

from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Input, Dense, Dropout

from matplotlib import pyplot as plt

from keras.datasets import mnist

import kagglehub

# Download latest version
path = kagglehub.dataset_download("brsdincer/vehicle-detection-image-set")

print("Path to dataset files:", path)

def load_images(dataset_dir):
    """
    Loads images from a given directory and its subdirectories.

    Args:
        dataset_dir (str): Path to the directory containing images.

    Returns:
        tuple: A tuple containing:
            - images (list): List of loaded images as NumPy arrays.
            - image_paths (list): List of corresponding file paths for the images.
    """
    images = []  # Initialize an empty list to store the loaded images
    image_paths = []  # Initialize an empty list to store the paths of the images

    # Walk through the directory structure
    for root, _, files in os.walk(dataset_dir):
        # Iterate over all files in the current directory
        for file in files:
            # Check if the file has an image extension (jpg, png, jpeg)
            if file.endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(root, file)  # Construct the full file path
                image = cv2.imread(image_path)  # Load the image using OpenCV
                if image is not None:  # Check if the image was loaded successfully
                    images.append(image)  # Add the image to the list
                    image_paths.append(image_path)  # Add the corresponding path to the list

    # Return the loaded images and their file paths
    return images, image_paths