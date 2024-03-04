# Let's write a program to generate 500 random noise images of 128x128 pixels and save them into individual .npy files.

import numpy as np
import os

# Directory to save the .npy files
directory_path = './frames/test_3/'

# Ensure the directory exists
os.makedirs(directory_path, exist_ok=True)

# Generate and save 500 random noise images
for i in range(20):
    # Generate a single random noise image
    # Scale to 0-255 for pixel values
    image = np.random.rand(128, 128, 1) * 255
    image = image.astype(np.uint8)  # Convert to unsigned 8-bit integer type

    # Define the file name
    file_name = f"fake_pong_{i}.npy"

    # Full path for the file to be saved
    file_path = os.path.join(directory_path, file_name)

    # Save the image array to a .npy file
    np.save(file_path, image)

# Return the directory path to confirm where the files have been saved
