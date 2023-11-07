import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the NPZ dataset
data = np.load('public_data.npz', allow_pickle=True)

images = data['data']
labels = data['labels']

# Check the dimensions of the dataset
print("Number of images:", len(images))
print("Image shape:", images[0].shape)
print("Number of labels:", len(labels))

num_samples_to_visualize = 10
for i in range(num_samples_to_visualize):
    plt.subplot(1, num_samples_to_visualize, i + 1)
    imageInt = images[i].astype(np.uint8)
    print(imageInt)
    plt.imshow(imageInt)  # Convert from RGB to GRAYSCALE
    plt.title("Label: " + labels[i])
    plt.axis('off')

plt.show()
