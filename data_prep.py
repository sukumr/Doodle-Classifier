import numpy as np
from numpy import save

from PIL import Image

"""
Load numpy array file
.npy :numpy bitmaps
"""

imgdata = np.load("bird.npy")  
# print(imgdata.shape) # (133572, 784)

# Considering only one image for visualization
sample = imgdata[209:210, :]
print(sample)
# converting a Numpy array to PIL image
# img = Image.fromarray(sample, "L") 
img = Image.fromarray(sample.reshape(28, 28), "L") # L mode indicates the array values represents luminance.
img.show()

data = []

for i in range(1000):
    # Creating new list with first 1000 images
    new_data = imgdata[i:i+1, :]
    data.append(new_data)

# save as 
# save("bird1000", data)

