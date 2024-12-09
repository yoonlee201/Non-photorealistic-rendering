import numpy as np

from screen import Screen
from PIL import Image


screen = Screen(1000,1000)
bg_color = (60, 60, 70)
buff = np.full((screen.width, screen.height, 3), bg_color, dtype=np.uint8)


# Load the image and ensure RGB format
image = Image.open("brush-0.png").convert("RGBA")  # Handle transparency

# Convert to a NumPy array
image_array = np.array(image)

# Separate RGB and Alpha channels
rgb_array = image_array[:, :, :3]
alpha_channel = image_array[:, :, 3]

# Identify non-black and non-transparent pixels
non_black_pixels = np.any(rgb_array != [0, 0, 0], axis=-1)
non_transparent_pixels = alpha_channel > 0
valid_pixels = non_black_pixels & non_transparent_pixels

# Find the bounding box of the valid pixels
coords = np.argwhere(valid_pixels)
x_min, y_min = coords.min(axis=0)
x_max, y_max = coords.max(axis=0)

# Crop to the bounding box
cropped_array = rgb_array[x_min:x_max+1, y_min:y_max+1]

# Convert back to an image
cropped_image = Image.fromarray(cropped_array)

# Resize the cropped image to 500x500
target_size = (100, 100)
resized_image = cropped_image.resize(target_size, Image.Resampling.LANCZOS)

# Convert to NumPy array for further processing
resized_array = np.array(resized_image)

# buff = np.zeros_like(resized_array)
for x in range(resized_array.shape[0]):
    for y in range(resized_array.shape[1]):
        if (resized_array[x, y] != [0, 0, 0]).all():
            buff[x, y] = resized_array[x, y] 
            # [200, 200, 200] 
        
screen.draw(buff)

screen.show()
final_image = Image.fromarray(buff.astype('uint8'))
final_image.show()
