
from PIL import Image
import numpy as np
import random

class Paint():
    def __init__(self, width=1000, height=1000):
        self.brush = []
        self.width = width
        self.height = height
        self.paint_coords = np.zeros((width, height), dtype=np.uint8)
        self.target_size = (31, 11)

    def load_brush(self, image_path):
        image = Image.open(image_path).convert("RGBA")  # Handle transparency

        image_array = np.array(image)

        rgb_array = image_array[:, :, :3]
        alpha_channel = image_array[:, :, 3]

        non_black_pixels = np.any(rgb_array != [0, 0, 0], axis=-1)
        non_transparent_pixels = alpha_channel > 0
        valid_pixels = non_black_pixels & non_transparent_pixels

        coords = np.argwhere(valid_pixels)
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)

        cropped_array = rgb_array[x_min:x_max+1, y_min:y_max+1]

        cropped_image = Image.fromarray(cropped_array)
        (width, height) = self.target_size
        resized_image = cropped_image.resize((height, width), Image.Resampling.LANCZOS)

        self.brush = self.brush + [np.array(resized_image)]
        

    def paint_at_pixel(self, buff: np.ndarray, org_x, org_y, canvas: np.ndarray):
        (target_x, target_y) = self.target_size
        start_x =  max(org_x - target_x // 2, 0)
        start_y = max(org_y - target_y // 2, 0)
        
        end_x = min(org_x + target_x // 2, self.width)
        end_y = min(org_y + target_y // 2, self.height)

        # print(start_x, start_y, end_x, end_y, target_x, target_y, org_x, org_y)
# 
        # Calculate mean color of the region in the buffer
        region = buff[start_x:end_x, start_y:end_y]
        mean_color = region.mean(axis=(0, 1))  # Calculate mean for each channel (R, G, B)

        # Apply the brush stroke
        for x, brush_x in zip(range(start_x, end_x), range(0,target_x)):
            for y, brush_y in zip(range(start_y, end_y), range(0, target_y)):
                # print(x, y, brush_x, brush_y)
                # print(self.brush.shape)
                brush_pixel = self.brush[np.random.randint(0,len(self.brush))][brush_x, brush_y]
                brush_alpha = np.mean(brush_pixel) / 255.0

                if brush_alpha > 0:  
                    canvas[x, y] = mean_color
                    self.paint_coords[x, y] = 1

    def is_filled_90_percent(self, threshold=1.0, fill_ratio=0.999):
        total_pixels = self.paint_coords.size
        filled_pixels = np.count_nonzero(self.paint_coords >= threshold)
        return filled_pixels / total_pixels >= fill_ratio
    
    def paint_random_pixel_of_100x100(self, row, col, size=100, samples_per_block=10):
        block = self.paint_coords[row:row + size, col:col + size]
        mask = block == 0 
        indices = np.argwhere(mask)

        if len(indices) >= samples_per_block:
            random_indices = indices[np.random.choice(len(indices), size=samples_per_block, replace=False)]
        else:
            random_indices = indices
        global_indices = random_indices + np.array([row, col])

        return global_indices.tolist()


