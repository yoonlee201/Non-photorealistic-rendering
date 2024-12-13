
from PIL import Image
import numpy as np
import random
import cv2

class Paint:
    def __init__(self, width=1000, height=1000):
        # self.brush = []
        
        self.paint_coords = np.zeros((width, height), dtype=np.uint8)
        self.width = width
        self.height = height
        
        # self.brush_size = [(80,80), (60,60), (40,40), (20,20)]
        self.gradient_magnitude = []
        self.rotate = {}
        self.gradient = False

    def initialize_paint_coords(self):
        """Reset the paint coordinates for a new canvas."""
        self.paint_coords = np.zeros((self.width, self.height), dtype=np.uint8)
        # self.brush = []
        self.rotate = {
            0: [],
            # 15: [],
            30: [],
            # 45: [],
            60: [],
            # 75: [],
            90: [],
            # 105: [],
            120: [],
            # 135: [],
            150: [],
            # 165: [],
            180: [],
            # 195: [],
            210: [],
            # 225: [],
            240: [],
            # 255: [],
            270: [],
            # 285: [],
            300: [],
            # 315: [],
            330: [],
            # 345: [],
        }
        
    def initialize_gradient_magnitude(self, gradient_magnitude):
        self.gradient_magnitude = gradient_magnitude.copy()
     

    def compute_color_gradient(self, color_buffer: np.ndarray) -> np.ndarray:
        # Calculate Sobel gradients for each channel (R, G, B)
        sobel_x = np.zeros_like(color_buffer, dtype=np.float32)
        sobel_y = np.zeros_like(color_buffer, dtype=np.float32)

        for i in range(3):  # For each color channel (R, G, B)
            sobel_x[:, :, i] = cv2.Sobel(color_buffer[:, :, i], cv2.CV_64F, 1, 0, ksize=5)
            sobel_y[:, :, i] = cv2.Sobel(color_buffer[:, :, i], cv2.CV_64F, 0, 1, ksize=5)

        # Compute gradient magnitude and direction
        gradient_magnitude = np.sqrt(np.sum(sobel_x**2 + sobel_y**2, axis=2))
        gradient_direction = np.arctan2(np.mean(sobel_y, axis=2), np.mean(sobel_x, axis=2))

        # Normalize gradient magnitude for visualization
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 40, cv2.NORM_MINMAX)
        # gradient_direction = cv2.normalize(gradient_direction, None, 0, 2*np.pi, cv2.NORM_MINMAX)

        # Return as ui  nt8 for further processing
        return gradient_magnitude.astype(np.uint8), gradient_direction
       
    def load_brush(self, image_path, size):
        print("Loading brush")
        images = []
        original_image = Image.open(image_path).convert("RGBA") 
        
        for i in range(0, 360, 30):
            images.append(original_image.rotate(i))

        for image, i in zip(images, range(0, 360, 30)):
            image_array = np.array(image)

            rgb_array = image_array[:, :, :3]
            alpha_channel = image_array[:, :, 3]

            non_black_pixels = np.any(rgb_array != [0, 0, 0], axis=-1)
            non_transparent_pixels = alpha_channel > 0
            valid_pixels = non_black_pixels & non_transparent_pixels

            coords = np.argwhere(valid_pixels)
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)

            if i in [0, 180]:  # Horizontal strokes
                cropped_array = rgb_array[:, y_min:y_max+1]
            elif i in [90, 270]:  # Vertical strokes
                cropped_array = rgb_array[x_min:x_max+1, :]
            else:
                cropped_array = rgb_array[x_min:x_max+1, y_min:y_max+1]

            cropped_image = Image.fromarray(cropped_array)
            (width, height) = size
            resized_image = cropped_image.resize((height, width), Image.Resampling.LANCZOS)
            self.rotate[i] = self.rotate[i] + [np.array(resized_image)]
        # self.brush = self.brush + [np.array(resized_image)]
        
    def get_angled_brush(self, x, y, angle_buffer: np.ndarray):
        angle = round(((np.degrees(angle_buffer[x, y]) + 360) % 360) / 30) * 30
        # print(np.degrees(angle_buffer[x, y]), angle)
        if angle in [0,90,180,270]:
            angle = angle + 90
            
        if angle == 360:
            angle = 0
            
            
        # print("Angle: " , angle, angle_buffer[x, y])
        return self.rotate[angle][np.random.randint(0,len(self.rotate[angle]))]
    
    def paint_at_pixel(self, buff: np.ndarray, org_x, org_y, canvas: np.ndarray, angle_buffer: np.ndarray, use_gradient=False):
        # Get the appropriate brush based on the angle
        brush = self.get_angled_brush(org_x, org_y, angle_buffer)
        (target_x, target_y) = brush.shape[:2]
        
        start_x = max(org_x - target_x // 2, 0)
        start_y = max(org_y - target_y // 2, 0)
        end_x = min(org_x + target_x // 2, self.width)
        end_y = min(org_y + target_y // 2, self.height)
        
        # Get the mean color of the region
        region = buff[start_x:end_x, start_y:end_y]
        mean_color = region.mean(axis=(0, 1))
        
        # Paint the region
        for x, brush_x in zip(range(start_x, end_x), range(0, target_x)):
            for y, brush_y in zip(range(start_y, end_y), range(0, target_y)):
                # Get a random brush pixel
                brush_pixel = brush[brush_x, brush_y]
                brush_alpha = np.mean(brush_pixel) / 255.0

                # Skip if brush alpha is 0
                if brush_alpha <= 0:
                    continue
                
                # Apply either gradient or standard painting logic
                if use_gradient:
                    canvas[x, y] = (1 - brush_alpha) * canvas[x, y] + brush_alpha * mean_color
                    self.gradient_magnitude[x, y] = 0  # Mark as painted in gradient mode
                else:
                    canvas[x, y] = mean_color
                
                # Mark the pixel as painted
                self.paint_coords[x, y] = 1


    def is_filled_90_percent(self, threshold=1.0, fill_ratio=0.999):
        total_pixels = self.paint_coords.size
        filled_pixels = np.count_nonzero(self.paint_coords >= threshold)
        print(filled_pixels / total_pixels)
        return filled_pixels / total_pixels >= fill_ratio

    def is_filled_color_gradient_magnitude(self, threshold=1.0, fill_ratio=0.999):
        total_pixels = self.gradient_magnitude.size
        filled_pixels = np.count_nonzero(self.gradient_magnitude >= threshold)
        return 1-(filled_pixels / total_pixels) >= fill_ratio



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
    
    def paint_random_pixel_of_gradient_magnitude(self, samples=10):
        # Identify pixels with non-zero gradient magnitude
        indices = np.argwhere(self.gradient_magnitude > 0)

        # Check if there are no valid pixels left
        if len(indices) == 0:
            return []

        # Randomly sample pixels
        random_indices = indices[np.random.choice(len(indices), size=min(samples, len(indices)), replace=False)]
        return random_indices.tolist()
        