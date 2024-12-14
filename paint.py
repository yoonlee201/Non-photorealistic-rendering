
from PIL import Image
import numpy as np
import random
import cv2

class Paint:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.set_up_palette()
    
    def set_up_palette(self):
        # type = { size: {angle: [brushes]}}
        self.brush_angle = 30
        self.brushes = {}
        
        self.brush_long = [60, 40 ,20]
        self.brush_short= [20, 10]
        
        brush_size=[60, 40, 20, 10]
        
        brush_paths = [
            'brush/brush-5.png', 
            'brush/brush-6.png',
            'brush/brush-7.png',
            'brush/brush-8.png',
            'brush/brush-9.png',
            'brush/brush-10.png',
            'brush/brush-11.png',
            'brush/brush-12.png',
            'brush/brush-13.png',
            'brush/brush-14.png', 
                       ]
        
        for size in brush_size:
            self.brushes[size] = {}
            for i in range(0, 360, self.brush_angle):
                self.brushes[size][i] = []
                
        for size in brush_size:
            for path in brush_paths:
                self.load_brush(path, size)
    
    def set_canvas(self, bg_color):
        self.canvas = np.full((self.height, self.width, 3), bg_color, dtype=np.uint8)
        
    def set_paint_coords(self, paint_coords):
        if paint_coords is not None:
            self.paint_coords = paint_coords
        else:
            self.paint_coords = np.zeros((self.width, self.height), dtype=np.uint8)
        
    def set_gradient_magnitude(self, gradient_magnitude: np.ndarray):
        self.gradient_magnitude = gradient_magnitude.copy()
     
    def load_brush(self, image_path, size):
        print("Loading brush, path:", image_path, 'size:', size)
        images = []
        original_image = Image.open(image_path).convert("RGBA") 
        
        for i in range(0, 360, self.brush_angle):
            images.append(original_image.rotate(i))

        for image, i in zip(images, range(0, 360, self.brush_angle)):
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
            resized_image = cropped_image.resize((size, size), Image.Resampling.LANCZOS)
            
            self.brushes[size][i] = self.brushes[size][i] + [np.array(resized_image)]
            
    def compute_gradient(self, color_buffer: np.ndarray):
        sobel_x = np.zeros_like(color_buffer, dtype=np.float32)
        sobel_y = np.zeros_like(color_buffer, dtype=np.float32)

        for i in range(3): 
            sobel_x[:, :, i] = cv2.Sobel(color_buffer[:, :, i], cv2.CV_64F, 1, 0, ksize=3)
            sobel_y[:, :, i] = cv2.Sobel(color_buffer[:, :, i], cv2.CV_64F, 0, 1, ksize=3)

        gradient_magnitude = np.sqrt(np.sum(sobel_x**2 + sobel_y**2, axis=2))
        gradient_direction = np.arctan2(np.mean(sobel_y, axis=2), np.mean(sobel_x, axis=2))

        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 100, cv2.NORM_MINMAX)
        
        return gradient_magnitude.astype(np.uint8), gradient_direction
        
    def get_angled_brush(self, x, y, size):
        angle = round(((np.degrees(self.gradient_direction[x, y]) + 360) % 360) / self.brush_angle) * self.brush_angle
        
        if angle in [0,90,180,270]:
            angle = angle + 90
            
        if angle == 360:
            angle = 0
        
        return self.brushes[size][angle][np.random.randint(0,len(self.brushes[size][angle]))]
    
    def paint_at_pixel(self, org_x, org_y, size, use_gradient=False):
        # Get the appropriate brush based on the angle
        brush = self.get_angled_brush(org_x, org_y, size)
        (target_x, target_y) = brush.shape[:2]
        
        start_x = max(org_x - target_x // 2, 0)
        start_y = max(org_y - target_y // 2, 0)
        end_x = min(org_x + target_x // 2, self.width)
        end_y = min(org_y + target_y // 2, self.height)
        
        # Get the mean color of the region
        if self.faded_buffer is not None:
            region = self.faded_buffer[start_x:end_x, start_y:end_y]
        else:
            region = self.buff[start_x:end_x, start_y:end_y]
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
                    self.canvas[x, y] = (1 - brush_alpha) * self.canvas[x, y] + brush_alpha * mean_color
                    self.gradient_magnitude[x, y] = 0  # Mark as painted in gradient mode
                else:
                    self.canvas[x, y] = mean_color
                
                # Mark the pixel as painted
                self.paint_coords[x, y] = 1

    def is_filled_paint_coords(self, fill_ratio, threshold=1.0):
        total_pixels = self.paint_coords.size
        filled_pixels = np.count_nonzero(self.paint_coords >= threshold)
        print(f'Filled pixels: {((filled_pixels / total_pixels)) * 100:.2f}%')
        return (filled_pixels / total_pixels) >= fill_ratio
    
        # print(f'Filled pixels: {filled_pixels/total_pixels * 100:.2f}%')
        # return filled_pixels / total_pixels >= fill_ratio

    def is_filled_gradient_magnitude(self, fill_ratio, threshold=1.0):
        total_pixels = self.gradient_magnitude.size
        filled_pixels = np.count_nonzero(self.gradient_magnitude >= threshold)
        print(f'Filled pixels: {(1-(filled_pixels / total_pixels)) * 100:.2f}%')
        return 1-(filled_pixels / total_pixels) >= fill_ratio

    def paint_random_pixel_of_100x100(self, row, col, size=100, samples_per_block=10):
        block = self.paint_coords[row:row + size, col:col + size]
        indices = np.argwhere(block == 0)

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

    def paint_on_canvas(self, buff:np.ndarray, bg_color, faded_buffer:np.ndarray=None,  phong_buffer:np.ndarray=None, paint_coords:np.ndarray=None):
        # initialize the buffers
        self.buff = buff
        self.set_canvas(bg_color)
        self.set_paint_coords(paint_coords)
        self.faded_buffer=faded_buffer
        
        # sobel filter objects
        if phong_buffer is not None:
            gradient_magnitude, self.gradient_direction = self.compute_gradient(phong_buffer)
        else :
            gradient_magnitude, self.gradient_direction = self.compute_gradient(buff)
        
        self.set_gradient_magnitude(gradient_magnitude)
        
        
        
        
        fill = [0.98, 0.7, 0.5]
        
        print('Painting')
        for brush_size, ratio in zip(self.brush_long, fill):
            print(f'Painting with brush size: {brush_size}, Fill: {ratio * 100:.2f}%')
            small_box_width = self.width//100
            small_box_height = self.height//100
            
            while not self.is_filled_paint_coords(ratio):
                for i in range(0, small_box_width):
                    for j in range(0, small_box_height):
                        random_indices = self.paint_random_pixel_of_100x100(i*100, j*100)
                        for x, y in random_indices:
                            self.paint_at_pixel(x, y, brush_size)
                        
                
    
            self.set_paint_coords(paint_coords)
            
            
        fill = [0.99, 0.97]
        print('Gradient')
        for brush_size, ratio in zip(self.brush_short, fill):    
            while not self.is_filled_gradient_magnitude(fill_ratio=ratio):
                random_indices = self.paint_random_pixel_of_gradient_magnitude()
                for x, y in random_indices:
                    self.paint_at_pixel(x, y, brush_size, use_gradient=True) 
            self.set_gradient_magnitude(gradient_magnitude)
            
            
        return self.canvas    
        