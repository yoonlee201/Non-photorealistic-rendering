import numpy as np
from screen import Screen
from typing import List, Tuple
from vector import Vector3
from camera import Camera
from mesh import Mesh
from light import Light
from PIL import Image
from paint import Paint
import math
import cv2

class Renderer:
    def __init__(self, screen: Screen, camera: Camera, mesh_list: List[Mesh], light: Light):
        self.screen = screen
        self.camera = camera
        self.mesh_list = mesh_list
        self.light = light
        self.paint = Paint(self.screen.width, self.screen.height)

    def compute_color_gradient(self, color_buffer: np.ndarray) -> np.ndarray:
        # Calculate gradients for each channel (R, G, B)
        sobel_x = np.zeros_like(color_buffer, dtype=np.float32)
        sobel_y = np.zeros_like(color_buffer, dtype=np.float32)

        for i in range(3):  # For each channel (R, G, B)
            sobel_x[:, :, i] = cv2.Sobel(color_buffer[:, :, i], cv2.CV_64F, 1, 0, ksize=3)
            sobel_y[:, :, i] = cv2.Sobel(color_buffer[:, :, i], cv2.CV_64F, 0, 1, ksize=3)

        # Compute gradient magnitude for each channel
        gradient_magnitude = np.sqrt(np.sum(sobel_x**2 + sobel_y**2, axis=2))

        # Normalize the gradient for visualization
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        return gradient_magnitude.astype(np.uint8)
    
    def compute_depth_gradient(self, z_buffer: np.ndarray) -> np.ndarray:
        # Calculate Sobel gradients in X and Y directions
        sobel_x = cv2.Sobel(z_buffer, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in X direction
        sobel_y = cv2.Sobel(z_buffer, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in Y direction

        # Compute gradient magnitude
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # Normalize the gradient for visualization
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        return gradient_magnitude.astype(np.uint8)
    
    def compute_normal_gradient(self, normal_buffer: np.ndarray) -> np.ndarray:
        # Calculate gradients for each normal component
        sobel_x = np.zeros_like(normal_buffer, dtype=np.float32)
        sobel_y = np.zeros_like(normal_buffer, dtype=np.float32)

        for i in range(3):  # For each component (x, y, z)
            sobel_x[:, :, i] = cv2.Sobel(normal_buffer[:, :, i], cv2.CV_64F, 1, 0, ksize=3)
            sobel_y[:, :, i] = cv2.Sobel(normal_buffer[:, :, i], cv2.CV_64F, 0, 1, ksize=3)

        # Combine gradients to calculate magnitude
        gradient_magnitude = np.sqrt(np.sum(sobel_x**2 + sobel_y**2, axis=2))

        # Normalize for visualization
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        return gradient_magnitude.astype(np.uint8)


    def render(self, shading, bg_color, ambient_light):
        def barycentric(p, a, b, c):
            (x, y), (xa, ya), (xb, yb), (xc, yc) = p, a, b, c
            denominator = (ya - yb) * (xc - xb) + (xb - xa) * (yc - yb)
            
            if denominator == 0:
                return False, 1, 1, 1
            
            alpha = ((yb - yc) * (x - xc) + (xc - xb) * (y - yc)) / denominator
            beta = ((yc - ya) * (x - xc) + (xa - xc) * (y - yc)) / denominator
            gamma = 1 - alpha - beta
            return (alpha >= 0) and (beta >= 0) and (gamma >= 0), alpha, beta, gamma
        
        
        # Create the buffer with background color
        buff = np.full((self.screen.width, self.screen.height, 3), bg_color, dtype=np.uint8)
        canvas = np.full((self.screen.width, self.screen.height, 3), (225,225,225), dtype=np.uint8)
        z_buffer = np.full((self.screen.width, self.screen.height), -np.inf, dtype=np.float64)
        normal_buffer = np.full((self.screen.width, self.screen.height, 3), (0,0,0), dtype=np.uint8)
        depth_buffer = np.full((self.screen.width, self.screen.height, 3), (0,0,0), dtype=np.uint8)
        
        for mesh in self.mesh_list:
            for (n1, n2, n3), i in zip(mesh.faces, range(len(mesh.faces))):
                triangle = [mesh.verts[n1], mesh.verts[n2], mesh.verts[n3]]
                # vector to world coordinates
                world_coords = [mesh.transform.apply_to_point(p) for p in triangle]
                # world to camera coordinates
                camera_coords = [self.camera.project_point(p) for p in world_coords]
                
                # camera to screen coordinates
                view_coords = [self.screen.project_point(p) for p in camera_coords]
                screen_coords = [(int(x), int(y)) for (x, y, _) in view_coords]
                z_coords = [z for (_, _, z) in view_coords]
                
                # basic vectors for lighting
                norm_vec  = mesh.transform.apply_to_normal(mesh.normals[i])
                # norm_degree = math.degrees(math.acos(norm_vec.dot(self.camera.get_view_vector())))

                
                point_light = self.light.transform.apply_to_point(Vector3(0, 0, 0))
                
                
                ambient = Vector3.from_array(ambient_light)  * mesh.ka
                _diffuse = self.light.apply_with_diffuse(Vector3.from_array(mesh.diffuse_color)) * mesh.kd
                _specular = (Vector3.from_array(mesh.specular_color) * mesh.ks)
                
                # get bounds for the screen
                min_x = max(min(v[0] for v in screen_coords), 0)
                max_x = min(max(v[0] for v in screen_coords), self.screen.width - 1)
                min_y = max(min(v[1] for v in screen_coords), 0)
                max_y = min(max(v[1] for v in screen_coords), self.screen.height - 1)
                
                for x in range(min_x, max_x + 1):
                    for y in range(min_y, max_y + 1):
                        
                        check, alpha, beta, gamma = barycentric((x, y), *screen_coords)
                        
                        if check:
                            z = (alpha * z_coords[0] + beta * z_coords[1] + gamma * z_coords[2]) 
                            if z >= 0 and z > z_buffer[x, y]:
                                p = self.screen.inverse_project_point(Vector3(x, y, z))
                                frag_world = self.camera.inverse_project_point(p)
                                light_vec = (point_light - frag_world)
                                
                                # Normalize the light direction and normal vectors
                                L = light_vec.normalized()
                                N = (mesh.vertes_normals[n1] * alpha + mesh.vertes_normals[n2] * beta + mesh.vertes_normals[n3] * gamma).normalized()
                                
                                # Diffuse reflection
                                d = light_vec.magnitude()
                                diffuse = (_diffuse * max(L.dot(N), 0)) * (1 / np.pi) * (self.light.intensity / (d ** 2))
                                
                                # Specular reflection
                                V = (self.camera.get_view_vector() - frag_world).normalized()
                                R = (L - N * 2 * L.dot(N)).normalized()  # Reflect L around N
                                specular = (_specular * max(R.dot(V), 0) ** mesh.ke)
                                
                                # Final color including ambient, diffuse, and specular
                                color = np.clip((ambient + diffuse + specular).to_array(), 0, 1) * 255
                                buff[x, y] = color
                                normal_buffer[x, y] = N.to_array()
                                z_buffer[x, y] = z
                                depth_buffer[x, y] = np.clip(255 * (z), 0, 255)
                                
               
        # if shading == 'paint':
            # Compute gradients
        color_gradient = self.compute_color_gradient(buff)
        normal_gradient = self.compute_normal_gradient(normal_buffer)

        # Stack grayscale gradients into RGB format
        if len(color_gradient.shape) == 2:
            color_gradient = np.stack((color_gradient,) * 3, axis=-1)
        if len(normal_gradient.shape) == 2:
            normal_gradient = np.stack((normal_gradient,) * 3, axis=-1)

        # Ensure uint8 format
        # color_gradient = color_gradient.astype(np.uint8)
        # depth_gradient = depth_gradient.astype(np.uint8)
        # normal_gradient = normal_gradient.astype(np.uint8)

        # Display normal gradient (or replace with color_gradient or depth_gradient)
        # self.screen.draw(normal_gradient)   
        # print(z_buffer)
        # depth_image = cv2.normalize(z_buffer, None, 0, 255, cv2.NORM_MINMAX)
        # depth_image = ((z_buffer + 1) / 2 * 255).astype(np.uint8)
        
        # cv2.imshow("Depth Buffer", depth_image)
        
        normal_image = ((normal_buffer + 1) / 2 * 255).astype(np.uint8)
        # cv2.imshow("Normal Buffer", normal_image)
        
        # print(depth_gradient)
        self.screen.draw(normal_image) 
                                    
        # elif shading == 'paint': 
        #     canvas = np.full((self.screen.width, self.screen.height, 3), bg_color, dtype=np.uint8)
        
        #     paint_long_large = [(41, 21), (31, 11)]
        #     paint_long_small = [(31, 7), (21, 7)]
        #     paint_point_large = [(21, 21), (11, 11)]
        #     paint_point_small = [(7, 7), (5, 5)]
        #     fill = [0.9999, 0.7, 0.5, 0.3]
            
        #     for brush_long, brush_point, ratio in zip(paint_long_large, paint_point_large, fill):
        #         # for brush_size in paint_size:
        #         self.paint.initialize_paint_coords()
        #         self.paint.load_brush('brush/brush-1.png', brush_point)
        #         self.paint.load_brush('brush/brush-2.png', brush_point)
        #         self.paint.load_brush('brush/brush-3.png', brush_point)
        #         self.paint.load_brush('brush/brush-4.png', brush_point)
        #         self.paint.load_brush('brush/brush-long-1.png', brush_long)
        #         self.paint.load_brush('brush/brush-long-3.png', brush_long)
        #         self.paint.load_brush('brush/brush-long-4.png', brush_long)
        #         self.paint.load_brush('brush/brush-long-6.png', brush_long)
        #         self.paint.load_brush('brush/brush-long-7.png', brush_long)
                
        #         small_box_width = self.screen.width//100
        #         small_box_height = self.screen.height//100
                
        #         while not self.paint.is_filled_90_percent(fill_ratio=ratio):
        #             for i in range(0, small_box_width):
        #                 for j in range(0, small_box_height):
        #                     random_indices = self.paint.paint_random_pixel_of_100x100(i*100, j*100)
        #                     for x, y in random_indices:
        #                         self.paint.paint_at_pixel(buff, x, y, canvas, index, z_buffer)
            
        #     # for brush_long, brush_point, ratio, index in zip(paint_long_small, paint_point_small, fill):
        #     #     # for brush_size in paint_size:
        #     #     self.paint.initialize_paint_coords()
        #     #     self.paint.load_brush('brush/brush-1.png', brush_point)
        #     #     self.paint.load_brush('brush/brush-2.png', brush_point)
        #     #     self.paint.load_brush('brush/brush-3.png', brush_point)
        #     #     self.paint.load_brush('brush/brush-4.png', brush_point)
        #     #     self.paint.load_brush('brush/brush-long-1.png', brush_long)
        #     #     self.paint.load_brush('brush/brush-long-3.png', brush_long)
        #     #     self.paint.load_brush('brush/brush-long-4.png', brush_long)
        #     #     self.paint.load_brush('brush/brush-long-6.png', brush_long)
        #     #     self.paint.load_brush('brush/brush-long-7.png', brush_long)
                
        #     #     small_box_width = self.screen.width//100
        #     #     small_box_height = self.screen.height//100
                
        #     #     while not self.paint.is_filled_90_percent(fill_ratio=ratio):
        #     #         for i in range(0, small_box_width):
        #     #             for j in range(0, small_box_height):
        #     #                 random_indices = self.paint.paint_random_pixel_of_100x100(i*100, j*100)
        #     #                 for x, y in random_indices:
        #     #                     self.paint.paint_at_pixel(buff, x, y, canvas, index, z_buffer)
        #     self.screen.draw(canvas)
            
        # else: 
        #     self.screen.draw(buff)
        
        
