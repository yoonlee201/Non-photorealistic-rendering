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
        faded = np.full((self.screen.width, self.screen.height, 3), (0,0,0), dtype=np.float32)
        depth_buffer = np.full((self.screen.width, self.screen.height, 3), (0,0,0), dtype=np.uint8)
        canvas = np.full((self.screen.width, self.screen.height, 3), bg_color, dtype=np.uint8)
        
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
                                depth = min(np.clip(255 * (z), 0, 255)/225 + 0.5, 1.0)
                                fade_color = np.array(bg_color) # Fading toward white

                                color = np.clip((ambient + diffuse + specular).to_array(), 0, 1) * 255
                                
                                fade_weight = 0.8
                                faded_color = (1 - depth * fade_weight) * fade_color + (depth * fade_weight) * color
                                canvas_color = (_diffuse * (1 / np.pi) * (self.light.intensity / (d ** 2))).to_array() * 255
                                canvas_fade = (1 - depth * fade_weight) * fade_color + (depth * fade_weight) * canvas_color
                                
                                buff[x, y] = color 
                                faded[x, y] = faded_color
                                canvas[x,y] = canvas_fade
                                z_buffer[x, y] = z
                                depth_buffer[x, y] = np.clip(255 * (z), 0, 255)
                                
        
        color_gradient_magnitude, color_gradient_direcion = self.paint.compute_color_gradient(buff)

        paint_size = [(60, 60),(40, 40),(20, 20)]
        fill = [0.98, 0.8, 0.7, 0.5]
        
        for brush_size, ratio in zip(paint_size, fill):
            # for brush_size in paint_size:
            self.paint.initialize_paint_coords()
            self.paint.load_brush('brush/brush-5.png', brush_size)
            self.paint.load_brush('brush/brush-6.png', brush_size)
            self.paint.load_brush('brush/brush-7.png', brush_size)
            self.paint.load_brush('brush/brush-8.png', brush_size)
            self.paint.load_brush('brush/brush-9.png', brush_size)
            self.paint.load_brush('brush/brush-10.png', brush_size)
            self.paint.load_brush('brush/brush-11.png', brush_size)
            self.paint.load_brush('brush/brush-12.png', brush_size)
            self.paint.load_brush('brush/brush-13.png', brush_size)
            self.paint.load_brush('brush/brush-14.png', brush_size)
            
            small_box_width = self.screen.width//100
            small_box_height = self.screen.height//100
            while not self.paint.is_filled_90_percent(fill_ratio=ratio):
                for i in range(0, small_box_width):
                    for j in range(0, small_box_height):
                        random_indices = self.paint.paint_random_pixel_of_100x100(i*100, j*100)
                        # print(random_indices)
                        for x, y in random_indices:
                            self.paint.paint_at_pixel(buff, x, y, canvas, color_gradient_direcion)
            
        
        paint_size = [(20, 20), (10, 10)]
        fill = [0.999, 0.98]
        for brush_size, ratio in zip(paint_size, fill):
            # for brush_size in paint_size:
            print("gradient")
            self.paint.initialize_paint_coords()
            self.paint.initialize_gradient_magnitude(color_gradient_magnitude)
            self.paint.load_brush('brush/brush-5.png', brush_size)
            self.paint.load_brush('brush/brush-6.png', brush_size)
            self.paint.load_brush('brush/brush-7.png', brush_size)
            self.paint.load_brush('brush/brush-8.png', brush_size)
            self.paint.load_brush('brush/brush-10.png', brush_size)
            self.paint.load_brush('brush/brush-11.png', brush_size)
            self.paint.load_brush('brush/brush-12.png', brush_size)
            self.paint.load_brush('brush/brush-13.png', brush_size)
            self.paint.load_brush('brush/brush-14.png', brush_size)
            
            while not self.paint.is_filled_color_gradient_magnitude(fill_ratio=ratio):
                random_indices = self.paint.paint_random_pixel_of_gradient_magnitude()
                for x, y in random_indices:
                    self.paint.paint_at_pixel(buff, x, y, canvas, color_gradient_direcion, use_gradient=True)

                
        self.screen.draw(canvas)
        
        
