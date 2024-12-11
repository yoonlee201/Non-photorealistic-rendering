import numpy as np
from screen import Screen
from typing import List, Tuple
from vector import Vector3
from camera import Camera
from mesh import Mesh
from light import Light
from PIL import Image
from paint import Paint

class Renderer:
    def __init__(self, screen: Screen, camera: Camera, mesh_list: List[Mesh], light: Light):
        self.screen = screen
        self.camera = camera
        self.mesh_list = mesh_list
        self.light = light
        self.paint = Paint(self.screen.width, self.screen.height)
        
    def render_barycentric(self, buff: np.ndarray, x, y, alpha, beta, gamma):
        buff[x, y] = (255 * np.array([alpha, beta, gamma]))

    def render_flat(self, buff: np.ndarray, x, y, z, point_light:Vector3, norm_vec:Vector3, ambient, _diffuse, is_edge):
        p = self.screen.inverse_project_point(Vector3(x, y, z))
        frag_world = self.camera.inverse_project_point(p)
        light_vec = (point_light-frag_world)
        
        # vector normalization
        L = (light_vec).normalized()
        N = norm_vec.normalized()
        
        # diffuse reflection
        d = (light_vec).magnitude()
        diffuse = (_diffuse * max(L.dot(N), 0)) * (1/np.pi)* (self.light.intensity  / (d**2))
        
        if is_edge:
            color = (255,0,0)
        else:
            color = np.clip((ambient + diffuse).to_array(), 0, 1) * 255
        
        buff[x,y] = color
        
    def render_phong(self, buff: np.ndarray, x, y, z, alpha, beta, gamma, n1, n2, n3,point_light:Vector3, ambient, _diffuse, _specular, mesh:Mesh, is_edge):
        
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
        if is_edge:
            color = (255,0,0)
        else:
            color = np.clip((ambient + diffuse + specular).to_array(), 0, 1) * 255
        buff[x, y] = color
        
    def render_paint(self, buff: np.ndarray, x, y, z, alpha, beta, gamma, n1, n2, n3,point_light:Vector3, ambient, _diffuse, _specular, mesh:Mesh):
        
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
        
    def render(self, shading, bg_color, ambient_light):
        def barycentric(p, a, b, c):
            (x, y), (xa, ya), (xb, yb), (xc, yc) = p, a, b, c
            denominator = (ya - yb) * (xc - xb) + (xb - xa) * (yc - yb)
            
            if denominator == 0:
                return True, 0, 0, 0
            
            alpha = ((yb - yc) * (x - xc) + (xc - xb) * (y - yc)) / denominator
            beta = ((yc - ya) * (x - xc) + (xa - xc) * (y - yc)) / denominator
            gamma = 1 - alpha - beta
            return (alpha >= 0) and (beta >= 0) and (gamma >= 0), alpha, beta, gamma
        
        
        # Create the buffer with background color
        buff = np.full((self.screen.width, self.screen.height, 3), bg_color, dtype=np.uint8)
        z_buffer = np.full((self.screen.width, self.screen.height), -np.inf)
        angle_buffer = np.zeros((self.screen.width, self.screen.height), dtype=np.float32)
        size_buffer = np.zeros((self.screen.width, self.screen.height), dtype=np.float32)
        
        for mesh in self.mesh_list:
            # for every triangle in the mesh
            
            for (n1, n2, n3), i in zip(mesh.faces, range(len(mesh.faces))):
                # print(n1, n2, n3)
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
                
                if norm_vec.dot(self.camera.get_view_vector()) > 0:
                    continue
                
                is_edge = False
                for face_index in mesh.adjacent[i]:
                    adjacent_normal = mesh.transform.apply_to_normal(mesh.normals[face_index])
                    if adjacent_normal.dot(self.camera.get_view_vector()) >= 0:  # Adjacent face is not visible
                        # print(face_index, adjacent_normal, i)
                        is_edge = True
                        break  # No need to cchek further


                
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
                        
                        # if (alpha <= 0 or beta <= 0 or gamma <= 0):
                        #     color = (255,0,0)
                        #     buff[x, y] = color
                        #     continue
                        if check:
                            z = (alpha * z_coords[0] + beta * z_coords[1] + gamma * z_coords[2]) 
                            if z >= 0 and z > z_buffer[x, y]:
                                z_buffer[x, y] = z
                                
                                if shading == 'barycentric':
                                    self.render_barycentric(buff, x, y, alpha, beta, gamma)
                                elif shading == 'flat':
                                    self.render_flat(buff, x, y, z, point_light, norm_vec, ambient, _diffuse, is_edge)
                                elif shading == 'phong' or shading == 'paint':
                                    self.render_phong(buff, x, y, z, alpha, beta, gamma, n1, n2, n3, point_light, ambient, _diffuse, _specular, mesh, is_edge)
                                elif shading == 'gouraud':
                                    self.render_gouraud(buff, x, y, z, alpha, beta, gamma, n1, n2, n3, point_light, ambient, _diffuse, _specular, mesh)
                                elif shading == 'paint':
                                    self.render_paint()
                                else:
                                    gray = np.clip(255 * (z), 0, 255)
                                    buff[x, y] = (gray, gray, gray)
                                    
        if shading == 'paint': 
            canvas = np.full((self.screen.width, self.screen.height, 3), bg_color, dtype=np.uint8)
        
            paint_long_size = [(41, 21), (31, 11), (31, 7), (21, 7)]
            paint_point_size = [(21, 21), (11, 11), (7, 7), (5, 5)]
            fill = [0.9999, 0.7, 0.5, 0.3]
            # brush_size = (21, 7)
            
            for brush_long, brush_point, ratio, index in zip(paint_long_size, paint_point_size, fill, range(3)):
                # for brush_size in paint_size:
                self.paint.initialize_paint_coords()
                self.paint.load_brush('brush/brush-1.png', brush_point)
                self.paint.load_brush('brush/brush-2.png', brush_point)
                self.paint.load_brush('brush/brush-3.png', brush_point)
                self.paint.load_brush('brush/brush-4.png', brush_point)
                self.paint.load_brush('brush/brush-long-1.png', brush_long)
                self.paint.load_brush('brush/brush-long-3.png', brush_long)
                self.paint.load_brush('brush/brush-long-4.png', brush_long)
                self.paint.load_brush('brush/brush-long-6.png', brush_long)
                self.paint.load_brush('brush/brush-long-7.png', brush_long)
                
                small_box_width = self.screen.width//100
                small_box_height = self.screen.height//100
                
                while not self.paint.is_filled_90_percent(fill_ratio=ratio):
                    for i in range(0, small_box_width):
                        for j in range(0, small_box_height):
                            random_indices = self.paint.paint_random_pixel_of_100x100(i*100, j*100)
                            for x, y in random_indices:
                                self.paint.paint_at_pixel(buff, x, y, canvas, index)
                # break
                # # Regular space painting
                # for x in range(0, self.screen.width, self.paint.target_size[0]):
                #     for y in range(0, self.screen.height, self.paint.target_size[1]):
                #         self.paint.paint_at_pixel(buff,  x, y, buff[x,y])
            self.screen.draw(canvas)
            
        else: 
            self.screen.draw(buff)
        
        
