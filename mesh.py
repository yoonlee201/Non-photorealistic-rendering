from typing import List
import numpy
from stl import mesh
from vector import Vector3
from transform import Transform


class Mesh:
    def __init__(self, diffuse_color,specular_color, ka, kd, ks, ke):
        self.verts: List[Vector3] = []
        self.faces = []
        self.normals: List[Vector3] = []
        self.vertes_normals: List[Vector3] = []
        self.transform = Transform()
        
        
        self.color = diffuse_color, specular_color
        self.reflection_coef = (ka, kd, ks, ke)
        

    def add_vertex(self, vertex):
        self.verts.append(vertex)
    
    
    def init_vertex_normal(self):
        self.vertes_normals.append(Vector3(0,0,0))
        
    def add_vertex_normal(self, index:int, normal:Vector3 ):
        original = self.vertes_normals[index]
        self.vertes_normals[index] = (original + normal).normalized()

    def add_face(self, face):
        self.faces.append(face)

    def add_normal(self, normal):
        self.normals.append(normal)
        
    def __getattr__(self, name):
        
        diffuse_color, specular_color = self.color
        (ka, kd, ks, ke) = self.reflection_coef
        # Convienent way to get elements
        if name == 'diffuse_color':
            return diffuse_color
        if name == 'specular_color':
            return specular_color
        if name == 'ka':
            return ka
        if name == 'kd':
            return kd
        if name == 'ks':
            return ks
        if name == 'ke':
            return ke
        
    @staticmethod
    def from_stl(stl_file, diffuse_color, specular_color, ka, kd, ks, ke):
        # load the mesh from the file
        stl_mesh = mesh.Mesh.from_file(stl_file)
        
        # create a new Mesh object
        this_mesh = Mesh(diffuse_color, specular_color, ka, kd, ks, ke)

        # extract vertices from the mesh
        for i in range(len(stl_mesh.vectors)):
            # this_mesh.add_normal(Vector3.from_array(stl_mesh.normals[i]))
            p0, p1, p2 = stl_mesh.vectors[i]
            
            # getting the vectors for cross product
            v1 = Vector3.from_array(p1) - Vector3.from_array(p0)
            v2 = Vector3.from_array(p2) - Vector3.from_array(p0)
            
            # calculating the normal and add it to the mesh
            n = v1.cross(v2).normalized()
            this_mesh.add_normal(n)
            
            # add the vertices to the mesh
            for arr in stl_mesh.vectors[i]:
                v = Vector3.from_array(arr)
                if v not in this_mesh.verts:
                    this_mesh.add_vertex(v)
                    this_mesh.init_vertex_normal()
            
            p0_index = this_mesh.verts.index(Vector3.from_array(p0))
            p1_index = this_mesh.verts.index(Vector3.from_array(p1))
            p2_index = this_mesh.verts.index(Vector3.from_array(p2))
            
            this_mesh.add_vertex_normal(p0_index, n)
            this_mesh.add_vertex_normal(p1_index, n)
            this_mesh.add_vertex_normal(p2_index, n)
            
            # getting the indicies of the triangle vertices and adding them to the face list
            this_mesh.add_face((p0_index, p1_index, p2_index))
        # return the mesh 
        return this_mesh
