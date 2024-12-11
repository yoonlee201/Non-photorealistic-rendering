
from transform import Transform
import numpy as np
from vector import Vector3, Vector4, Vector4x4

class Camera:
    def __init__(self, left, right, bottom, top, near, far):
        self.transform = Transform()
        self.coord = (left, right, bottom, top, near, far)
    
    def get_view_vector(self):
        return self.transform.apply_to_normal(Vector3(0, 0, 1))

    def depth(self):
        (_, _, _, _, n, f) = self.coord
        return abs(f - n)
        
    def ratio(self):
        (left, right, bottom, top, _, _) = self.coord
        return abs((right - left) / (top - bottom))
    
    def project_point(self, p: Vector3):
        return Vector3(0,0,0)
    
    def inverse_project_point(self, p: Vector3):
        return Vector3(0,0,0)
    

class OrthoCamera (Camera):
    def __init__(self, left, right, bottom, top, near, far):
        Camera.__init__(self, left, right, bottom, top, near, far)
    
    def ratio(self):
        return Camera.ratio(self)
    
    def project_point(self, p: Vector3):
        (left, right, bottom, top, near, far) = self.coord
        ortho = Vector4x4(
            [[2/(right-left), 0, 0, -(right+left)/(right-left)],
             [0, 2/(top-bottom), 0, -(top+bottom)/(top-bottom)],
             [0, 0, 2/(near-far), -(near+far)/(near-far)],
             [0, 0, 0, 1]])
        
        p = self.transform.apply_inverse_to_point(p)
        return ((ortho) * p.homogeneous()).remove_W()

    
    def inverse_project_point(self, p: Vector3):
        (left, right, bottom, top, near, far) = self.coord
        
        ortho_inv = Vector4x4(np.linalg.inv(np.array(
            [[2/(right-left), 0, 0, -(right+left)/(right-left)],
             [0, 2/(top-bottom), 0, -(top+bottom)/(top-bottom)],
             [0, 0, 2/(near-far), -(near+far)/(near-far)],
             [0, 0, 0, 1]])))
        
        p_inv = (ortho_inv * p.homogeneous()).remove_W()
        return self.transform.apply_to_point(p_inv)


class PerspectiveCamera(Camera):
    def __init__(self, left, right, bottom, top, near, far):
        Camera.__init__(self, left, right, bottom, top, near, far)
    
    def ratio(self):
        return Camera.ratio(self)
    
    def project_point(self, p: Vector3):
        (left, right, bottom, top, near, far) = self.coord
        perspective = Vector4x4(
            [[near, 0, 0, 0],
             [0, near, 0, 0],
             [0, 0, near+far, -near*far],   
             [0, 0, 1, 0]])
        
        ortho = Vector4x4(
            [[2/(right-left), 0, 0, -(right+left)/(right-left)],
             [0, 2/(top-bottom), 0, -(top+bottom)/(top-bottom)],
             [0, 0, 2/(near-far), -(near+far)/(near-far)],
             [0, 0, 0, 1]])
        
       
        p = self.transform.apply_inverse_to_point(p)
        return ((ortho * perspective) * p.homogeneous()).remove_W()
    
    def inverse_project_point(self, p: Vector3):
        (left, right, bottom, top, near, far) = self.coord
        
        ortho_inv = Vector4x4(np.linalg.inv(np.array(
            [[2/(right-left), 0, 0, -(right+left)/(right-left)],
             [0, 2/(top-bottom), 0, -(top+bottom)/(top-bottom)],
             [0, 0, 2/(near-far), -(near+far)/(near-far)],
             [0, 0, 0, 1]])))

        perspective_inv = Vector4x4(np.linalg.inv(np.array(
            [[near, 0, 0, 0],
             [0, near, 0, 0],
             [0, 0, near+far, -near*far],   
             [0, 0, 1, 0]]))) 

        p_inv = ((perspective_inv * ortho_inv) * p.homogeneous()).remove_W()
        return self.transform.apply_to_point(p_inv)
    
    @staticmethod    
    def from_FOV(fov, near, far, ratio):
        fov_rad = np.deg2rad(fov)
        
        right = near * np.tan(fov_rad/2)
        left = -right
        top = right / ratio
        bottom = -top
        
        cls = PerspectiveCamera(left, right, bottom, top, near, far)
        
        return cls
        
        