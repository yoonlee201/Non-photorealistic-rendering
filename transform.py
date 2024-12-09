

import numpy as np
from vector import Vector3, Vector4x4


class Transform:
    def __init__(self):
        self.position = (0,0,0)
        self.rotation = (0,0,0)
        self.axis = (0,0,0)
        self.transform = Vector4x4(np.identity(4))

    def transformation_matrix(self):
        return self.transform.to_array()
    
    def set_position(self, x, y, z):
        self.position = (x, y, z)   
        self.transform = self.T() * self.R()                          

    def set_rotation(self, x, y, z):
        self.rotation = (x, y, z)
        self.transform = self.T() * self.R()
    
    def inverse_matrix(self):
        pos = self.T().to_array()
        pos[0][3] *= pos[0][3] == 0 and 0 or -1
        pos[1][3] *= pos[1][3] == 0 and 0 or -1
        pos[2][3] *= pos[2][3] == 0 and 0 or -1
        
        return (self.R().transpose() * Vector4x4(pos)).to_array() 

    
    def apply_to_point(self, p: Vector3):
        transform = self.transform
        return (transform * p.homogeneous()).remove_W()
        
        
    
    def apply_inverse_to_point(self, p: Vector3):
        v = p.homogeneous()
        return (Vector4x4(self.inverse_matrix())* v).remove_W()
       

    def apply_to_normal(self, n: Vector3):
        transform = self.R()
        
        v = (transform * n.homogeneous()).remove_W()
        return v.normalized()

    
    def set_axis_rotation(self, axis, rotation):
        self.axis = axis * rotation
        self.transform = self.T() * self.R()
    
    
     
    def T (self):
        (x, y, z) = self.position
        return Vector4x4(np.array([
            [1.,         0.,             0.,             x ],
            [0.,         1.,             0.,             y ],
            [0.,         0.,             1.,             z ],
            [0.,         0.,             0.,             1.]
        ]))
        
    def R (self):
        (Rx, Ry, Rz) = self.rotation
        (Ax, Ay, Az) = self.axis
        
        theta = np.deg2rad(Rx + Ax)
        xM = Vector4x4(np.array([
            [1.,         0.,             0.,             0.],
            [0.,         np.cos(theta), -np.sin(theta),  0.],
            [0.,         np.sin(theta),  np.cos(theta),  0.],
            [0.,         0.,             0.,             1.]
        ]))
        
        
        theta = np.deg2rad(Ry + Ay)
        yM = Vector4x4(np.array([
            [np.cos(theta),  0., np.sin(theta),  0.],
            [0.,             1., 0.,             0.],
            [-np.sin(theta), 0., np.cos(theta),  0.],
            [0.,             0., 0.,             1.]
        ]))
        
        theta = np.deg2rad(Rz + Az)
        zM = Vector4x4(np.array([
            [np.cos(theta), -np.sin(theta), 0., 0.],
            [np.sin(theta),  np.cos(theta), 0., 0.],
            [0.,             0.,            1., 0.],
            [0.,             0.,            0., 1.]
        ]))
        
        return xM * yM * zM
        