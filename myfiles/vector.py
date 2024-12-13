import numpy as np

class Vector3:
    @staticmethod
    def zeros():
        return Vector3(0, 0, 0)
    
    def __init__(self, x, y, z):
        self.vec = np.array([x, y, z])

    def __eq__(self, other):
        if isinstance(other, Vector3):
            return self.x == other.x and self.y == other.y and self.z == other.z
        elif isinstance(other, np.ndarray):
            return self.vec[0] == other[0] and self.vec[1] == other[1] and self.vec[2] == other[2]
        return False
    
    def __hash__(self):
        return hash((self.x, self.y, self.z))
    
    #p: Vector3
    def equals(self,p):
        return self.vec[0] == p.vec[0] and self.vec[1] == p.vec[1] and self.vec[2] == p.vec[2]

    #v: Vector3
    def dot(self, v):
        return np.dot(self.vec, v.vec)

    #todo: cross product
    def cross(self,v):
        return Vector3(self.y*v.z - self.z*v.y, self.z*v.x - self.x*v.z, self.x*v.y - self.y*v.x)

    @staticmethod
    def from_array(array):
        return Vector3(array[0],array[1],array[2])

    def __add__(self,rhs):
        return Vector3.from_array(self.vec + rhs.vec)

    def __sub__(self,rhs):
        return Vector3.from_array(self.vec - rhs.vec)
    
    def __mul__(self,rhs):
        if not isinstance(rhs,Vector3):
            return self.multiply(rhs)
        return Vector3.from_array(self.vec - rhs.vec)
    
    def __div__(self,rhs):
        return Vector3.from_array(self.vec / rhs)

    def __repr__(self):
        return f"({self.x}, {self.y}, {self.z})"

    def swap(self,index1,index2):
        self.vec[index1],self.vec[index2] = self.vec[index2],self.vec[index1] 

    def multiply(self,scalar):
        return Vector3.from_array(self.vec * scalar)
        
    def distance_to(self,v):
        pass

    def magnitude(self):
        return np.sqrt(np.sum(np.power(self.vec,2)))

    def normalized(self):
        return Vector3.from_array(self.vec / self.magnitude())
    
    def direction_to(self,v):
        return (v-self).normalized()
    
    def to_array(self):
        return self.vec
    
    def homogeneous(self):
        return Vector4(self.x, self.y, self.z, 1)
    
    def __getattr__(self, name):
        # Convienent way to get elements
        if name == 'x':
            return self.vec[0]
        if name == 'y':
            return self.vec[1]
        if name == 'z':
            return self.vec[2]

class Vector4:
    def __init__(self, x, y, z, a):
        self.vec = np.array([x, y, z, a])

    def __eq__(self, other):
        return np.array_equal(self.vec, other.vec)

    def __hash__(self):
        return hash(tuple(self.vec))

    def __add__(self, other):
        return Vector4(self.vec + other.vec)

    def __sub__(self, other):
        return Vector4(self.vec - other.vec)

    def __mul__(self, scalar):
        return Vector4(self.vec * scalar)

    def dot(self, v):
        return self.vec[0]*v.vec[0] + self.vec[1]*v.vec[1] + self.vec[2]*v.vec[2] + self.vec[3]*v.vec[3]

    @staticmethod
    def from_array(array):
        return Vector4(array[0],array[1],array[2],array[3])
    
    def to_array(self):
        return self.vec
    
    @staticmethod
    def zeros():
        return Vector4(0, 0, 0, 0)
    
    def magnitude(self):
        return np.linalg.norm(self.vec)

    def normalized(self):
        return Vector4(self.vec / self.magnitude())
    
    def remove_W(self):
        if (self.vec[3] == 0):
            return Vector3(self.vec[0], self.vec[1], self.vec[2])   
        return Vector3(self.vec[0]/self.vec[3], self.vec[1]/self.vec[3], self.vec[2]/self.vec[3])

    def __repr__(self):
        return f"({self.vec[0]}, {self.vec[1]}, {self.vec[2]}, {self.vec[3]})"

class Vector4x4:
    def __init__(self, rows=None):
        if rows is None:
            self.mat = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        else: 
            self.mat = rows
            
    def __eq__(self, other):
        return np.array_equal(self.mat, other.mat)

    def __hash__(self):
        return hash(tuple(map(tuple, self.mat)))
    
    def __add__(self, other):
        if isinstance(other, Vector4x4):
            
            return Vector4x4(self.mat + other.mat)

    def __mul__(self, other):
        result = np.linalg.multi_dot([self.to_array(), other.to_array()])
        if isinstance(other, Vector4):
            return Vector4.from_array(result)
        elif isinstance(other, Vector4x4):
            return Vector4x4(result)
    
    @staticmethod
    def from_array(array):
        return Vector4x4(array)
    
    def to_array(self):
        return self.mat
    
    def transpose(self):
        transp = np.transpose(self.to_array())
        return Vector4x4(transp)

    def __repr__(self):
        return f"{self.mat[0]}\n{self.mat[1]}\n{self.mat[2]}\n{self.mat[3]}"
 




