
from transform import Transform
from vector import Vector3

class Light ():
    def __init__(self):
        self.transform = Transform()
        self.intensity = 1.0
        self.color = Vector3(1.0, 1.0, 1.0)
        
    def apply_with_diffuse(self, diffuse:Vector3):
        return Vector3(self.color.x * diffuse.x, self.color.y * diffuse.y, self.color.z * diffuse.z)

class PointLight (Light):
    def __init__(self, intensity, color):
        super().__init__()
        self.intensity = intensity
        self.color = Vector3.from_array(color)
        