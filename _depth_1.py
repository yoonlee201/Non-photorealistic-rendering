import numpy as np

from screen import Screen
from camera import PerspectiveCamera,OrthoCamera
from mesh import Mesh
from renderer import Renderer
from light import PointLight



if __name__ == '__main__':
    screen = Screen(1000,1000)

    camera = PerspectiveCamera(-1.0, 1.0, -1.0, 1.0, -1.0, -20)
    camera.transform.set_position(0, 0, 4)


    color_1 = [255, 246, 208]
    mesh_1 = Mesh.from_stl("suzanne.stl", np.array(color_1)/225,\
        np.array([1.0, 1.0, 1.0]),0.05,1.0,0.2,100)
    mesh_1.transform.set_rotation(15, -20, 0)
    mesh_1.transform.set_position(1,0,1)

    color_2 = [27, 71, 92]
    mesh_2 = Mesh.from_stl("unit_cube.stl", np.array(color_2)/225,\
        np.array([1.0, 1.0, 1.0]),0.05,1.0,0.2,100)
    mesh_2.transform.set_position(-1,-2,-1.5)

    color_3 = [180, 189, 98]
    mesh_3 = Mesh.from_stl("unit_sphere.stl", np.array(color_3)/225,\
        np.array([1.0, 1.0, 1.0]),0.05,0.8,0.2,100)
    mesh_3.transform.set_position(-2, 2, 0)

    light = PointLight(50.0, np.array([1, 1, 1]))
    light.transform.set_position(-4, 3, 4)

    renderer = Renderer(screen, camera, [mesh_1,mesh_2,mesh_3], light)
    renderer.render("paint",[80,80,80], [0.2, 0.2, 0.2])

    screen.show()