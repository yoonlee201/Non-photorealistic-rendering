
import numpy as np
import pygame
from vector import Vector3

class Screen:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        pygame.init()
        self.screen = pygame.display.set_mode([width, height])

    def ratio(self):
        return self.width / self.height
    
    
    def inverse_project_point(self, p: Vector3):
        x = p.x / self.width * 2 - 1
        y = p.y / self.height * 2 - 1
        z = p.z * 2 - 1
        return Vector3(x, y, z)
    
    def project_point(self, p: Vector3):
        # v = self.zero_and_one_coordinates(p)
        x = int((1 + p.x) / 2  * self.width / self.ratio())
        y = int((1 + p.y) / 2  * self.height )
        z = (1 + p.z) / 2 
        return (x,y,z)

    def draw(self, buf: np.ndarray):
        """Takes a buffer of 8-bit RGB pixels and puts them on the canvas.
        buf should be a ndarray of shape (height, width, 3)"""
        # Make sure that the buffer is HxWx3
        # if buf.shape != (self.height, self.width, 3):
        #     raise Exception("buffer and screen not the same size")

        # Flip buffer to account for 0,0 in bottom left while plotting, but 0,0 in top left in pygame
        buf = np.fliplr(buf)

        # The prefered way to accomplish this
        pygame.pixelcopy.array_to_surface(self.screen, buf)

        # An alternative (slower) way, but still valid
        # Iterate over the pixels and paint them
        # for x, row in enumerate(buf):
            # for y, pix in enumerate(row):
                # self.screen.set_at((x, y), pix.tolist())

        # Update the display
        pygame.display.flip()

    def show(self):
        """Shows the canvas"""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        from datetime import datetime
                        pygame.image.save(self.screen, datetime.now().strftime("./%m-%d-%Y_%H--%M--%S.png"))


        pygame.quit()
        
        
        
        