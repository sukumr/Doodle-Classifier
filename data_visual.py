import pygame
import numpy as np

pygame.init()
screen = pygame.display.set_mode((280, 280))
pygame.display.set_caption("Doodle Classifier")

imgdata = np.load('cats1000.npy')

data = []

def gray(im):
    w, h = im.shape
    #np.uint8: Unisigned integer(0 to 255)
    ret = np.empty((w, h, 3), dtype=np.uint8) # 28 arrays of (28 rows X 3 columns)
    # copying image data in to the arrays
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = (255 - im) # inverting a color
    return ret

for i in range(100):
    sample = imgdata[i:i+1, :]
    # reshape from (1 X 784) to (28 X 28)
    arr = sample.reshape(28,28)
    # swapping axes
    arr = np.swapaxes(arr, 0, 1)
    arr = gray(arr)
    data.append(arr)

close = False

while not close:
   
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            close = True
    
    screen.fill('white')

    for i in range(len(data)):
        # Copying an array to a new surface
        surf = pygame.surfarray.make_surface(data[i])
        # surf = pygame.pixelcopy.make_surface(data[i])
        
        # surf = pygame.transform.scale2x(surf)
        # x = 28 * (i % 10) * 2
        # y = 28 * (i // 10) * 2

        x = 28 * (i % 10)
        y = 28 * (i // 10)

        
        screen.blit(surf, (x, y))
 
    pygame.display.flip()

pygame.quit()
