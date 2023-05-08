from PIL import Image
import numpy
import cv2
import matplotlib.pyplot as plt

img =cv2.imread('./cut_csd/id.jpg')
plt.imshow(img[:,:,1],cmap='gray')
plt.show()
