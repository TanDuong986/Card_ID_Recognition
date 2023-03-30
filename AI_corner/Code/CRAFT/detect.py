import cv2
import matplotlib.pyplot as plt
from model import CraftTextDetector

# Initialize the detector with the path to the pre-trained CRAFT model
detector = CraftTextDetector('craft_mlt_25k.pth', link_refinement=True)

# Load the input image
image = cv2.imread('input_image.jpg')

# Detect text in the image
boxes, polys = detector.detect_text('input_image.jpg')

# Draw the bounding boxes on the input image
for box in boxes:
    x1, y1, x2, y2, x3, y3, x4, y4 = box.astype(int)
    cv2.rectangle(image, (x1, y1), (x3, y3), (0, 0, 255), 2)

# Show the result
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
