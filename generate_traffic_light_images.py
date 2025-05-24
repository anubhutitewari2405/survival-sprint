import cv2
import numpy as np

# Image size
img_size = (64, 64, 3)

# RED Light Image
red_img = np.zeros(img_size, dtype=np.uint8)
cv2.circle(red_img, center=(32, 32), radius=20, color=(0, 0, 255), thickness=-1)  # BGR -> Red
cv2.imwrite('red_light.jpg', red_img)

# GREEN Light Image
green_img = np.zeros(img_size, dtype=np.uint8)
cv2.circle(green_img, center=(32, 32), radius=20, color=(0, 255, 0), thickness=-1)  # BGR -> Green
cv2.imwrite('green_light.jpg', green_img)

print("Images generated: red_light.jpg and green_light.jpg")
