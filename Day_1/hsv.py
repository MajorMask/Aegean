import cv2
import numpy as np

# Load an image or use a video capture
image = cv2.imread('sea_with_buoys.jpg')  # Replace with your image
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define HSV range for red or green buoys (tweak as needed)
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])

# Create a mask and apply it
mask = cv2.inRange(hsv, lower_red, upper_red)
result = cv2.bitwise_and(image, image, mask=mask)

# Display
cv2.imwrite('original.jpg', image)
cv2.imwrite('red_buoy_mask.jpg', mask)
cv2.imwrite('detected_buoy.jpg', result)

contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 70:  # Ignore tiny noise
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Save the image with bounding boxes
cv2.imwrite('bounding_boxes.jpg', image)

print("Images saved: original.jpg, red_buoy_mask.jpg, detected_buoy.jpg, bounding_boxes.jpg")
cv2.waitKey(0)
cv2.destroyAllWindows()

