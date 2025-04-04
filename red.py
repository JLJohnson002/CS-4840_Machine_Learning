import cv2
import numpy as np
import os

folder_path = r"OriginlTrainImages/Death Star"
output_path = r"Red Circles"
os.makedirs(output_path, exist_ok=True)

def detect_red_circles(image_path, output_name):
    # Read image
    img = cv2.imread(image_path)
    output = img.copy()

    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define red color range in HSV (two ranges for light and dark red)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Blur to reduce noise before circle detection
    blurred = cv2.GaussianBlur(red_mask, (9, 9), 2)

    # Detect circles using Hough Transform
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=100, param2=30, minRadius=10, maxRadius=100)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for i, (x, y, r) in enumerate(circles):
            # Draw the circle and center
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.rectangle(output, (x-2, y-2), (x+2, y+2), (0, 128, 255), -1)

            # Crop and save circular region with transparency
            padding = 4
            height, width = img.shape[:2]

            x1 = max(x - r - padding, 0)
            y1 = max(y - r - padding, 0)
            x2 = min(x + r + padding, width)
            y2 = min(y + r + padding, height)
            crop = img[max(y1, 0):y2, max(x1, 0):x2]



            # Create circular mask
            offset_x = x - max(x1, 0)
            offset_y = y - max(y1, 0)

            height, width = crop.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(mask, (offset_x, offset_y), r + padding, 255, -1)
            # Apply mask to crop
            crop_bgra = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
            crop_bgra[:, :, 3] = mask  # Set alpha channel using mask

            # Save as transparent PNG
            crop_filename = os.path.join(output_path, f"{output_name}_circle_{i}.png")
            cv2.imwrite(crop_filename, crop_bgra)

            # Save output with drawn circles
            # result_path = os.path.join(output_path, f"{output_name}_detected.png")
            # cv2.imwrite(result_path, output)



# Process all images in folder
for file in os.listdir(folder_path):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        full_path = os.path.join(folder_path, file)
        name = os.path.splitext(file)[0]
        detect_red_circles(full_path, name)

print("\n✅ FINISHED\n")
