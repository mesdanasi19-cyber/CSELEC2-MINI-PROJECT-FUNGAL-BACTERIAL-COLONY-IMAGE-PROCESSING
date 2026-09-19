import cv2  # Import OpenCV library for computer vision tasks
import os   # Import OS module to automatically create output folders

# Define the input image path and the output folder path
input_path = 'Life in a petri dish.jpg'
output_dir = 'outputs'

# Create the outputs directory if it does not already exist
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------
# STEP 1: Load Image using OpenCV
# ---------------------------------------------------------
# cv2.imread loads the image file from disk as a NumPy array[cite: 1]
img = cv2.imread(input_path)

# Verify if the image was successfully loaded
if img is None:
    print(f"Error: Could not load image from '{input_path}'. Check file name/path.")
    exit()

# Save output 1: Original image[cite: 1]
cv2.imwrite(os.path.join(output_dir, '1_original.jpg'), img)

# ---------------------------------------------------------
# STEP 2: Crop the original image[cite: 1]
# ---------------------------------------------------------
# Get image height and width to calculate bounding coordinates
height, width = img.shape[:2]

# Crop 10% off each border to focus on the central petri dish using array slicing[cite: 1]
start_y, end_y = int(height * 0.10), int(height * 0.90)
start_x, end_x = int(width * 0.10), int(width * 0.90)
cropped = img[start_y:end_y, start_x:end_x]

# Save output 2: Cropped image[cite: 1]
cv2.imwrite(os.path.join(output_dir, '2_cropped.jpg'), cropped)

# ---------------------------------------------------------
# STEP 3: Convert the image to Grayscale[cite: 1]
# ---------------------------------------------------------
# Convert BGR (Blue-Green-Red) image to single-channel Grayscale for shape detection[cite: 1]
gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

# Save output 3: Grayscale image[cite: 1]
cv2.imwrite(os.path.join(output_dir, '3_grayscale.jpg'), gray)

# ---------------------------------------------------------
# STEP 4: Convert the image to HSV color space[cite: 1]
# ---------------------------------------------------------
# Convert BGR to HSV (Hue, Saturation, Value) color space for color filtering[cite: 1]
hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

# Save output 4: HSV image[cite: 1]
cv2.imwrite(os.path.join(output_dir, '4_hsv.jpg'), hsv)

# ---------------------------------------------------------
# STEP 5: Apply Noise Filtering / Blur[cite: 1]
# ---------------------------------------------------------
# GaussianBlur smooths high-frequency noise using a 5x5 kernel matrix[cite: 1]
filtered = cv2.GaussianBlur(gray, (5, 5), 0)

# Save output 5: Filtered image[cite: 1]
cv2.imwrite(os.path.join(output_dir, '5_filtered.jpg'), filtered)

print("Phase 1 completed successfully! Images 1 through 5 saved to '/outputs/'.")