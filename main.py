import cv2  # Import OpenCV library for computer vision tasks
import os   # Import OS module to automatically create output folders
import numpy as np # Import NumPy for arry manipulations

# Define the input image path and the output folder path
input_path = 'Life in a petri dish.jpg'
output_dir = 'outputs'

# Create the outputs directory if it does not already exist
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------
# STEP 1: Load Image using OpenCV
# ---------------------------------------------------------
# cv2.imread loads the image file from disk as a NumPy array 
img = cv2.imread(input_path)

# Verify if the image was successfully loaded
if img is None:
    print(f"Error: Could not load image from '{input_path}'. Check file name/path.")
    exit()

# Save output 1: Original image 
cv2.imwrite(os.path.join(output_dir, '1_original.jpg'), img)

# ---------------------------------------------------------
# STEP 2: Crop the original image 
# ---------------------------------------------------------
# Get image height and width to calculate bounding coordinates
height, width = img.shape[:2]

# Modified: Crop only 2% to preserve full petri dish rim
start_y, end_y = int(height * 0.02), int(height * 0.98)
start_x, end_x = int(width * 0.02), int(width * 0.98)
cropped = img[start_y:end_y, start_x:end_x]

# Save output 2: Cropped image 
cv2.imwrite(os.path.join(output_dir, '2_cropped.jpg'), cropped)

# ---------------------------------------------------------
# STEP 3: Convert the image to Grayscale 
# ---------------------------------------------------------
# Convert BGR (Blue-Green-Red) image to single-channel Grayscale for shape detection 
gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

# Save output 3: Grayscale image 
cv2.imwrite(os.path.join(output_dir, '3_grayscale.jpg'), gray)

# ---------------------------------------------------------
# STEP 4: Convert the image to HSV color space 
# ---------------------------------------------------------
# Convert BGR to HSV (Hue, Saturation, Value) color space for color filtering 
hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

# Save output 4: HSV image 
cv2.imwrite(os.path.join(output_dir, '4_hsv.jpg'), hsv)

# ---------------------------------------------------------
# STEP 5: Apply Noise Filtering / Blur 
# ---------------------------------------------------------
# GaussianBlur smooths high-frequency noise using a 5x5 kernel matrix 
filtered = cv2.GaussianBlur(gray, (5, 5), 0)

# Save output 5: Filtered image 
cv2.imwrite(os.path.join(output_dir, '5_filtered.jpg'), filtered)

# ---------------------------------------------------------
# STEP 6: Perform Canny Edge Detection 
# ---------------------------------------------------------
#Canny detects edges using low (50) and high (150) hysteresis thresholds
edges = cv2.Canny(filtered, 50, 150)

#Save output 6: canny edges
cv2.imwrite(os.path.join(output_dir, '6_canny_edges.jpg'), edges)

# ---------------------------------------------------------
# STEP 7: Extract and Draw Contours
# ---------------------------------------------------------
# findContours locates continuous boundary points along Canny edges
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#Save output 6: canny edges
contour_img = cropped.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
cv2.imwrite(os.path.join(output_dir, '7_contours.jpg'), contour_img)

# ---------------------------------------------------------
# STEP 8 & 9: Geometric Detections & Final Image Overlay
# ---------------------------------------------------------
# Create final canvas to draw all required geometric detections
final_img = cropped.copy()

print("\n================ GEOMETRIC ANALYSIS RESULTS ================")

# ---------------------------------------------------------
# STEP 8a: Hough Line Detection
# ---------------------------------------------------------
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold = 100, minlineLength = 50, maxLineGap = 10)
if lines is not None:   
    for line in lines[:5]:  # Draw first 5 lines in red
        x1, y1, x2, y2 = line[0]  
        cv2.line(final_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  
    print(f"[Hough Lines] Detected {len(lines)} line segments.")

# ---------------------------------------------------------
# STEP 8b: Hough Circle Detection
# --------------------------------------------------------- 
circles = cv2.HoughCircles(filtered, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=50, param2=30, minRadius=50, maxRadius=400)   
if circles is not None:   
    circles = np.uint16(np.around(circles))  # Convert circle values to integers
    for c in circles[0, :3]:  # Draw top 3 detected circles in yellow
        center_x, center_y, radius = c[0], c[1], c[2]  
        cv2.circle(final_img, (center_x, center_y), radius, (0, 255, 255), 2)  
        print(f"[Hough Circle] Center: ({center_x}, {center_y}), Radius: {radius} px")  
# ---------------------------------------------------------
# STEP 8c: Contour Loop: Bounding Box, Min Area Rect, Enclosing Circle, Polygon Approx, Convex Hull
# --------------------------------------------------------- 
valid_colony_count = 0

for i, cnt in enumerate(contours): #Loop through every detected contour 
    area = cv2.contourArea(cnt) #Calculate area in pixels
    if area < 300: #Filter out small noise artifacts
        continue

    valid_colony_count += 1
    # 1. Bounding Box (Blue) 
    bx, by, bw, bh = cv2.boundingRect(cnt)   
    cv2.rectangle(final_img, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)   

    # 2. Minimum Area Rectangle (Magenta) 
    rect = cv2.minAreaRect(cnt)   
    box_points = cv2.boxPoints(rect)   
    box_points = np.int32(box_points)  # Convert coordinates to integer 
    cv2.drawContours(final_img, [box_points], 0, (255, 0, 255), 2)   

    # 3. Minimum Enclosing Circle (Cyan) 
    (cx, cy), cradius = cv2.minEnclosingCircle(cnt)   
    cv2.circle(final_img, (int(cx), int(cy)), int(cradius), (255, 255, 0), 2)   

    # 4. Polygon Approximation (White) 
    epsilon = 0.02 * cv2.arcLength(cnt, True)  # Perimeter tolerance 
    approx = cv2.approxPolyDP(cnt, epsilon, True)   
    cv2.drawContours(final_img, [approx], -1, (255, 255, 255), 2)   

    # 5. Convex Hull (Orange) 
    hull = cv2.convexHull(cnt)  # Outer convex boundary 
    cv2.drawContours(final_img, [hull], -1, (0, 165, 255), 2)   

    # ---------------------------------------------------------
    # Step 10: Print Text Analysis and Terminals
    # --------------------------------------------------------- 
    print(f"\n[Colony {valid_colony_count}]")   
    print(f"  - Area: {area:.1f} sq px")   
    print(f"  - Bounding Box: x={bx}, y={by}, w={bw}, h={bh}")   
    print(f"  - Min Enclosing Circle: Center=({int(cx)}, {int(cy)}), Radius={int(cradius)} px")   
    print(f"  - Polygon Vertices: {len(approx)}")   

# Save output 8: Final detection overlay image 
cv2.imwrite(os.path.join(output_dir, '8_final_detection_result.jpg'), final_img)   

print("\n============================================================")   
print(f"TOTAL COLONIES DETECTED: {valid_colony_count}")   
print("All 8 required output images successfully saved in '/outputs/' directory!")   
print("============================================================\n")   