import cv2
import math

# Initialize a list to store points
points = []

image_directory = "C:/Users/pooji/Downloads/Aeroprobe Testing - 0612/Annotated Images/"

# Define the callback function for mouse events
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Store the point coordinates
        points.append((x, y))

        # Draw a small circle at the clicked point
        cv2.circle(img_resized, (x, y), 2, (0, 255, 0), -1)

        # If there are at least two points, draw a line between the last two points
        if len(points) > 1:
            cv2.line(img_resized, points[-2], points[-1], (255, 0, 0), 1)

            # Calculate the Euclidean distance between the last two points
            dx = points[-1][0] - points[-2][0]
            dy = points[-1][1] - points[-2][1]
            distance = math.sqrt(dx**2 + dy**2)

            # Display the distance
            mid_x = (points[-2][0] + points[-1][0]) // 2
            mid_y = (points[-2][1] + points[-1][1]) // 2
            cv2.putText(img_resized, f"{distance:.2f}", (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)

        # Refresh the image window with the new drawing
        cv2.imshow('Image', img_resized)

# Load the image
img_path = 'C:/Users/pooji/Downloads/Aeroprobe Testing - 0612/Final Script/Annotated_nvcamtest_6357_s00_00048.jpg.jpg'
img = cv2.imread(img_path)

# Resize the image to fit within the screen dimensions
screen_width = 1344  # Use this value, it worked for me
screen_height = 756  # Use this value, it worked for me
img_height, img_width = img.shape[:2]

if img_width > screen_width or img_height > screen_height:
    scaling_factor = min(screen_width / img_width, screen_height / img_height)
    img_resized = cv2.resize(img, (int(img_width * scaling_factor), int(img_height * scaling_factor)))
else:
    img_resized = img

# Call back
cv2.imshow('Image', img_resized)
cv2.setMouseCallback('Image', draw_circle)


while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key to exit
        break
    elif key == ord('s') or key == ord('S'):
        image_name = input("Name of the Image : ")
        cv2.imwrite(image_directory + image_name + '.jpg', img_resized)
        print(f'Image saved as {image_name}.jpg')
cv2.destroyAllWindows()
