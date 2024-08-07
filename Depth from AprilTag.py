import pupil_apriltags as apriltag
import cv2 as cv
import os
import numpy as np
import math


# Loading the intrinsics
root = 'C:/Users/pooji/Downloads/Aeroprobe Testing - 0612/Image Distance Analysis Scripts/Intrinsic_Parameters.npz'
intrinsics = np.load(root,mmap_mode = 'r')
camera_matrix = intrinsics['camera_matrix']
dist_coeffs = intrinsics['dist_coeff']
print(f' Camera Matrix : \n {camera_matrix} \n dist = {dist_coeffs}')

# Function to get the distance form the centre of the AprilTags
def get_tag_to_tag_distance(tag_1, tag_2, image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    tags = detector.detect(gray_image)
    # print(f'Tags::::::::::----:::::::{len(tags)}')
    
    tag_1_center = tag_2_center = None
    tag_1_x = tag_1_y = tag_2_x = tag_2_y = None
    
    for tag in tags:
        if tag.tag_id == tag_1:
            tag_1_center = tag.center
            tag_1_x, tag_1_y = int(tag_1_center[0]), int(tag_1_center[1])
        elif tag.tag_id == tag_2:
            tag_2_center = tag.center
            tag_2_x, tag_2_y = int(tag_2_center[0]), int(tag_2_center[1])
    
    if tag_1_center is not None and tag_2_center is not None:
        text_x = int(abs((tag_2_x + tag_1_x) / 2))
        text_y = int(abs((tag_2_y + tag_1_y) / 2))
        # cv.line(image, (tag_1_x, tag_1_y), (tag_2_x, tag_2_y), (255, 0, 0), 4, 1)
        distance = math.dist(tag_1_center, tag_2_center)
         # Draw a filled rectangle on the image
        pts = np.array([[0, 0], [300, 0], [300, 100], [0, 100]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv.fillPoly(image, [pts], (225, 225, 225))
        # cv.circle(image, (50,50),30,(225,225,225),1,2,0)
        cv.putText(image, ("Tag to Tag :"+str(int(distance))), (10, 20), 1, 1, (0, 45, 200), 1, 1, False)
        print(f'Distance between tags: {distance}')
    else:
        print('Tags not found')

    return image


def get_camera_distance(tag_id, image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    tags = detector.detect(gray_image)
    print(f'Tags::::::::::----:::::::{len(tags)}')
    for tag in tags:
        print(f'The tag ids detected are : {tag.tag_id}')
        if tag.tag_id == tag_id:
            # Known size of the AprilTag side (in meters)
            tag_size = 0.175
            object_points = np.array([
                [-tag_size/2, -tag_size/2, 0],
                [ tag_size/2, -tag_size/2, 0],
                [ tag_size/2,  tag_size/2, 0],
                [-tag_size/2,  tag_size/2, 0]
            ])

            image_points = tag.corners
            for image_point in image_points:
                print(f'Image Point :: :: : {image_point}')
                cv.circle(image, (int(image_point[0]), int(image_point[1])), 1,(225,0,0),1,1)
            
            # SolvePnP to find the pose
            retval, rvec, tvec = cv.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
            if retval:
                distance = np.linalg.norm(tvec)
                cv.putText(image, ("Camera Distance to tag : " + str(float(distance))), (10,70), 2, 1, (0, 0, 255), 1, 1, False)
                save_image(output_imagePath,filename, image)
                print(f'Distance from camera to tag {tag_id}: {distance} meters')
                return distance
    
    print(f'Tag {tag_id} not found')
    return None

def save_image(output_imagePath, filename, image):
    os.makedirs(output_imagePath, exist_ok=True)
    cv.imwrite(output_imagePath + "Annotated_" + filename + '.jpg', image)
    cv.waitKey(100)  # Adjust the display duration (in milliseconds)
    cv.destroyAllWindows()

image_dir = 'C:/Users/pooji/Downloads/Aeroprobe Testing - 0612/Image Distance Analysis Scripts/Distance Measurement Images/Undistorted'
output_imagePath = 'C:/Users/pooji/Downloads/Aeroprobe Testing - 0612/Image Distance Analysis Scripts/Distance Measurement Images/Distance Measurement/'
detector = apriltag.Detector(families="tag36h11")

for filename in os.listdir(image_dir):
    if filename.endswith((".jpg", ".jpeg", ".png", ".JPEG", ".JPG")):
        image_path = os.path.join(image_dir, filename)
        print(f'Processing image: {image_path}')
        
        image = cv.imread(image_path)
        if image is not None:
            image = get_tag_to_tag_distance(0, 1, image)
            distance_to_tag = get_camera_distance(0, image)
            # if distance_to_tag is not None:
            #     cv.putText(image, f'Distance to tag 0: {distance_to_tag:.2f}m', (50, 50), 1, 4, (0, 255, 0), 4, 1, False)
            save_image(output_imagePath, filename, image)
        else:
            print(f"Failed to load or process image: {image_path}")

cv.destroyAllWindows()
print("Success!!")
