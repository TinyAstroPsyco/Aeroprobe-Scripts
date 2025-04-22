import pupil_apriltags as apriltag
import cv2 as cv
import os
import numpy as np
import math
import csv
import time

# Loading the intrinsics
root = 'C:/Users/pooji/Downloads/Aeroprobe Testing - 0612/Image Distance Analysis Scripts/Intrinsic_Parameters_Repair_2.npz'
intrinsics = np.load(root,mmap_mode = 'r')
camera_matrix = intrinsics['camera_matrix']
dist_coeffs = intrinsics['dist_coeff']
print(f' Camera Matrix : \n {camera_matrix} \n dist = {dist_coeffs}')


# Function to append the altitude values to a csv
def add_alt(altitude):
     with open('estimated Altitiude-3.csv', 'a', newline= '') as csvfile:
        fieldnames = ['Altitude']
        writer = csv.writer(csvfile)
        # If the file is empty, write the header first
        csvfile.seek(0, 2)  # Move the cursor to the end of the file
        if csvfile.tell() == 0:
                    writer.writerows(['Altitude'])

        # Write the altitude values
        writer.writerows(altitude)

# Function to get the distance form the centre of the AprilTags
def get_tag_to_tag_distance(tag_1, tag_2, image, tag_side_length_from_image, tag_size):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    tags = detector.detect(gray_image)
  
    tag_1_center = tag_2_center = None
    tag_1_x = tag_1_y = tag_2_x = tag_2_y = None
    distance = None
    distance_pixels = None

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
        cv.line(image, (tag_1_x, tag_1_y), (tag_2_x, tag_2_y), (255, 0, 0), 4, 1)

        distance_pixels = math.dist(tag_1_center, tag_2_center) 
        distance = math.dist(tag_1_center, tag_2_center) # Extracting the centre to centre distance of the april tags in pixels
        # distance = tag_size/distance 
        print(f'Centre to centre distance = {distance} pixels \n Tag side length = {tag_side_length_from_image} pixels')
        distance = distance - tag_side_length_from_image*2 #Substracting one more side length because we are measuring the centre to centre distance of the tag.

        distance = distance/tag_side_length_from_image # Converting the pixel centre to centre distances in number of side pixel length of the tag
        # Draw a filled rectangle on the image
        pts = np.array([[0, 0], [300, 0], [300, 100], [0, 100]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        # cv.fillPoly(image, [pts], (225, 225, 225))
        # cv.circle(image, (50,50),30,(225,225,225),1,2,0)
        cv.putText(image, ("Tag to Tag :"+str(int(distance))) + "Feet", (10, 30), 2, 1.5, (0, 255, 255), 1, 1, False)
        cv.putText(image, ("Tag to Tag :"+str(int(distance_pixels))) + "Pixels", (10, 80), 2, 1.5, (0, 255, 0), 2, 1, False)
        print(f'Distance between tags: {distance}')
    else:
        print('Tags not found')

    
    return image, distance, distance_pixels


def get_camera_distance(tag_id, image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY) #Conevert to gray scale
    tags = detector.detect(gray_image) #Detecting the tags in the gray image 
    print(f'Tags{len(tags)}') #Printing the number of tags in the image
    # tag_side_length_from_image = 0
    
    #iterating thorught the tags in the image
    for tag in tags:
        print(f'The tag ids detected are : {tag.tag_id}')
        if tag.tag_id == tag_id:
            # Known size of the AprilTag side (in meters)
            
            tag_size = 0.170 #Tag know size in meters
            
            object_points = np.array([
                [-tag_size/2, -tag_size/2, 0],
                [ tag_size/2, -tag_size/2, 0],
                [ tag_size/2,  tag_size/2, 0],
                [-tag_size/2,  tag_size/2, 0]
            ])

            image_points = tag.corners 
            print(f'Tag Corners are : \n{image_points}')
            p1 = image_points[0]
            p2 = image_points[3]
            cv.circle(image, (int(p1[0]), int(p1[1])), 2,(0,255,0),2,1)
            cv.circle(image, (int(p2[0]), int(p2[1])), 2,(0,255,0),2,1)
         

            tag_side_length_from_image = (math.dist(p2,p1)) #Calculating the tag side length in pixels
            print(f'Lenght of the side in pixels = {(tag_side_length_from_image)}')
            
            #Annotating the tag corners in the image (Will only do this for one tag  - Hard code)
            for image_point in image_points:
                print(f'Image Point :: :: : {image_point}')
                cv.circle(image, (int(image_point[0]), int(image_point[1])), 1,(225,0,0),1,1)
            
            # SolvePnP to find the pose
            retval, rvec, tvec = cv.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
            if retval:
                distance = np.linalg.norm(tvec[0][0])
                # breakpoint()
                if tvec is not None:
                    print(f'Tvecs >>>>>>> {tvec}')
                    # break
                    # cv.putText( cv.putText(image, str(tvec), (600,120), 1, 2, (0, 255, 0), 1, 1, False))
                cv.putText(image, ("Camera Distance to tag : " + str(int(distance))+ "Meters"), (5,120), 1, 2, (0, 255, 0), 1, 1, False)
                # save_image(output_imagePath,filename, image)
                print(f'Distance from camera to tag {tag_id}: {distance} meters')
                print(f'Number of pixes for one side of the the tag :: {tag_side_length_from_image}')
                return distance, tag_side_length_from_image, tag_size
        
    
    print(f'Tag {tag_id} not found')
    return None,None,None





# We use the thin lens method
def get_camera_distance_thin_lens(tag_id, image):
    # Convert the image to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    camera_params, dist_coeffs = load_intrinsics()
    fx = camera_params[0][0]
    fy = camera_params[1][1]
    cx = camera_params[0][2]
    cy = camera_params[1][2]
    cam_para_list = [fx, fy, cx, cy]

    # print(f'camera_parameters >> {cam_para_list}')

    tag_size = 0.13826
    tags = detector.detect(gray_image, estimate_tag_pose=True,camera_params= cam_para_list, tag_size=tag_size) # Instancing the detector and detecting the tags from the image

    if not tags:
        print("No tags detected in the image. Stuck here")
        return None, None, None, None, None, None

    for tag in tags:
        if tag.tag_id == tag_id:
            tag_size = 0.13826  # Physical side length of the tag in meters

            image_points = tag.corners
            tag_angle_vector = []
            # tag_angle =
       
            # Calculate average tag side length in pixels
            # tag_side_length_from_image = (
            #     math.dist(image_points[0], image_points[1]) +q
            #     math.dist(image_points[1], image_points[2]) +
            #     math.dist(image_points[2], image_points[3]) +
            #     math.dist(image_points[3], image_points[0])
            # ) / 4

            # Focal length in pixels (from camera matrix)
            focal_length = camera_matrix[0, 0]  # fx from camera intrinsics
            tag_side_length_from_image = math.dist(image_points[0], image_points[3])
            cv.circle(image, (int(image_points[0][0]), int(image_points[0][1])), 2,(225,0,0),1,1)
            cv.circle(image, (int(image_points[3][0]), int(image_points[3][1])), 2,(225,0,0),1,1)
            # cv.imshow("Tag Point", image)
            # cv.waitKey(1000)

            # Apply the thin lens formula
            if tag_side_length_from_image > 0:
                distance = (focal_length * tag_size) / tag_side_length_from_image
                distance = float(distance)

                # print(f"Focal Length (pixels): {focal_length}")
                # print(f"Tag Side Length (pixels): {tag_side_length_from_image:.2f}")
                # print(f"Physical Tag Size (meters): {tag_size}")
                # print(f"Estimated Distance using Thin Lens Formula: {distance:.3f} meters")
                print(f'Tag :: {tag}') 
                half_size = tag_size / 2.0
                object_points = np.array([
                    [-half_size,  half_size, 0],  # Top-left
                    [ half_size,  half_size, 0],  # Top-right
                    [ half_size, -half_size, 0],  # Bottom-right
                    [-half_size, -half_size, 0],  # Bottom-left
                ], dtype=np.float32)

                success, rvec, tvec = cv.solvePnP(objectPoints = object_points, imagePoints = image_points, cameraMatrix= camera_params, distCoeffs = dist_coeffs )
                if success:
                    # Convert rotation vector to rotation matrix
                    R, _ = cv.Rodrigues(rvec)

                    # Extract Euler angles from rotation matrix
                    # OpenCV uses a right-handed coordinate system:
                    # Yaw (Z), Pitch (Y), Roll (X)

                    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
                    singular = sy < 1e-6

                    if not singular:
                        x = np.arctan2(R[2, 1], R[2, 2])  # Roll
                        y = np.arctan2(-R[2, 0], sy)      # Pitch
                        z = np.arctan2(R[1, 0], R[0, 0])  # Yaw
                    else:
                        x = np.arctan2(-R[1, 2], R[1, 1])
                        y = np.arctan2(-R[2, 0], sy)
                        z = 0

                    # Convert from radians to degrees
                    roll  = np.degrees(x)
                    pitch = np.degrees(y)
                    yaw   = np.degrees(z)

                    print(f"Euler Angles (degrees):\nRoll (X): {roll:.2f}, Pitch (Y): {pitch:.2f}, Yaw (Z): {yaw:.2f}")
                    euler_text = f"Roll: {roll:.2f}, Pitch: {pitch:.2f}, Yaw: {yaw:.2f}"
                    cv.putText(
                        image,                      # the image you're drawing on
                        euler_text,                 # the text to draw
                        (20, 150),                  # bottom-left corner of the text
                        cv.FONT_HERSHEY_SIMPLEX,    # font
                        0.7,                        # font scale
                        (0, 255, 0),                # color (green)
                        2,                          # thickness
                        cv.LINE_AA                  # anti-aliased line
                                    )

                return distance, tag_side_length_from_image, tag_size, roll, pitch, yaw

    print(f"Tag {tag_id} not found in the detected tags.")
    return None, None, None, None, None, None


def load_intrinsics():
    root = 'C:/Users/pooji/Downloads/Aeroprobe Testing - 0612/Image Distance Analysis Scripts/Intrinsic_Parameters_Repair_2.npz'
    intrinsics = np.load(root,mmap_mode = 'r')
    camera_matrix = intrinsics['camera_matrix']
    dist = intrinsics['dist_coeff']
    print(f' Camera Matrix : \n {camera_matrix} \n dist = {dist}')
    return camera_matrix, dist

# Function to save the image
def save_image(output_imagePath, filename, image):
    os.makedirs(output_imagePath, exist_ok=True)
    cv.imwrite(output_imagePath + filename + '.jpg', image)
    cv.waitKey(100)  # Adjust the display duration (in milliseconds)
    cv.destroyAllWindows()

# Function to save the csv file
def save_csv(path, imageData):
    with open(path, 'w', newline= '') as csvfile:
        altitudeWriter = csv.writer(csvfile, delimiter= ',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        altitudeWriter.writerow(["Image Name", "Tag c2c", "Altitude", "c2cPixelDistance", "Roll", "Pitch", "Yaw"])
        
        for data in imageData:
            altitudeWriter.writerow([data[2], data[1],data[0], data[3], data[4], data[5], data[6]])

                            #############  Main Code  ##############


#Input undistorted image path
image_dir = 'C:/Users/pooji/Downloads/Aeroprobe Testing - 0612/Image Distance Analysis Scripts/March 2nd with angles/undistorted-3/'

# Output image path for images with the height overlayed
output_imagePath = 'C:/Users/pooji/Downloads/Aeroprobe Testing - 0612/Image Distance Analysis Scripts/March 2nd with angles/Altitude Estimate-3/'

detector = apriltag.Detector(families="tag36h11")

alt_estimates = []

for filename in os.listdir(image_dir):
    if filename.endswith((".jpg", ".jpeg", ".png", ".JPEG", ".JPG")):
        image_path = os.path.join(image_dir, filename)
        print(f'Processing image: {image_path}')        
        image = cv.imread(image_path)
        if image is not None:
            # pts = np.array([[0, 0], [500, 0], [300, 100], [0, 100]], np.int32)
            # pts = pts.reshape((-1, 1, 2))
            # # cv.fillPoly(image, [pts], (225, 225, 225))
            distance_to_tag, tag_side_length_from_image, tag_size, roll, pitch, yaw = get_camera_distance_thin_lens(0, image)
            image , c2cDistance, c2cPixelDistance = get_tag_to_tag_distance(0, 1, image, tag_side_length_from_image, tag_size)
            if distance_to_tag is not None:
                alt_estimates.append((distance_to_tag, c2cDistance, filename, c2cPixelDistance, roll, pitch, yaw))
            if distance_to_tag is not None:
            #   cv.putText(image, f'Distance to tag 0: {distance_to_tag:.2f}m', (50, 50), 1, 4, (0, 255, 0), 4, 1, False)
                cv.putText(image, ("Camera Distance to tag : " + str(float(round(distance_to_tag,2)))+ "Meters"), (5,120), 1, 2, (0, 255, 0), 2, cv.LINE_AA, False)
                save_image(output_imagePath, filename, image)
        else:
            print(f"Failed to load or process image: {image_path}")
   
print(f'final altitude estimates = {alt_estimates}')
# add_alt(alt_estimates)

save_csv("C:/Users/pooji/Downloads/Aeroprobe Testing - 0612/Image Distance Analysis Scripts/March 2nd with angles/Altitude Estimate-3/altitudeEstimate_hallway_tilted_corrected_for_angle-03.csv", alt_estimates)

cv.destroyAllWindows()
print("Success!!")