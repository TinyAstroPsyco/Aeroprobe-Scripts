import cv2 as cv
import numpy as np
import os

def calibrate_camera(image_dir, chessboard_size, criteria, object_points, img_points):
    # Prepare object points
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    # Iterate through all image files in the directory
    for filename in os.listdir(image_dir):
        if filename.endswith((".jpg", ".jpeg", ".png", "JPEG")):  # Check for image file extensions
            print(filename)  # Printing the file name of the image
            image_path = os.path.join(image_dir, filename)  # Concatinating the path and the image name
            print(f'Processing image: {image_path}')

            # Load the image
            image = cv.imread(image_path)
            # Convert the image to grayscale
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            # Find chessboard corners
            ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

            if ret:
                object_points.append(objp)
                # Improving the resolution of the corner points
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                img_points.append(corners2)  # Appending the image points

                # Draw the corners on the original image
                cv.drawChessboardCorners(image, chessboard_size, corners2, ret)
                cv.imshow('Chessboard Corners', image)
                cv.waitKey(100)
                cv.destroyAllWindows()
            else:
                print("Chessboard corners not found in the image.")    

    # Calibrate the camera
    h, w = gray.shape[:2]
    camera_matrix = np.array([[w, 0, w / 2],
                              [0, h, h / 2],
                              [0, 0, 1]], dtype=np.float32)
    
    print(f'Initial Camera Matrix = {camera_matrix}\n')

    ret, camera_matrix, dist, rvecs, tvecs = cv.calibrateCamera(object_points, img_points, gray.shape[::-1], camera_matrix, None, flags=cv.CALIB_USE_INTRINSIC_GUESS)
    print("Camera Calibrated: ", ret)
    print("\nCamera Matrix K = \n", camera_matrix)
    print(f'Distortion parameters : {dist}')

    return camera_matrix, dist, rvecs, tvecs

def compute_reprojection_error(object_points, img_points, rvecs, tvecs, camera_matrix, dist):
    total_error = 0
    for i in range(len(object_points)):
        imgpoints2, _ = cv.projectPoints(object_points[i], rvecs[i], tvecs[i], camera_matrix, dist)
        error = cv.norm(img_points[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        total_error += error
    mean_error = total_error / len(object_points)
    print("Total reprojection error: {}".format(mean_error))
    return mean_error

def undistort_images(undistorted_images_dir, camera_matrix, dist):
    os.makedirs(undistort_images_dir, exist_ok=True)#Directory existance check

    for filename in os.listdir(image_dir):
        if filename.endswith((".jpg", ".jpeg", ".png", "JPEG")):  # Check for image file extensions
            print(f'Undistorting image: {filename}')
            image_path = os.path.join(image_dir, filename)
            image = cv.imread(image_path)

            h, w = image.shape[:2]
            new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist, (w, h), 1, (w, h))

            # Undistort the image
            undistorted_image = cv.undistort(image, camera_matrix, dist, None, new_camera_matrix)

            # Crop the image based on the ROI
            x, y, w, h = roi
            undistorted_image = undistorted_image[y:y+h, x:x+w]

            # Save the undistorted image
            undistorted_image_path = os.path.join(undistort_images_dir, filename)
            cv.imwrite(undistorted_image_path, undistorted_image)
            print(f'Saved undistorted image: {undistorted_image_path}')

def load_intrinsics():
    root = 'C:/Users/pooji/Downloads/Aeroprobe Testing - 0612/Image Distance Analysis Scripts/Intrinsic_Parameters_1080.npz'
    intrinsics = np.load(root,mmap_mode = 'r')
    camera_matrix = intrinsics['camera_matrix']
    dist = intrinsics['dist_coeff']
    print(f' Camera Matrix : \n {camera_matrix} \n dist = {dist}')
    return camera_matrix, dist


def altitude(camera_matrix, dist, image_dir, img_points):
    # Prepare object points
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = (np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2))*1.4
    print(f'objp : \n {objp}')

    # Iterate through all image files in the directory
    for filename in os.listdir(image_dir):
        if filename.endswith((".jpg", ".jpeg", ".png", "JPEG")):  # Check for image file extensions
            print(filename)  # Printing the file name of the image
            image_path = os.path.join(image_dir, filename)  # Concatinating the path and the image name
            print(f'Processing image: {image_path}')

            # Load the image
            image = cv.imread(image_path)
            # Convert the image to grayscale
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            # Find chessboard corners
            ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

            if ret:
                object_points.append(objp)
                # Improving the resolution of the corner points
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                img_points.append(corners2)  # Appending the image points



                # Draw the corners on the original image
                cv.drawChessboardCorners(image, chessboard_size, corners2, ret)

                # Solve PnP
                ret, r_vecs, t_vecs = cv.solvePnP(objp, corners2, camera_matrix, dist)
                
                if ret:
                    # image_points, _ = cv.projectPoints(objp, r_vecs, t_vecs, camera_matrix, dist)
                    # image_points = image_points.reshape(-1, 2)
                    cv.putText(image, str(t_vecs), (50,50),1,1,(0,0,255),1,1,False)
                    # Save the undistorted image
                    distances_in_image = os.path.join(image_dir, filename)
                    cv.imwrite(distances_in_image, image)
                    print(f'Saved undistorted image: {distances_in_image}')


                cv.imshow('Chessboard Corners', image)
                cv.waitKey(100)
                cv.destroyAllWindows()
            else:
                print("Chessboard corners not found in the image.")   

                cv.putText(image, "Not found", (50,50),1,1,(0,0,255),1,1,False)
                # Save the undistorted image
                distances_in_image = os.path.join(image_dir, filename)
                cv.imwrite(distances_in_image, image)
                print(f'Saved undistorted image: {distances_in_image}')
        


if __name__ == "__main__":
    # Define the chessboard dimensions (inside corners)
    chessboard_size = (8, 6)  # For an 8 * 6 chessboard
    frameSize = (1280, 720)  # Image frame size

    # Termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Initialize lists to store object points and image points
    object_points = []  # Object points
    img_points = []  # Image points

    # Directory containing calibration images
    # image_dir = 'C:/Users/pooji/Downloads/Aeroprobe Testing - 0612/Image Distance Analysis Scripts/aeroprobe camera calib/'  # Replace with your image directory path
    # image_dir = 'C:/Users/pooji/Downloads/Aeroprobe Testing - 0612/Image Distance Analysis Scripts/Checker Board Calibration 1080P images'
    image_dir = 'C:/Users/pooji/Downloads/Aeroprobe Testing - 0612/Image Distance Analysis Scripts/October Images'
    
    # undistort_images_dir = 'C:/Users/pooji/Downloads/Aeroprobe Testing - 0612/Image Distance Analysis Scripts/undistorted Images HD/'
    undistort_images_dir = 'C:/Users/pooji/Downloads/Aeroprobe Testing - 0612/Image Distance Analysis Scripts/Oct_Flight/Altitude'
    
    # Calibrate the camera
    # camera_matrix, dist, rvecs, tvecs = calibrate_camera(image_dir, chessboard_size, criteria, object_points, img_points)

    # Save the calibration results
    # np.savez('Intrinsic_Parameters_1080', camera_matrix=camera_matrix, dist_coeff=dist)

    # Compute the reprojection error
    # compute_reprojection_error(object_points, img_points, rvecs, tvecs, camera_matrix, dist)

    # Load intrincsics
    camera_matrix, dist = load_intrinsics()


    # Undistort the images
    # undistort_images(undistort_images_dir, camera_matrix, dist)

    # Get the distances in the image
    altitude(camera_matrix, dist, undistort_images_dir, img_points)
