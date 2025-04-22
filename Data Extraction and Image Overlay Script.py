'''
This scrip is used to extract the data from the google sheet and then overlay 
the data on to the corresponding images taken on Aeroprobe.
'''

# Import modules
import os
import csv
from PIL import Image, ImageDraw, ImageFont

# Give your paths here:
input_csv = 'C:/Users/pooji/Downloads/Aeroprobe Testing - 0612/Flight 2 final - Sheet1.csv'  # Csv file path of the flight data recorded
image_directory = 'C:/Users/pooji/Downloads/Aeroprobe Testing - 0612/Image Distance Analysis Scripts/Distance Measurement Images/Measured_KRI - With Overlay' # Path of the images captured by Aeroprobe
output_directory = 'C:/Users/pooji/Downloads/Aeroprobe Testing - 0612/Image Distance Analysis Scripts/Distance Measurement Images/Measured_KRI - Final Overlay' # Path where the images with their associated data should be saved
font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'  # Path to a TTF font file

# Data Overlay Function
def overlay_data(image_directory, csv_data):
    # Process each image in the directory
    for filename in os.listdir(image_directory):
        if filename.endswith('.jpg') and filename in csv_data:
            image_path = os.path.join(image_directory, filename)
            output_path = os.path.join(output_directory, filename)
            # Prepare the text to overlay
            data = csv_data[filename]
            text = (f"Timestamp: {data['Timestamp']}\n"
                    f"Temperature: {data['Temperature']}°C\n"
                    f"Humidity: {data['Humidity']}%\n"
                    f"Pressure: {data['Pressure']} hPa\n"
                    f"Altitude: {data['Altitude']} m\n"
                    f"Wind Speed: {data['Wind Speed']} m/s")
            
            # Overlay text on the image and save it
            # Open the image
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)
            # Load a font
            try:
                font = ImageFont.truetype(font_path, 20)  # Try to load the specified font
            except IOError:
                font = ImageFont.load_default()  # Fallback to default font if the specified font is not found
            # Define text position
            text_position = (10, 150)  # You can adjust the position
            # Overlay text
            draw.text(text_position, text, font=font, fill="white")
            # Save the image
            image.save(output_path)
            print(f'Image Saved')


# Main Function
def main():
    try:
        # Directory exixstance check
        os.makedirs(output_directory, exist_ok=False)
    except FileExistsError:
        # Read CSV file into a dictionary
        csv_data = {}
        with open(input_csv, mode='r', newline='') as csvfile:
            print(f'File is opened')
            csvreader = csv.DictReader(csvfile)    
            for row in csvreader:
                csv_data[row['Filename']] = {
                    'Timestamp': row['Timestamp'],
                    'Temperature': row['Temperature'],
                    'Humidity': row['Humidity'],
                    'Pressure': row['Pressure'],
                    'Altitude': row['Altitude'],
                    'Wind Speed': row['Wind Speed']
                }
            print(len(csv_data))
        overlay_data(image_directory, csv_data = csv_data)
        print(f"Overlayed images have been saved to : {output_directory}")


if __name__ == "__main__":
    main()