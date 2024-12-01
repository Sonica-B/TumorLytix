import os
import shutil

# Define the source directory containing the case folders
source_directory = r"D:\\bhang\\Downloads\\extracted_data\\images\\"

# Define the destination directory where `t1ce.png` images will be copied
destination_directory = r"D:\\bhang\\Downloads\\t1ce_images"

# Create the destination directory if it doesn't exist
os.makedirs(destination_directory, exist_ok=True)

# Iterate through all the case folders in the source directory
for case_folder in os.listdir(source_directory):
    case_path = os.path.join(source_directory, case_folder)
    t1ce_image_path = os.path.join(case_path, "t1ce.png")
    
    # Check if `t1ce.png` exists in the current case folder
    if os.path.isfile(t1ce_image_path):
        # Copy the `t1ce.png` image to the destination directory
        shutil.copy(t1ce_image_path, os.path.join(destination_directory, f"{case_folder}_t1ce.png"))
        print(f"Copied: {t1ce_image_path} to {destination_directory}")
    else:
        print(f"No t1ce.png found in {case_folder}")

print("All available t1ce.png images have been copied!")
