import os
import nibabel as nib
import numpy as np
from PIL import Image



import zipfile
import os

# Input and output directories
archive_dir = 'D:\\bhang\\Downloads\\'  # Folder containing the zip files
output_dir = 'D:\\bhang\\Downloads\\extracted_data\\input'               # Folder to extract contents
def extract_zip(file_path, extract_to):
    """Extracts a ZIP file to a specified directory."""
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted: {file_path} to {extract_to}")

def extract_all_zips(input_dir, output_dir):
    """Recursively extracts all ZIP files in a directory."""
    for root, _, files in os.walk(input_dir):
        for file_name in files:
            if file_name.endswith('.zip'):
                archive_path = os.path.join(root, file_name)
                extract_dir = os.path.join(output_dir, os.path.splitext(file_name)[0])
                os.makedirs(extract_dir, exist_ok=True)
                extract_zip(archive_path, extract_dir)
                # Recursively extract nested ZIP files
                extract_all_zips(extract_dir, output_dir)

# Extract all zip files in the archive directory
extract_all_zips(archive_dir, output_dir)

print(f"All nested files extracted to: {output_dir}")


# Input folder containing .nii files
input_folder = 'D:\\bhang\\Downloads\\extracted_data\\input'
# Output folder to save .jpg images
output_folder = 'D:\\bhang\\Downloads\\extracted_data\\images'
os.makedirs(output_folder, exist_ok=True)

def normalize_image(image):
    """Normalize the image to 0-255 for saving as JPEG."""
    image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Scale to 0-1
    image = (image * 255).astype(np.uint8)  # Scale to 0-255
    return image

def save_slices_as_jpg(nifti_file, output_folder):
    """Convert a NIfTI file to .jpg slices and save."""
    # Load the NIfTI file
    img = nib.load(nifti_file)
    img_data = img.get_fdata()
    
    # Loop through each slice along the z-axis
    for i in range(img_data.shape[2]):
        slice_data = img_data[:, :, i]  # Get the i-th slice
        normalized_slice = normalize_image(slice_data)  # Normalize the slice
        slice_image = Image.fromarray(normalized_slice)  # Convert to image
        
        # Create a unique filename for each slice
        base_name = os.path.splitext(os.path.basename(nifti_file))[0]
        slice_filename = os.path.join(output_folder, f"{base_name}_slice_{i}.jpg")
        
        # Save the slice as a .jpg file
        slice_image.save(slice_filename)

# Process all .nii files in the input folder
for nii_file in os.listdir(input_folder):
    if nii_file.endswith(".nii") or nii_file.endswith(".nii.gz"):
        full_path = os.path.join(input_folder, nii_file)
        save_slices_as_jpg(full_path, output_folder)

print("Conversion complete. Check the output folder for .jpg images.")
