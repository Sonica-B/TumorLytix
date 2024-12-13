import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Paths
input_file = "D:/bhang/Downloads/extracted_data/dataset.csv"  # Replace with your CSV file path
output_base_dir = "D:/bhang/Downloads/extracted_data/images/"  # Folder to save organized data
os.makedirs(output_base_dir, exist_ok=True)


# Load the data
data = pd.read_csv(input_file)

# Function to save a slice as an image
def save_as_image(nii_path, output_path, slice_index=None):
    # Load the NIfTI file
    nii_data = nib.load(nii_path)
    img_array = nii_data.get_fdata()
    
    # Choose a slice (default is the middle slice if not specified)
    if slice_index is None:
        
        slice_index = img_array.shape[2] // 2  # Middle slice
    slice_img = img_array[:, :, slice_index]
    
    # Normalize the image to 0-255
    slice_img_normalized = ((slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img)) * 255).astype(np.uint8)
    
    # Save as PNG
    plt.imsave(output_path, slice_img_normalized, cmap="gray")

# Process each row
for index, row in data.iterrows():
    # Folder for each brain tumor case
    case_folder = os.path.join(output_base_dir, f"Case_{index+1}")
    os.makedirs(case_folder, exist_ok=True)
    
    # Iterate through image columns
    for column_name in ['flair', 't1', 't1ce', 't2', 'seg']:
        nii_file_path = row[column_name]
        if pd.notna(nii_file_path):  # Check if the file path is valid
            output_file_path = os.path.join(case_folder, f"{column_name}.png")
            save_as_image(nii_file_path, output_file_path)

print(f"Images successfully converted and saved in: {output_base_dir}")
