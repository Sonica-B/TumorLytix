import nibabel as nib
import matplotlib.pyplot as plt

# File paths for the five NIfTI images
file_paths = [
    '/media/farhan/HD-B1/CYCLE_GAN_BT/BraTS20_Training_012_seg.nii',
    '/media/farhan/HD-B1/CYCLE_GAN_BT/BraTS20_Training_012_t1.nii',
    '/media/farhan/HD-B1/CYCLE_GAN_BT/BraTS20_Training_012_t1ce.nii',
    '/media/farhan/HD-B1/CYCLE_GAN_BT/BraTS20_Training_012_t2 .nii',
    '/media/farhan/HD-B1/CYCLE_GAN_BT/BraTS20_Training_012_flair.nii'
]

# Load the images and extract the middle slice of each
images = []
slice_indices = []
for file_path in file_paths:
    img = nib.load(file_path)  # Load the NIfTI file
    data = img.get_fdata()     # Extract image data
    slice_idx = data.shape[2] // 2  # Get the middle slice along the z-axis
    images.append(data[:, :, slice_idx])  # Append the middle slice
    slice_indices.append(slice_idx)

# Plot the images side by side
fig, axes = plt.subplots(1, 5, figsize=(20, 10))  # Create a 1x5 grid
titles = ['Seg (003)', 'T1 (003)', 'Seg (004)', 'T1 (004)', 'T1 (005)']

for i, ax in enumerate(axes):
    ax.imshow(images[i], cmap='gray')  # Display each image
    ax.set_title(f'{titles[i]} (Slice {slice_indices[i]})', fontsize=10)  # Set titles
    ax.axis('off')  # Turn off axes for clean visualization

plt.tight_layout()
plt.show()
