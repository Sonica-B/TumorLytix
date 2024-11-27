import numpy as np
import torch

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt


def calculate_metrics(original, reconstructed):
    """Calculate PSNR and SSIM between the original and reconstructed 3D volumes."""
    original = original.cpu().detach().numpy()
    reconstructed = reconstructed.cpu().detach().numpy()

    # Calculate the data range from the original image
    data_range = original.max() - original.min()

    # Compute PSNR and SSIM slice-by-slice along the depth axis
    psnr_values = []
    ssim_values = []
    for i in range(original.shape[0]):  # Iterate over slices
        psnr_values.append(psnr(original[i], reconstructed[i], data_range=data_range))
        ssim_values.append(ssim(original[i], reconstructed[i], data_range=data_range, multichannel=False))

    # Return mean PSNR and SSIM across slices
    return np.mean(psnr_values), np.mean(ssim_values)


# def multi_modality_ensemble(modalities, weights):
#     """Perform multi-modality fusion based on weighted averaging."""
#     assert len(modalities) == len(weights), "Mismatch between modalities and weights"
#     weights = np.array(weights) / np.sum(weights)  # Normalize weights to sum to 1
#     ensemble = np.sum([w * m for w, m in zip(weights, modalities)], axis=0)
#     return ensemble


def visualize_results(original, reconstructed, anomaly_map, mask, epoch, save_dir="output_images/"):
    """Visualize selected slices from the 3D original, reconstructed, anomaly map, and segmentation mask."""
    # Select the middle slice from the depth dimension
    depth_idx = anomaly_map.shape[1] // 2  # Depth is the second dimension in (Batch, Depth, Height, Width)

    original_slice = original[depth_idx]  # Extract middle slice
    reconstructed_slice = reconstructed[depth_idx]
    anomaly_map_slice = anomaly_map[0, depth_idx]  # Use batch index 0
    mask_slice = mask[0, depth_idx]  # Use batch index 0

    # Plot the slices
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(original_slice, cmap="gray")
    axs[0].set_title("Original")
    axs[1].imshow(reconstructed_slice, cmap="gray")
    axs[1].set_title("Reconstructed")
    axs[2].imshow(anomaly_map_slice, cmap="jet")
    axs[2].set_title("Anomaly Map")
    axs[3].imshow(mask_slice, cmap="gray")
    axs[3].set_title("Segmentation Mask")

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(f"{save_dir}/epoch_{epoch}.png")
    plt.show()



def apply_segmentation(reconstructed, original, threshold=0.5):
    """Generate 3D anomaly map and segmentation mask based on pixel-wise differences."""
    diff_map = np.abs(reconstructed - original)
    anomaly_map = diff_map / np.max(diff_map, axis=(1, 2, 3), keepdims=True)  # Normalize across the volume
    mask = (anomaly_map > threshold).astype(np.uint8)  # Binary mask
    return anomaly_map, mask




# # Test data (example tensors)
# original = torch.randn(1, 64, 256, 256)  # [Depth, Height, Width]
# reconstructed = torch.randn(1, 64, 256, 256)
#
# # Calculate metrics
# psnr_val, ssim_val = calculate_metrics(original, reconstructed)
# print(f"PSNR: {psnr_val}, SSIM: {ssim_val}")
#
#
# # Apply segmentation
# anomaly_map, mask = apply_segmentation(reconstructed.numpy(), original.numpy(), threshold=0.5)
# print(f"Anomaly Map Shape: {anomaly_map.shape}")
# print(f"Mask Shape: {mask.shape}")
# # Visualize results
# # Test visualization
# visualize_results(
#     original[0].numpy(),  # Remove batch dimension
#     reconstructed[0].numpy(),
#     anomaly_map,
#     mask,
#     epoch=1
# )
