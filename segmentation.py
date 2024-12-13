import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from DIffusion_Model.VE_JP import VEJP_Diffusion
from dataloader import PairedTrainData
from utils import load_checkpoint, seed_everything
import config
import cv2
import os

# Apply segmentation using anomaly detection
def apply_segmentation(reconstructed, original, threshold=0.5):
    diff_map = np.abs(reconstructed - original)
    anomaly_map = diff_map / (np.max(diff_map) + 1e-8)
    anomaly_map = np.clip(anomaly_map, 0, 1)  # Normalize anomaly map

    mask = (anomaly_map > threshold).astype(np.uint8)

    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    refined_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)

    return anomaly_map, refined_mask

# Visualize and save results
def visualize_results(original, reconstructed, anomaly_map, mask, index, save_dir="output_images/"):
    os.makedirs(save_dir, exist_ok=True)

    # Normalize slices
    original_slice = original.squeeze() / np.max(original)
    reconstructed_slice = reconstructed.squeeze() / np.max(reconstructed)
    anomaly_map_slice = anomaly_map.squeeze()
    mask_slice = mask.squeeze()

    # Highlight tumor area
    tumor_highlight = np.zeros_like(original_slice)
    tumor_highlight[mask_slice > 0] = 1.0  # Highlight segmented tumor regions

    # Plot
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    axs[0].imshow(original_slice, cmap="gray")
    axs[0].set_title("Original (Tumor)")
    axs[1].imshow(reconstructed_slice, cmap="gray")
    axs[1].set_title("Reconstructed (Healthy)")
    axs[2].imshow(anomaly_map_slice, cmap="jet")
    axs[2].set_title("Anomaly Map")
    axs[3].imshow(mask_slice, cmap="gray")
    axs[3].set_title("Segmentation Mask")
    axs[4].imshow(original_slice, cmap="gray")
    axs[4].imshow(tumor_highlight, cmap="jet", alpha=0.5)
    axs[4].set_title("Overlay")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/image_{index}.png")
    plt.close()

# Denoise images using the trained model
def denoise_images(model, dataloader, device):
    reconstructed_images = []
    original_images = []

    with torch.no_grad():
        for xA, xB in dataloader:
            xA, xB = xA.to(device), xB.to(device)
            noisy_xB, _, _ = model.forward_process(xB, xA, t=0)  # Pass xB (tumor) as primary input
            input_tensor = torch.cat([noisy_xB, xA], dim=1)  # xA as auxiliary input
            reconstructed_xB = model.unet(input_tensor)
            reconstructed_images.append(reconstructed_xB.cpu())
            original_images.append(xB.cpu())  # Use xB for tumor data

    return torch.cat(original_images), torch.cat(reconstructed_images)

# Main workflow
if __name__ == "__main__":
    seed_everything()

    # Load the trained model
    checkpoint_path = "DIffusion_Model/output_images/diffusion_epoch_50.pth.tar"
    timesteps = 1000
    noise_schedule = torch.linspace(0.01, 0.999, timesteps).to(config.DEVICE)
    model = VEJP_Diffusion(noise_schedule=noise_schedule, input_channels=1).to(config.DEVICE)
    load_checkpoint(checkpoint_path, model, optimizer=None, lr=config.learning_rate)

    # Load dataset
    dataset = PairedTrainData(
        root_dir_normal=config.train_dir_normal,
        root_dir_abnormal=config.train_dir_abnormal,
        resize_to=256,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.num_workers)

    # Process each image
    original_images, reconstructed_images = denoise_images(model, dataloader, config.DEVICE)

    # Generate segmentation for each image
    for i in range(original_images.shape[0]):
        anomaly_map, mask = apply_segmentation(
            reconstructed=reconstructed_images[i].numpy(),
            original=original_images[i].numpy(),
            threshold=0.25
        )
        visualize_results(
            original_images[i].numpy(),
            reconstructed_images[i].numpy(),
            anomaly_map,
            mask,
            index=i,
            save_dir="output_images/"
        )

# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# from DIffusion_Model.VE_JP import VEJP_Diffusion
# from dataloader import PairedTrainData
# from utils import load_checkpoint, seed_everything
# import config
# import cv2
# import os
#
# # Apply segmentation using anomaly detection
# def apply_segmentation(reconstructed, original, threshold=0.5):
#     diff_map = np.abs(reconstructed - original)
#     anomaly_map = diff_map / (np.max(diff_map) + 1e-8)
#     anomaly_map = np.clip(anomaly_map, 0, 1)  # Normalize anomaly map
#
#     mask = (anomaly_map > threshold).astype(np.uint8)
#
#     # Morphological operations
#     kernel = np.ones((3, 3), np.uint8)
#     refined_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
#
#     return anomaly_map, refined_mask
#
# # Visualize and save only the segmentation mask with resizing
# def visualize_results(mask, index, save_dir="output_images/"):
#     os.makedirs(save_dir, exist_ok=True)
#
#     # Extract the segmentation mask slice and resize it to 256x256
#     mask_slice = mask.squeeze()
#     mask_resized = cv2.resize(mask_slice, (256, 256), interpolation=cv2.INTER_NEAREST)
#
#     # Convert the mask to 8-bit format for saving
#     mask_resized = (mask_resized * 255).astype(np.uint8)
#
#     # Save the resized mask directly as a grayscale image
#     save_path = f"{save_dir}/segmentation_mask_{index}.png"
#     cv2.imwrite(save_path, mask_resized)
#
#
#
#
# # Denoise images using the trained model
# def denoise_images(model, dataloader, device):
#     reconstructed_images = []
#     original_images = []
#
#     with torch.no_grad():
#         for xA, xB in dataloader:
#             xA, xB = xA.to(device), xB.to(device)
#             noisy_xB, _, _ = model.forward_process(xB, xA, t=0)  # Pass xB (tumor) as primary input
#             input_tensor = torch.cat([noisy_xB, xA], dim=1)  # xA as auxiliary input
#             reconstructed_xB = model.unet(input_tensor)
#             reconstructed_images.append(reconstructed_xB.cpu())
#             original_images.append(xB.cpu())  # Use xB for tumor data
#
#     return torch.cat(original_images), torch.cat(reconstructed_images)
#
# # Main workflow
# if __name__ == "__main__":
#     seed_everything()
#
#     # Load the trained model
#     checkpoint_path = "DIffusion_Model/output_images/diffusion_epoch_50.pth.tar"
#     timesteps = 1000
#     noise_schedule = torch.linspace(0.01, 0.999, timesteps).to(config.DEVICE)
#     model = VEJP_Diffusion(noise_schedule=noise_schedule, input_channels=1).to(config.DEVICE)
#     load_checkpoint(checkpoint_path, model, optimizer=None, lr=config.learning_rate)
#
#     # Load dataset
#     dataset = PairedTrainData(
#         root_dir_normal=config.train_dir_normal,
#         root_dir_abnormal=config.train_dir_abnormal,
#         resize_to=256,
#     )
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.num_workers)
#
#     # Process each image
#     original_images, reconstructed_images = denoise_images(model, dataloader, config.DEVICE)
#
#     # Generate segmentation for each image
#     for i in range(original_images.shape[0]):
#         anomaly_map, mask = apply_segmentation(
#             reconstructed=reconstructed_images[i].numpy(),
#             original=original_images[i].numpy(),
#             threshold=0.25
#         )
#         visualize_results(
#             mask,
#             index=i,
#             save_dir="output_images/"
#         )