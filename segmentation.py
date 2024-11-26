import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt


def calculate_metrics(original, reconstructed):
    """Calculate PSNR and SSIM between the original and reconstructed image."""
    original = original.cpu().detach().numpy()
    reconstructed = reconstructed.cpu().detach().numpy()

    psnr_value = psnr(original, reconstructed)
    ssim_value = ssim(original, reconstructed, multichannel=True)
    return psnr_value, ssim_value


def multi_modality_ensemble(modalities, weights):
    """Perform multi-modality fusion based on weighted averaging."""
    assert len(modalities) == len(weights), "Mismatch between modalities and weights"
    weights = np.array(weights) / np.sum(weights)  # Normalize weights to sum to 1
    ensemble = np.sum([w * m for w, m in zip(weights, modalities)], axis=0)
    return ensemble


def visualize_results(original, reconstructed, anomaly_map, mask, epoch, save_dir="output_images/"):
    """Visualize the original image, reconstructed image, anomaly map, and segmentation mask."""
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(original, cmap="gray")
    axs[0].set_title("Original")
    axs[1].imshow(reconstructed, cmap="gray")
    axs[1].set_title("Reconstructed")
    axs[2].imshow(anomaly_map, cmap="jet")
    axs[2].set_title("Anomaly Map")
    axs[3].imshow(mask, cmap="gray")
    axs[3].set_title("Segmentation Mask")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/epoch_{epoch}.png")
    plt.show()


def apply_segmentation(reconstructed, original, threshold=0.5):
    """Generate an anomaly map and segmentation mask based on pixel differences."""
    diff_map = np.abs(reconstructed - original)
    anomaly_map = diff_map / np.max(diff_map)
    mask = (anomaly_map > threshold).astype(np.uint8)
    return anomaly_map, mask
