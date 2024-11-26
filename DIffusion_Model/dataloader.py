import os
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import torch

class TrainData(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Root directory containing all subject directories.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        # List all subject directories in the root directory
        self.subject_dirs = sorted(
            [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        )

    def __len__(self):
        return len(self.subject_dirs)

    def __getitem__(self, idx):
        # Get the directory for the current subject
        subject_dir = self.subject_dirs[idx]

        # Load each modality and segmentation mask
        flair_path = os.path.join(subject_dir, f"{os.path.basename(subject_dir)}_flair.nii")
        t1_path = os.path.join(subject_dir, f"{os.path.basename(subject_dir)}_t1.nii")
        t1ce_path = os.path.join(subject_dir, f"{os.path.basename(subject_dir)}_t1ce.nii")
        t2_path = os.path.join(subject_dir, f"{os.path.basename(subject_dir)}_t2.nii")
        seg_path = os.path.join(subject_dir, f"{os.path.basename(subject_dir)}_seg.nii")

        flair_img = np.array(nib.load(flair_path).get_fdata())
        t1_img = np.array(nib.load(t1_path).get_fdata())
        t1ce_img = np.array(nib.load(t1ce_path).get_fdata())
        t2_img = np.array(nib.load(t2_path).get_fdata())
        seg_img = np.array(nib.load(seg_path).get_fdata())  # Ground truth segmentation

        # Combine the modalities into a multi-channel image
        input_image = np.stack([flair_img, t1_img, t1ce_img, t2_img], axis=0)

        # Apply transformations if provided
        if self.transform:
            input_image = self.transform(input_image)
            seg_img = self.transform(seg_img)

        # Convert to PyTorch tensors
        input_image = torch.tensor(input_image, dtype=torch.float32)
        seg_img = torch.tensor(seg_img, dtype=torch.long)  # Segmentation masks as labels

        return input_image, seg_img
