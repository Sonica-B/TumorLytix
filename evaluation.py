import os
import numpy as np
from sklearn.metrics import precision_score, recall_score
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

# Function to calculate Dice Coefficient
def dice_coefficient(gt_mask, pred_mask):
    intersection = np.sum(gt_mask * pred_mask)
    return (2 * intersection) / (np.sum(gt_mask) + np.sum(pred_mask))

# Function to calculate Precision
def precision(gt_mask, pred_mask):
    return precision_score(gt_mask.flatten(), pred_mask.flatten())

# Function to calculate Recall
def recall(gt_mask, pred_mask):
    return recall_score(gt_mask.flatten(), pred_mask.flatten())

# Function to load and process image files
def load_image(image_path):
    img = Image.open(image_path)
    img = img.convert('L')  # Convert to grayscale (if not already)
    img_bin = np.array(img) > 128  # Convert to binary based on a threshold
    return img_bin.astype(np.uint8)  # Return as binary array

# Function to process a single file
def process_file(gt_file):
    gt_image_path = os.path.join(gt_dir, gt_file)
    pred_image_path = os.path.join(pred_dir, gt_file)
    
    # Check if the corresponding predicted mask exists
    if not os.path.exists(pred_image_path):
        print(f'Predicted mask for {gt_file} not found.')
        return None

    # Load and process the images
    gt_mask = load_image(gt_image_path)  # Ground truth mask (binary)
    pred_mask = load_image(pred_image_path)  # Predicted mask (binary)

    # Ensure that both masks have the same shape
    if gt_mask.shape != pred_mask.shape:
        print(f'Shape mismatch for {gt_file}.')
        return None

    # Calculate Dice Coefficient
    dice = dice_coefficient(gt_mask, pred_mask)
    print(f'{gt_file} - Dice Coefficient: {dice:.4f}')

    # Calculate Precision
    prec = precision(gt_mask, pred_mask)
    print(f'{gt_file} - Precision: {prec:.4f}')

    # Calculate Recall
    rec = recall(gt_mask, pred_mask)
    print(f'{gt_file} - Recall: {rec:.4f}')

    return {
        'File': gt_file,
        'Dice Coefficient': dice,
        'Precision': prec,
        'Recall': rec
    }

# Directories containing the ground truth and predicted masks
gt_dir = 'D:\\bhang\\Downloads\\ground_truth_mask'
pred_dir = 'D:\\bhang\\Downloads\\CycleGANs_Segmentation'

# List of image files in the ground truth directory
gt_files = os.listdir(gt_dir)

# Initialize lists to store the results
results = {
    'File': [],
    'Dice Coefficient': [],
    'Precision': [],
    'Recall': []
}

# Use ThreadPoolExecutor to process files in parallel
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_file, gt_file) for gt_file in gt_files]
    for future in as_completed(futures):
        result = future.result()
        if result:
            results['File'].append(result['File'])
            results['Dice Coefficient'].append(result['Dice Coefficient'])
            results['Precision'].append(result['Precision'])
            results['Recall'].append(result['Recall'])

# Create a DataFrame from the results
df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
df.to_csv('evaluation_results.csv', index=False)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(df['File'], df['Dice Coefficient'], label='Dice Coefficient', marker='o')
plt.plot(df['File'], df['Precision'], label='Precision', marker='o')
plt.plot(df['File'], df['Recall'], label='Recall', marker='o')
plt.xlabel('File')
plt.ylabel('Score')
plt.title('Evaluation Metrics')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
