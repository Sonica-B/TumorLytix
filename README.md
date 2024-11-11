# TumorLytix
### _Unsupervised Brain Tumor Detection and Segmentation_

## Project Overview

**TumorLytix** is a deep learning project that addresses the problem of brain tumor segmentation and detection using MRI images. The goal is to create a tool that accurately detects and segments brain tumors by integrating CycleGAN and diffusion models, even without labeled data. This method is designed to synthesize paired healthy-diseased images and use conditional guidance to enhance tumor segmentation accuracy.

### Team Members
- Ankit Gole
- Devesh Bhangale
- Farhan Seliya
- Sarthak Mehta
- Shreya Boyane

---

## Project Objectives and Approach

### Key Goals
1. **Detection and Segmentation of Tumors**: Detect brain tumor regions and accurately outline tumor boundaries.
2. **Paired Image Synthesis**: Generate pseudo-paired healthy-diseased images to train the model effectively.
3. **Integration of Generative Models**: Use GANs to detect anomalies and diffusion models to segment tumors with high precision.

### Why This Project?
Detecting brain tumors using MRI images can be challenging due to variability in tumor size, shape, and position. Traditional supervised methods require extensive labeled datasets, which can be difficult to obtain in medical imaging. Our unsupervised approach leverages generative models to detect anomalies in brain scans, making it practical for real-world applications without relying on labeled data.

### Differentiation
While GANs are often used for similar tasks, our approach combines CycleGAN for anomaly detection and diffusion models for precise boundary segmentation. This dual approach addresses common challenges like subtle tumor boundaries and irregular shapes, making it suitable for clinical environments.

---

## Implementation Details

### Dataset
- **BraTs2020 Dataset**: Contains MRI scans from 369 patients across four modalities (T1, T2, T1CE, and FLAIR), with a total of 23,180 healthy and 41,892 diseased images after processing.
- **Diversity**: The dataset includes a variety of tumors, enhancing the model's ability to generalize across different tumor shapes and sizes.

### Performance Metrics
To evaluate the effectiveness of the model, we will use:
1. **Dice Similarity Coefficient (DSC)**: Measures overlap between predicted tumor regions and ground truth.
2. **Precision and Recall**: To assess the accuracy and reliability of detecting tumor regions.

Our baseline models include CycleGAN, VAE, and DPM classifier models, with a goal of surpassing their performance, particularly in terms of DSC.

### Methodology
1. **CycleGAN for Image Synthesis**: We use CycleGAN to create paired healthy-tumor images by training on unpaired tumor images, generating synthetic abnormal regions on healthy images.
2. **Diffusion Models for Segmentation**: Using conditional guidance, the diffusion model gradually highlights tumor regions, focusing on refining boundary segmentation.

### Technical Steps
1. **Anomaly Detection with GANs**: Train GANs on healthy images, detecting anomalies by identifying where the GAN struggles to reconstruct abnormal features.
2. **Segmentation with Diffusion Models**:
   - **Step 1**: Train the diffusion model with conditional guidance, enabling it to focus on differentiating tumor boundaries.
   - **Step 2**: During inference, the model reconstructs sharper tumor boundaries, creating a detailed segmentation map.
   - **Step 3**: Post-process the segmentation map to smooth tumor boundaries and ensure continuity.

---

## Resources and References

### Data and Tools
- **Datasets**: BraTs2020 MRI Dataset from Kaggle.
- **Tools and Libraries**: CycleGAN, Denoising Diffusion Probabilistic Models, scikit-learn, TensorFlow.

### References
1. *Fast Unsupervised Brain Anomaly Detection and Segmentation with Diffusion Models* - SpringerLink.
2. *Conditional Diffusion Models for Semantic 3D Brain MRI Synthesis* by Zolnamar Dorjsembe et al.
3. GitHub: [CycleGAN-medical-image-segmentation](https://github.com/H2K804/CycleGAN-medical-image-segmentation).

---

## Demonstration and Evaluation

### Testing
Our model’s effectiveness will be validated through metrics such as DSC, precision, and recall. We plan to compare our results with baseline models to measure improvements in tumor segmentation accuracy.

---

## Branching Strategy

![Image](https://github.com/user-attachments/assets/7df4e810-02ff-4304-9905-8a6b1745bf5a)


### 1. Main Branches
- **main branch:** This is the stable branch with production-ready code. Only thoroughly reviewed and tested code is merged here.
- **integration branch:** This branch is for integrating features before they’re finalized. All new features and bug fixes are merged into this branch first. Regularly updated from team members' contributions, it serves as the main staging area.

### 2. Development Branches
- **Feature branches (feature/branch-name):** Each team member should create a feature branch off `integration`for a specific task or feature. These branches focus on specific parts of the project (e.g., `feature/data-preprocessing`, `feature/model-selection`, etc.). Each `feature branch` is merged back into `integration`once complete.
- Naming Convention:  ` feature/short-description`  (e.g., `feature/data-collection`, `feature/model-training`)

- **Bugfix branches (bugfix/branch-name):** If issues arise during development, team members can create branches off `integration`for bug fixes. These branches are named descriptively (e.g., `bugfix/data-cleaning-issue`) and merged back into `integration`after testing.
- Naming Convention:  `bugfix/short-description` (e.g., `bugfix/missing-values`, `bugfix/model-overfitting`)

### 3. Pull Requests:
When ready, team members open Pull Requests (PRs) from feature branches into integration, moving the linked task in GitHub Projects to Review.
Other team members review the PR and suggest improvements if needed. Once approved, merge it into `integration`.

## Workflow
Planning and Assignment: Break down the project into distinct tasks (e.g., data preprocessing, feature engineering, model training, evaluation). Ensure that everyone has a clear feature branch in which to work.

**Daily Development:**
- Each team member pulls the latest integration branch and works on their respective feature branches.
- Team members commit frequently and push updates to their feature branches.

**Regular Integration:**
- Designate short integration sessions (e.g., every 2-3 days) for all members to merge their feature branches into integration.
- Review each feature branch with code reviews to check for any conflicts or issues.

 **Testing and Finalizing:**
- Toward the end of the project, focus on integrating and thoroughly testing the code in integration.
- Conduct final testing and review on the `integration` branch, and then merge it into `main `once the project reaches a stable, production-ready state.

---

## Final Wrap-Up
When the project is complete and ready to submit, ensure all code from `integration` is thoroughly tested and merged into `main` for the final version. Tag this final commit as **v1.0.**
