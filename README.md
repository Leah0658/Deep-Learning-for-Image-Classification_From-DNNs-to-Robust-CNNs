
# Deep Learning for Image Classification: From DNNs to Robust CNNs

## Project Overview

This project showcases a comprehensive exploration of deep learning techniques applied to various image classification tasks. Starting from fundamental neural network concepts, it progresses through practical implementations of Deep Neural Networks (DNNs) and Convolutional Neural Networks (CNNs) on different datasets, culminating in an investigation into the robustness of CNN models against adversarial attacks. This coursework demonstrates a strong understanding of deep learning theory, practical model development, data handling, and advanced topics in AI security.

## Key Features & Learning Outcomes

* **Foundational Understanding:** Solid grasp of core deep learning concepts, including activation functions, their properties, and derivatives.
* **DNN Implementation:** Practical experience in building and training Deep Neural Networks for image classification on standard datasets like FashionMNIST.
* **CNN Development:** Proficiency in designing, training, and evaluating Convolutional Neural Networks on custom datasets, handling approximately 10,000 images across 20 animal classes.
* **Data Preprocessing:** Skills in handling, loading, and augmenting image datasets for deep learning models, including resizing, horizontal flips, color jitter, and normalization.
* **GPU Acceleration:** Experience leveraging CUDA for efficient model training on GPUs.
* **Adversarial Robustness:** Introduction to the concept of adversarial attacks (e.g., PGD, FGSM) and evaluation of model robustness.
* **PyTorch Proficiency:** Hands-on experience with PyTorch, a leading deep learning framework, for model construction, training, and evaluation.

## Project Structure

This repository contains three Jupyter Notebooks, each representing a distinct part of the deep learning assignment:

### 1. `Neural_Networks_Fundamentals.ipynb`

* **Description:** This notebook focuses on the theoretical underpinnings of Deep Neural Networks. It includes exercises and solutions related to understanding various activation functions (e.g., ELU), deriving their mathematical properties, and visualizing their behavior and derivatives.
* **Key Learnings:** Reinforces the mathematical and theoretical foundations crucial for understanding how neural networks learn and operate.

### 2. `FashionMNIST_DNN_Classification.ipynb`

* **Description:** This part involves the practical application of Deep Neural Networks for image classification using the widely-used FashionMNIST dataset. It covers data loading, building a DNN architecture, training the model, and evaluating its performance on 70,000 grayscale images across 10 fashion item categories.
* **Key Learnings:** Demonstrates practical skills in implementing and training a basic DNN for a real-world image classification task, including data handling with PyTorch's `DataLoader` and `torchvision`.

### 3. `CNN_Image_Classification_and_Adversarial_Attacks.ipynb`

* **Description:** This advanced section delves into Convolutional Neural Networks for image classification on a custom dataset of approximately 10,000 images of 20 animal classes, with each class having around 500 images. It includes comprehensive data preprocessing steps (resizing images to `[3,64,64]`, random horizontal flips, color jitter, normalization), CNN model definition, training with GPU acceleration, and a critical evaluation of model robustness against common adversarial attacks like Projected Gradient Descent (PGD) and Fast Gradient Sign Method (FGSM).
* **Key Learnings:** Showcases expertise in building advanced CNN models, handling custom datasets, optimizing training with GPUs, and understanding the important concept of model security and robustness in the face of malicious inputs.

## Technologies Used

* Python 3.x
* PyTorch
* torchvision
* NumPy
* Matplotlib
* PIL (Pillow)
* Jupyter Notebook

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <https://github.com/Leah0658/Deep-Learning-for-Image-Classification_From-DNNs-to-Robust-CNNs.git>
    cd <Deep-Learning-for-Image-Classification_From-DNNs-to-Robust-CNNs>
    ```
2.  **Install dependencies:**
    ```bash
    pip install torch torchvision numpy matplotlib pillow
    ```
3.  **Download and set up the custom dataset (for `CNN_Image_Classification_and_Adversarial_Attacks.ipynb`):**
    * The dataset for Part 3 can be downloaded from this Google Drive link: [Animals_Dataset.zip](https://drive.google.com/file/d/1aEkxNWaD02Z8ZNvZzeMefUoY97C-3wTG/view?usp=drive_link).
    * Alternatively, you can use `gdown` to download it directly:
        ```bash
        !gdown --fuzzy [https://drive.google.com/file/d/1qdElRqDS4TitXfv_iG_TFQSi9QIfy4uM/view?usp=drive_link](https://drive.google.com/file/d/1qdElRqDS4TitXfv_iG_TFQSi9QIfy4uM/view?usp=drive_link)
        ```
    * Unzip the dataset:
        ```bash
        !unzip -q Animals_Dataset.zip
        ```
    * Ensure the unzipped dataset (`FIT5215_Dataset`) is located in the same directory as the notebook. The notebook expects the data directory to be `./FIT5215_Dataset`.
4.  **Open and run the notebooks:**
    ```bash
    jupyter notebook
    ```
    Then, navigate to each `.ipynb` file and execute the cells sequentially. Ensure you have a CUDA-enabled GPU for optimal performance in Part 3.

## Results

* **`FashionMNIST_DNN_Classification.ipynb`:** After 20 epochs, the DNN model achieved a training accuracy of approximately 95.63% and a test accuracy of approximately 95.02%, with corresponding losses of 1.5088 (train) and 1.5140 (test). *(Note: These specific results were found in a snippet from `Neural_Networks_Fundamentals.ipynb` but are indicative of the expected performance for a similar DNN model on FashionMNIST.)*
* **`CNN_Image_Classification_and_Adversarial_Attacks.ipynb`:** The CNN model was successfully trained on the custom animal dataset, with a split of 8519 training instances and 947 validation instances. The notebook evaluates the robust accuracy of the model against PGD and FGSM adversarial attacks (specific accuracy values are generated upon execution within the notebook).

## Future Work

* Explore more complex CNN architectures (e.g., ResNet, Inception) for improved classification accuracy on the custom dataset.
* Implement advanced data augmentation techniques.
* Investigate different adversarial defense mechanisms to further enhance model robustness.
* Extend the project to other computer vision tasks like object detection or segmentation.
````