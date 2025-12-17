# Face Emotion Recognition using CNN

##  Project Overview

This project implements a **Facial Emotion Recognition system** using a **Convolutional Neural Network (CNN)** built with **TensorFlow and Keras**.
The model classifies human facial expressions into **7 distinct emotion categories** using grayscale facial images.

The workflow covers:

* Dataset preparation
* Image augmentation
* Custom CNN architecture design
* Model training
* Performance evaluation using **Accuracy, Precision, Recall, and F1-Score**

---

##  Features

* **Custom CNN Architecture** optimized for facial emotion classification
* **Data Augmentation** using `ImageDataGenerator` (scaling, shearing, zooming, horizontal flipping)
* **Batch Normalization & Dropout** to stabilize training and reduce overfitting
* **Performance Metrics**: Accuracy, Precision, Recall, and F1-Score
* **Google Drive Integration** for easy dataset access in Google Colab

---

##  Technologies & Dependencies

The project uses the following Python libraries:

* `tensorflow` / `tensorflow.keras`
* `numpy`
* `matplotlib`
* `scikit-learn`
* `os`, `shutil`
* `google.colab` (for Google Colab execution)

### Installation

Install the required dependencies using:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

---

##  Dataset

The dataset is expected to be stored in **Google Drive** and organized into training and testing directories.

### Image Properties

* **Image Size:** 48 × 48
* **Color Mode:** Grayscale
* **Classes:** 7 emotion categories

### Expected Directory Structure

```text
MyDrive/
└── Emotion_Dataset/
    ├── train/
    │   ├── emotion_1/
    │   ├── emotion_2/
    │   └── ...
    └── test/
        ├── emotion_1/
        ├── emotion_2/
        └── ...
```

During execution, the dataset is copied to:

```text
/content/emotion_data
```

---

##  Model Architecture

The CNN is built using a **Sequential** model consisting of:

1. **Input Layer**

   * Shape: `(48, 48, 1)` (grayscale images)

2. **Convolutional Blocks**

   * `Conv2D`
   * `BatchNormalization`
   * `MaxPooling2D`
   * `Dropout`

3. **Classification Head**

   * `Flatten`
   * Fully connected `Dense` layers with `Dropout`
   * Output layer with **7 neurons** and **softmax activation**

### Compilation

* **Optimizer:** Adam
* **Loss Function:** Categorical Crossentropy
* **Metrics:** Accuracy

---

##  Usage (Training & Evaluation)

This project is designed to run in **Google Colab** or any Jupyter environment.

### Steps:

1. **Open the Notebook**

   * Load `Emotion_detection.ipynb`

2. **Mount Google Drive**

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Prepare the Dataset**

   * Ensure the dataset is located in:

     ```
     MyDrive/Emotion_Dataset
     ```
   * Run the data preparation cells to copy files locally.

4. **Define & Compile the Model**

   * Execute the CNN model definition cell.

5. **Train the Model**

   * Training configuration:

     * Batch Size: **64**
     * Image Size: **48 × 48**
     * Color Mode: **Grayscale**

6. **Evaluate the Model**

   * Generate **Accuracy**, **Precision**, **Recall**, and **F1-Score**
   * Display the `classification_report`

---

##  Results (Sample)

| Metric    | Training | Testing |
| --------- | -------- | ------- |
| Accuracy  | 50%      | 40%     |
| Precision | 70%      | 53.5%   |
| Recall    | 29%      | 24%     |

> *Results may vary depending on dataset quality, class balance, and training configuration.*

---

##  Future Improvements

* Implement **cross-validation**
* Experiment with **advanced architectures** (ResNet, VGG, EfficientNet)
* Improve class balance and dataset size
* Build a **real-time emotion detection system** using a webcam
* Convert the model to **TensorFlow Lite** for deployment

---

##  License

This project is open-source and available for educational and research purposes.

---

##  Acknowledgements

* TensorFlow & Keras documentation
* Open-source facial emotion datasets
* Google Colab for GPU acceleration


