# Face-Emotion-Recognition

Emotion Detection using Convolutional Neural Networks (CNN)
Project Description
This project implements a deep learning model, specifically a Convolutional Neural Network (CNN), to classify human emotions from facial images. The model is trained to distinguish between 7 distinct emotional states using the Keras and TensorFlow libraries.
The workflow includes data preparation, image augmentation, custom CNN model definition, training, and evaluation using common metrics like Precision, Recall, and F1-score.
Features
•	Custom CNN Architecture: A deep convolutional network designed for image classification tasks.
•	Data Augmentation: Utilizes ImageDataGenerator for real-time augmentation (scaling, shearing, zooming, flipping) to improve model generalization.
•	Batch Normalization and Dropout: Includes Batch Normalization and Dropout layers to stabilize training and prevent overfitting.
•	Performance Metrics: Tracks and reports Accuracy, Precision, and Recall during training and final evaluation.
•	Google Drive Integration: Includes code to mount Google Drive for easy access to the dataset.
Dependencies
The following Python libraries are required to run the notebook and the model:
•	tensorflow (and tensorflow.keras)
•	numpy
•	matplotlib
•	scikit-learn (sklearn.metrics)
•	os, shutil (for file operations)
•	google.colab (if running on Google Colab)
You can install the primary dependencies using pip:
Bash
pip install tensorflow numpy matplotlib scikit-learn
Dataset
This project is configured to use a dataset stored on Google Drive.
•	Expected Directory Structure: The notebook expects the raw data to be located at a path relative to your Google Drive root.
•	Local Data Path: After mounting Google Drive, the script copies the data into a local directory for processing: /content/emotion_data.
•	Data Split: The data is expected to be organized into train and test subdirectories.
•	Image Properties: The model is configured to process images of size 48x48 pixels in grayscale format.
Expected Drive Structure
The notebook anticipates a structure similar to this within your mounted Drive:
MyDrive/
└── Emotion_Dataset/
    ├── train/
    |   ├── emotion_A/
    |   └── emotion_B/
    └── test/
        ├── emotion_A/
        └── emotion_B/
Model Architecture
The CNN model uses a sequential stack of layers:
1.	Input Layer: Expects grayscale images of shape (48, 48, 1).
2.	Multiple Convolutional Blocks: Each block consists of:
o	Conv2D (Convolutional Layer)
o	Batch Normalization
o	MaxPooling2D
o	Dropout (to prevent overfitting)
3.	Classification Head:
o	Flatten layer.
o	One or more Dense (Fully Connected) layers with Dropout.
o	Output Layer: A final Dense layer with 7 units and a softmax activation function for multi-class classification.
Compilation: The model is compiled using the adam optimizer and categorical_crossentropy as the loss function.
Prepare the Data:
Ensure your emotion dataset is uploaded to the required location on your Google Drive (e.g., MyDrive/Emotion_Dataset).
Usage (Training and Evaluation)
The project is designed to be run within a Jupyter environment like Google Colab (as indicated by the google.colab import).
1.	Open the Notebook: Load Emotion_detection.ipynb in your preferred Jupyter environment.
2.	Mount Google Drive: Run the cell containing drive.mount('/content/drive') and follow the authentication steps.
3.	Run Data Preparation Cells: Execute the cells that define file paths and copy the data to the local runtime environment.
4.	Define and Compile Model: Execute the cell that defines and compiles the CNN model.
5.	Train the Model: Run the cell that calls the model.fit() function. The training configuration uses:
o	Batch Size: 64
o	Image Size: (48, 48)
o	Color Mode: Grayscale
6.	Evaluate: Run the final cells to evaluate the model on the test set and display the classification_report.
Results
Metric	Training Accuracy (Example)	Test Accuracy (Example)
Accuracy	50%	40%
Precision	70%	53.5%
Recall	29%	24%
Classification Report
    
Future Work
•	Implement cross-validation or a different validation strategy.
•	Explore advanced architectures like ResNet or VGG.
•	Integrate a real-time emotion detection webcam application.

