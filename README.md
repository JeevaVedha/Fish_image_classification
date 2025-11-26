## 🐠 Multiclass Fish Image Classification
This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras for the multiclass classification of fish species from images, and deploys the model using a Streamlit web application for interactive, real-time predictions.

### 📘 Overview
The project provides a complete end-to-end solution: from data loading and model training to deployment. The CNN is trained to identify different fish species based on images, and the Streamlit app allows users to upload their own image to test the classifier.


### Project Goals 

Develop a CNN model for multiclass image classification of fish species.

Utilize ImageDataGenerator for efficient loading and preprocessing of image data.

Deploy the trained model using Streamlit for a user-friendly prediction interface.

### 🧩 Key FeaturesData Preprocessing: 

Uses Keras ImageDataGenerator for automatic image resizing and pixel value normalization (rescaling to $[0, 1]$).CNN Model: A sequential model with three Convolutional blocks (Conv2D + MaxPooling2D) followed by Dense layers.Model Training: Compilation using the Adam optimizer and categorical_crossentropy loss.Streamlit Web App: Interactive interface for users to upload fish images.Real-time Prediction: Outputs the predicted fish class and the associated prediction confidence.

### 🧠 Technologies Used

Category,Tools / Libraries
Language,Python
Deep Learning,"TensorFlow, Keras"
Image Processing,"ImageDataGenerator, PIL (Pillow)"
Web App,Streamlit
Numerical Ops,NumPy

### 📂 Project Structure

The structure is assumed based on the provided code snippets:
```plaintext
Fish_image_classification/
│
├── Data/
│   ├── train/                # Training images organized by sub-folders (classes)
│   └── val/                  # Validation images organized by sub-folders (classes)
│
├── train_and_deploy.py       # (Implied) The main script containing both training and Streamlit code
└── fish_model.keras          # Trained model file (saved after running the script) 

```  
### ⚙️ Installation & Setup
#### 1. Requirements
The project requires the following Python packages. You can install them via pip.

pip install tensorflow keras numpy streamlit Pillow

#### 2. Data Setup
Ensure your fish image data is correctly structured in the specified directories. Each fish species must be in its own sub-folder.

Training Data Path:
C:\Users\Jeeva\Documents\Fishimageclassification\Fish_image_classification\Data\train

Validation Data Path: 
C:\Users\Jeeva\Documents\Fishimageclassification\Fish_image_classification\Data\val

#### 3. Execution
The provided code is logically divided into two parts: training and deployment.

A. Train the Model

The first part of the code trains the model and saves it. Run this portion first:

--> Assuming your training and deployment code is in a single file
--> Run the training section first to generate 'fish_model.keras'
python your_training_script_name.py

This step will create the fish_model.keras file.

B. Run the Streamlit App

Once the model is saved, run the Streamlit application to start the web service:

streamlit run your_deployment_script_name.py

