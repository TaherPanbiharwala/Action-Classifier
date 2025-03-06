# Action Classifier using CNN and Transfer Learning

## Overview
This project implements an action classifier using Convolutional Neural Networks (CNN) enhanced by transfer learning with the VGG16 architecture. It classifies human actions into two categories: **Running** and **Walking**. The model is developed using Python, TensorFlow, and Keras, with hyperparameter tuning performed using Keras Tuner.

## Key Features
- **Dataset Management**: Efficient splitting of the dataset into training, validation, and test sets using `train_test_split`.
- **Data Augmentation**: Implemented using `ImageDataGenerator` to increase robustness and reduce overfitting.
- **Transfer Learning**: Leveraged the pre-trained VGG16 model to extract powerful feature representations.
- **Hyperparameter Optimization**: Utilized Keras Tuner's `RandomSearch` for optimal parameter selection.
- **Video and Image Prediction**: Capable of predicting actions from both images and video streams using OpenCV.

## Findings from Training
- **Best Validation Accuracy**: Achieved **83.59%**.
- **Optimal Model Parameters**:
  - Convolution Blocks: **1**
  - Filters: **96**
  - Kernel Size: **3x3**
  - Dense Units: **128**
  - Dropout Rate: **0.2**
  - Optimizer: **RMSProp**

## Performance on Test Data
- **Accuracy**: **86.04%**
- **Loss**: **0.4691**

The results demonstrate the effectiveness of using transfer learning and systematic hyperparameter tuning for image-based action classification tasks.

## Technologies Used
- Python
- TensorFlow & Keras
- OpenCV
- Keras Tuner
- Google Colab

## Usage
- Clone the repository.
- Install required dependencies (`tensorflow`, `keras-tuner`, `opencv-python`, etc.).
- Organize your dataset following the provided structure.
- Train and evaluate your model using the provided scripts.

## Future Improvements
- Extend the dataset with additional action classes.
- Explore more complex models and deeper CNN architectures.
- Integrate real-time video stream prediction for live-action classification.
