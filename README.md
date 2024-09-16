# Digit Recognition Project

This project implements a simple neural network for recognizing handwritten digits using the MNIST dataset. It includes code for training the model and using the trained model for predictions on new images.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Setup](#setup)
3. [Training the Model](#training-the-model)
4. [Making Predictions](#making-predictions)

## Project Structure

The project consists of two main Python scripts:

- `data_train.py`: Handles data loading, model definition, and training.
- `ia_model.py`: Loads the trained model and makes predictions on new images.

## Setup

1. Ensure you have Python 3.x installed on your system.

2. Install the required dependencies:
   ```
   pip install torch torchvision matplotlib numpy Pillow
   ```

3. Clone or download this project to your local machine.

## Training the Model

1. Navigate to the project directory in your terminal.

2. Run the training script:
   ```
   python data_train.py
   ```

3. The script will download the MNIST dataset, train the model, and save it as `mnist_model.pth`.

4. After training, the script will also save some test images as `mnist_test_image_0.png` through `mnist_test_image_5.png`.

## Making Predictions

1. Ensure you have a trained model file `mnist_model.pth` in your project directory.

2. To make predictions on the test images or your own images:
   - Place your images in the project directory.
   - Open `ia_model.py` and modify the image file names in the main section if needed.

3. Run the prediction script:
   ```
   python ia_model.py
   ```

4. The script will display each image and print the predicted digit.

## Improving the Model

To improve the model's performance on external data:

1. Try data augmentation in the training process.
2. Experiment with more complex model architectures.
3. Increase the number of training epochs.
4. Use regularization techniques to prevent overfitting.

Feel free to modify the code and experiment with different approaches!
