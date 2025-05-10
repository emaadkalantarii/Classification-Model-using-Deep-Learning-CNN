# Brain Response Classification: Good vs. Bad Design

This project develops a deep learning image classifier to distinguish between brain responses to "good" and "bad" designs. The model, implemented in PyTorch using a Convolutional Neural Network (CNN), processes brain topomap images to perform this binary classification.

## Dataset

The dataset is expected to be structured as follows, with a root folder named `topomaps` (or as specified in `train.py`):

/topomaps
|_ good/
|_ Good_6s_1.png
|_ Good_6s_2.png
|_ ...
|_ bad/
|_ Bad_6s_1.png
|_ Bad_6s_2.png
|_ ...

Each `.png` file is an image representing a brain topomap.

## Objective

The primary goal is to train an image classifier that can accurately distinguish between two categories of brain responses:
* **bad** (sparse label encoding: `0`)
* **good** (sparse label encoding: `1`)

## Methodology

* **Language:** Python 3
* **Core Library:** PyTorch, Torchvision
* **Model Type:** A Convolutional Neural Network (CNN). The architecture includes multiple convolutional layers with ReLU activations, Max Pooling layers, an Adaptive Average Pooling layer, and fully connected layers with Dropout for regularization, outputting a single value via a Sigmoid activation for binary classification.
* **Preprocessing:**
    * Images are loaded using Pillow and converted to RGB.
    * Images are resized to 128x128 pixels.
    * Images are converted to PyTorch tensors.
    * Tensors are normalized using standard ImageNet mean and standard deviation values.

## Files in the Repository

* `train.py`: Script for training the brain response classification model. It handles data loading, preprocessing, model definition, training loop, validation, and saving the best model checkpoint.
* `eval.py`: Script for evaluating a trained model (`.pth` file) on a new set of images. It loads the model, preprocesses the input images, and predicts labels.
* `requirements.txt`: File listing the necessary Python packages for this project.
* `README.md`: This file.

## Setup & Installation

1.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv brain_env
    source brain_env/bin/activate  # On Windows: brain_env\Scripts\activate
    ```

2.  **Install Dependencies:**
    Navigate to the project directory and install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    This will install libraries such as PyTorch, NumPy, Pillow, Scikit-learn, and Torchvision as specified in `requirements.txt`.

## Usage

### 1. Training the Model
Ensure your dataset is in a directory named `topomaps` in the same directory as `train.py`, or modify the `data_dir` variable within `train.py` if your data is elsewhere.

To train the model, run:
```bash
python train.py
```

This script will:

- Load and preprocess the data.
- Split the data into training and validation sets.
- Train the CNN model.
- Print training and validation loss/accuracy for each epoch.
- Save the model checkpoint with the best validation loss as model.pth in the current directory.


### 2. Evaluating a Trained Model
The eval.py script is designed to load the trained model.pth and predict labels for images in a specified directory.

To run the evaluation using the default test directory (topomaps) and model file (model.pth) as defined in its if __name__ == "__main__": block:

python eval.py

The load_and_predict(directory, model_file) function within eval.py can also be called by other scripts or systems. It takes the path to the image directory (structured like the training data) and the path to the .pth model file.

The function returns a dictionary where keys are absolute file paths of the images and values are the predicted integer labels (0 for "bad", 1 for "good").

## Hyperparameters
Key hyperparameters and settings are defined at the beginning of train.py or within its main() function, including:

data_dir: Path to the dataset directory.
batch_size: Number of samples per training/validation batch.
num_epochs: Maximum number of training epochs.
lr: Initial learning rate for the Adam optimizer.
Image Resize: Images are resized to (128, 128).
Normalization Stats: Standard ImageNet means [0.485, 0.456, 0.406] and stds [0.229, 0.224, 0.225].
Train/Validation Split: test_size=0.3 for initial validation split, then test_size=0.5 of that for test (effectively 70% train, 15% val, 15% test - though the test split is not directly used in the training loop apart from being set aside).


## Results
The model's performance (accuracy, precision, recall, F1-score) will depend on the dataset characteristics, the training process, and hyperparameter tuning. The train.py script includes a validation loop to monitor performance and save the best model based on validation loss. The eval.py script includes an example of how to calculate accuracy on a test set if ground truth labels can be inferred from filenames.

