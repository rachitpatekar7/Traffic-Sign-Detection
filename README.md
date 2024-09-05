# Traffic Sign Detection

## Overview
Traffic Sign Detection is a computer vision project aimed at recognizing and classifying traffic signs from images. This project is particularly useful for autonomous driving systems, driver assistance, and road safety applications. The goal is to detect traffic signs such as speed limits, turn directions, and vehicle restrictions using image processing and machine learning techniques.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Usage](#usage)
5. [Model Architecture](#model-architecture)
6. [Results](#results)
7. [Contributors](#contributors)
8. [License](#license)

## Project Structure
```
Traffic-Sign-Detection/
│
├── data/
│   ├── 1.png          # Example traffic sign (left curve warning)
│   ├── 2.jpg          # Example traffic sign (turn left mandatory)
│   ├── 3.png          # Example traffic sign (no trucks allowed)
│   ├── 4.jpg          # Example traffic sign (speed limit 30 km/h)
│   └── Test.csv       # Test dataset containing details for evaluation
│
├── src/
│   └── model-training.ipynb   # Notebook for training and testing the model
│
├── README.md        # Project documentation
└── requirements.txt # Python dependencies
```

## Installation

### Requirements
- Python 3.x
- Jupyter Notebook
- OpenCV
- TensorFlow/PyTorch (for machine learning models)
- Numpy
- Pandas
- Matplotlib
- Scikit-learn

You can install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

## Dataset
The project uses a set of traffic sign images stored in the `data/` folder. The dataset can be expanded by adding more labeled images in appropriate formats. Each image represents a traffic sign with different shapes and designs, including speed limits, turn directions, and vehicle restrictions.

- **1.png**: Left curve warning sign.
- **2.jpg**: Turn left mandatory sign.
- **3.png**: No trucks allowed sign.
- **4.jpg**: Speed limit 30 km/h sign.
- **Test.csv**: A CSV file containing additional test samples.

## Usage

1. Clone the repository:
```bash
git clone https://github.com/rachitpatekar7/Traffic-Sign-Detection.git
cd Traffic-Sign-Detection
```

2. Launch the Google Collab to train and test the model

3. Run the cells in the notebook to process images and train the model for detecting traffic signs.

4. You can modify the images in the `data/` folder to test new signs and evaluate the model.

## Model Architecture

The traffic sign detection model uses a Convolutional Neural Network (CNN) to classify traffic signs. The architecture includes:

- **Input Layer**: Preprocessed traffic sign images.
- **Convolutional Layers**: Detect features like edges, shapes, and patterns of the signs.
- **Pooling Layers**: Down-sample the feature maps.
- **Fully Connected Layers**: Classify the traffic sign based on learned features.

You can refer to the `model-training.ipynb` notebook for more details on the architecture and training process.

## Results

After training the model on the provided dataset, the model is capable of detecting various traffic signs such as:

- Left curve warning
- Turn left mandatory
- No trucks allowed
- Speed limit 30 km/h

You can view more results in the notebook and adjust the model parameters to improve performance.
