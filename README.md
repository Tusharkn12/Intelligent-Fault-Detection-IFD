
# Intelligent Fault Detection (IFD)

The `IFD` class provides a comprehensive framework for integrating Convolutional Neural Networks (CNNs) with various machine learning models like XGBoost, Isolation Forest, and Support Vector Machines (SVM). It is designed to streamline the process of creating, training, and evaluating models for intelligent fault detection.

## Features

- Easily configure and integrate CNN models with other machine learning techniques.
- Automates the creation of data pipelines for efficient training.
- Supports training with advanced algorithms such as XGBoost, Isolation Forest, and SVM.

---

## How to Use

### 1. Initialization
The `IFD` class is initialized with configuration parameters for the model, including input shapes and hyperparameters.

```python
def __init__(self, transformed_img_shape, n_f, T_x, num_pred, batch_size=32, replace_part=True) -> None:
```
- **Parameters**:
  - `transformed_img_shape`: Shape of the transformed image for model input.
  - `n_f`: Number of features in the input data.
  - `T_x`: Time steps for the model.
  - `num_pred`: Number of predictions the model will make.
  - `batch_size`: Batch size for training (default is 32).
  - `replace_part`: Whether to replace parts of the model (default is `True`).

---

### 2. Configuring the Convolutional Layer

```python
def set_conv_layer(self, conv_model, auto_encoder=False) -> None:
```
- Configures the CNN layer.
- **Parameters**:
  - `conv_model`: Predefined convolutional model.
  - `auto_encoder`: Flag to indicate the use of an autoencoder (default is `False`).

---

### 3. Setting Up the Data Generator

```python
def set_Data_generator(self) -> None:
```
- Sets up the data generator for training.
- Ensures the training pipeline is efficiently configured.

---

### 4. Validating Conditions

```python
def conditions_check(self) -> bool:
```
- Validates conditions before training.
- **Returns**: `True` if conditions are satisfied, otherwise `False`.

---

### 5. Generating the Model

```python
def create_model(self):
```
- Generates the Main CNN model structure.

---

### 6. Creating and Training Models
-All of them could only be created after training main model.

#### a. Convolutional + XGBoost

```python
def create_and_train_ConvXGB(self, created_model, XGB_model, X_train, y_train):
```
- Combines a convolutional model with XGBoost for training.
- **Parameters**:
  - `created_model`: The CNN model structure.
  - `XGB_model`: An instance of the XGBoost model.
  - `X_train`: Training data.
  - `y_train`: Training labels.

#### b. Convolutional + Isolation Forest

```python
def create_and_train_ConvISO(self, created_model, ISO_model, X_train):
```
- Combines a convolutional model with Isolation Forest.
- **Parameters**:
  - `created_model`: The CNN model structure.
  - `ISO_model`: An instance of the Isolation Forest model.
  - `X_train`: Training data.

#### c. Convolutional + SVM

```python
def create_and_train_ConvSVM(self, created_model, SVM_model, X_train):
```
- Combines a convolutional model with Support Vector Machine.
- **Parameters**:
  - `created_model`: The CNN model structure.
  - `SVM_model`: An instance of the SVM model.
  - `X_train`: Training data.

---


## Example Usage

Here is an example workflow:

```python
from IFD import IFD

# Initialize the IFD class
ifd = IFD(transformed_img_shape=(64, 64, 3), n_f=64, T_x=10, num_pred=1)

# Set up convolutional layer
ifd.set_conv_layer(conv_model=pretrained_model, auto_encoder=False)

# Set up data generator
ifd.set_Data_generator()

# Validate conditions
if ifd.conditions_check():
    print("Conditions satisfied. Proceeding with training.")

#Creating Main model
main_model = ifd.create_model()

#ConvXGB, ConvISO, ConvSVM could only be created after training main model
# Train with ConvXGB
ifd.create_and_train_ConvXGB(created_model= main_model, XGB_model=xgb_model, X_train=X_train, y_train=y_train)

#Train with ConvISO
ifd.create_and_train_ConvISO(created_model= main_model, XGB_model=xgb_model, X_train=X_train)

#Train with ConvSVM
ifd.create_and_train_ConvSVM(created_model= main_model, XGB_model=xgb_model, X_train=X_train)

```

---
## Dependencies

This project requires the following Python libraries:

- `numpy`
- `matplotlib`
- `tensorflow`
- `pandas`
- `xgboost`
- `umap-learn`
- `seaborn`
- `scikit-learn`
- `keras`

### Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
---

---

## License

This project is licensed under the Apache 2.0 License.

---

Feel free to contribute or raise issues on the [GitHub Repository](#).
