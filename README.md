# Oil Spill Classification Using Deep Learning

This project focuses on classifying oil spills using a deep learning model built with TensorFlow. The objective is to accurately predict whether an observation represents an oil spill or not, using features from the dataset.

## Overview

- **Language**: Python
- **Libraries**: 
  - [pandas](https://pandas.pydata.org/), 
  - [numpy](https://numpy.org/), 
  - [matplotlib](https://matplotlib.org/), 
  - [seaborn](https://seaborn.pydata.org/), 
  - [scikit-learn](https://scikit-learn.org/stable/), 
  - [tensorflow](https://www.tensorflow.org/)
- **Model**: Fully connected neural network with TensorFlow and Keras.
- **Dataset**: A CSV file containing features and a binary target (oil spill vs non-oil spill).

## Workflow

1. **Data Preprocessing**
   - Data is loaded and split into features (`X`) and the target (`y`).
   - The dataset is split into training and testing sets (80/20 split).
   - Standardization is applied using `StandardScaler` to normalize the input features.

2. **Model Architecture**
   - A Sequential neural network is built with the following layers:
     - Input layer.
     - Two Dense layers with 64 and 32 neurons, using ReLU activation.
     - Dropout layer (0.5) to prevent overfitting.
     - Sigmoid activation in the output layer for binary classification.

3. **Training**
   - The model is compiled with the `Adam` optimizer and `binary_crossentropy` loss function.
   - The network is trained for 50 epochs with a batch size of 32, tracking accuracy on both training and validation sets.

4. **Evaluation**
   - The model’s performance is evaluated using Accuracy, Precision, Recall, and F1-Score.
   - A confusion matrix is generated to visualize true/false positives and negatives.
   - Training history is plotted to show accuracy and loss trends across epochs.

## Results

- **Accuracy**: `99%`
- **Precision**: `1.00`
- **Recall**: `83%`
- **F1 Score**: `91%`

The confusion matrix and training-validation plots help in visualizing the model's performance.

## Conclusion

This project demonstrates an effective deep learning approach to classify oil spills. Future work can focus on hyperparameter tuning and experimenting with different architectures to improve performance.

## References

- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [Keras Documentation](https://keras.io/guides/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/user_guide.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Numpy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/stable/users/index.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)
