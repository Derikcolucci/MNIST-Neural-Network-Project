MNIST Handwritten Digit Classifier
==================================

This project demonstrates how to classify handwritten digits using the MNIST dataset with both Convolutional Neural Networks (CNNs) and a Dense Neural Network (DNN), implemented in PyTorch and TensorFlow.

Project Structure
-----------------

.
├── mnist_cnn_pytorch.ipynb        # CNN using PyTorch
├── mnist_cnn_tensorflow.ipynb     # CNN using TensorFlow
├── mnist_dnn_tensorflow.ipynb     # DNN using TensorFlow
├── mnist/                         # MNIST dataset (optional, used if loading locally)
│   └── ...                        # Contains training and test data
├── images/                        # Folder with test images (handwritten digits)
│   ├── digit_0.png
│   ├── digit_1.png
│   └── ...                        # Additional test images
└── README.md                      # Project overview

Getting Started
---------------

Requirements:

- Python 3.x
- Jupyter Notebook
- Packages:
  - torch, torchvision (for PyTorch notebook)
  - tensorflow (for TensorFlow notebooks)
  - matplotlib, numpy, PIL for image visualization and preprocessing

You can install all dependencies with:

pip install torch torchvision tensorflow matplotlib numpy pillow

Running the Notebooks:

1. Launch Jupyter:

   jupyter notebook

2. Open one of the following notebooks:
   - mnist_cnn_pytorch.ipynb
   - mnist_cnn_tensorflow.ipynb
   - mnist_dnn_tensorflow.ipynb

3. Run all cells to train the model and evaluate its accuracy.

Testing with Custom Images
--------------------------

To test the trained model on your own handwritten digits:

1. Place your images in the `images/` folder. Each image should:
   - Be grayscale
   - Have a size of 28x28 pixels
   - Have a black background with white digits (inverted if necessary)

2. Use the provided code cells in each notebook to load and predict the digit from your custom image.

Notes
-----

- The `mnist/` directory is optional if you download the dataset automatically via `torchvision.datasets.MNIST` or `tensorflow.keras.datasets.mnist`.
- The CNN architectures significantly outperform the DNN in classification accuracy.

License
-------

This project is open-source and free to use under the MIT License.
