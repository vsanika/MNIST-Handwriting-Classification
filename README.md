# MNIST Handwriting Classification
This project implements two popular classification algorithms, Naive Bayes and Logistic Regression, to classify handwritten digits from the MNIST dataset. The goal is to demonstrate how to vectorize image data and how to evaluate the performance of different classifiers using accuracy and confusion matrices.


The notebook loads the MNIST data, preprocesses it by flattening the images into vectors and scaling the pixel values between 0 and 1, trains Naive Bayes and Logistic Regression models using SciKit Learn, evaluates the accuracy of the models on the test set, and displays the confusion matrices for both models on the train and test sets.


## Project structure
MNIST Classification.ipynb: a Jupyter Notebook that contains the code for loading, preprocessing, training, and evaluating the classifiers.

data folder: tar.gz folder containing data used in this notebook


## Results
After running the notebook, you should see the following results:

The accuracy of the Naive Bayes model on the test set is around 83.65%.

The accuracy of the Logistic Regression model on the test set is around 92.23%.

The confusion matrices for Naive Bayes and Logistic Regression on the train and test sets, which show how many instances of each class were correctly and incorrectly classified.


Based on the confusion matrices, you can also see which digits are most often confused by the classifiers. For example, for Logistic Regression, digits 5, 3, and 8 are often misclassified, while for Naive Bayes, digits 4 and 5 are problematic. These errors may occur because certain digits have similar shapes, and the classifiers may have difficulty distinguishing them.
