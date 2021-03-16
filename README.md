# Multiclass-Image-Classification-using-SVMs
In this implementation a multiclass image classification system is created, using SVMs and a dictionary with the use of Bag of (Visual) Words.

For this implementation openCV is used along with dirent.h which can be found with full documentation and description here: https://github.com/tronkko/dirent
*As image dataset, I am using a subset of Caltech 101 image dataset.
*As a first step I am using the images as a train set to create a vocabulary based on the BOVW model. Number of features is chosen with K-means.
*I am using 1 svm for every different class (10 SVMs in total), setted up for 1 vs all classification
