# KNN Classifier (README STILL IN WORK)

## Purpose
The main purpose of this project was to challenge myself to create a program 
that makes classification predictions using the KNN algorithm. I only used
numpy and base Python in writing this program.

## How to use the program
your data must be csv format and the classifications must be in the last
column. If classifications are present in the test data then the program will
print accuracy results.  If your data contains any non-numerical data you will
need to provide a list of indices of the non-numerical data. The program will
one-hot encode the non-numerical columns.

The knn class has 3 methods:

knn().trainDataProcess(TrainDataCsvFile, IndexList)

This method does not return anything but after the method runs you will be able
to access the feature map, train data classifications, non-numerical index list
and the processed train data using:

self.featureMap
self.trainClassifications
self.indexList
self.trainData