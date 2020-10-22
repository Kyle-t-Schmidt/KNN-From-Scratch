# KNN Classifier

## Purpose
The main purpose of this project was to challenge myself to create a program 
that makes classification predictions using the KNN algorithm. I only used
numpy and base Python in writing this program.

## How to use the knn class
your data must be csv format and the classifications must be in the last
column. If classifications are present in the test data then the program will
print accuracy results.  If your data contains any non-numerical data you will
need to provide a list of indices of the non-numerical data. The program will
one-hot encode the non-numerical columns. The knn clas is dependent on the
dataPreProcess class.

The knn class has 3 methods:

knn().trainDataProcess(TrainDataCsvFile, IndexList)

Use this method to create a feature map and transform the data into the correct
format for the KNN algorithm. Pass your train data and list of non-numerical 
columns indices as parameters. This method does not return anything but after
the method runs you will be able to access the feature map, train data
classifications, non-numerical index list and the processed train data using:

self.featureMap
self.trainClassifications
self.indexList
self.trainData

knn().testDataProcess(TestDataCsvFile, Classifications)

Use this method after the trainDataProcess process to transform your test data
to the correct format for the KNN algorithm. Pass the test dataset as a
parameter and indicate if the test dataset has classifications in the last
column by passing Classifications=True|False (False is the defualt). This
method uses the feature map and indices list procuded by using the
trainDataProcess method. This method does not return anythin but after running 
you can access the test data and test classifications using:

self.newClassifications
self.testData

knn().predict(K)

This method is used after using the trainDataProcess and testDataProcess
methods and predicts the classifications of the test data. If classifications
are present in the test data accuracy results are printed. Pass the number of
nearest neighbors as the only parameter. This method does not return anything
but after running it you will be able to access the predictions using:

self.predictions
