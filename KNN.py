import numpy as np

class KNN:


    def __init__(self):
        self.x = []


    def PreProcess(self, csvFile, CategoricalIndexList, NumericalIndexList):
        """Converts csv file to a Numpy array to be used in KNN Algorithm.
        Categorical columns, given as a list of indices, are
        one-hot encoded.

        Args:
            csvFile ([Str]):  a path to a csv file.
            indexList ([List]): A list of indices of columns to be one-hot
            encoded.
        """
        # Import data as plain text from file, separate each line on comma, and
        # Strip any leading/trailing spaces or /n's.
        data = [line.strip().split(",") for line in open(csvFile).readlines()]

        # Separate the numerical values from the categorical values (as given
        # in the indexList). Also separate the classifications.
        numerical = []
        categorical = []
        classifications = []

        for row in data:
            classifications += [row[len(row)-1]]


        print (data)

    def Train(self, TrainData, K):
        """Trains a KNN classifier. Last column should be the classifier.

        Args:
            TrainData ([Numpy Array]): A Numpy Array returned after using
            KNN.PreProcess.
            K ([int]): number of nearest neighbors.
        """
        pass

    def Predict(self, TestData, TrainObject):
        """Returns predictions on a test dataset. If actual classifications are
        present they should be in the last column. MSE will be returned if 
        actual classifications exist.

        Args:
            TestData ([Numpy Array]): [description]
            TrainObject ([Numpy Array]): [description]
        """
        pass