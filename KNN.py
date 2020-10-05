import numpy as np

class KNN:


    def __init__(self):


    def PreProcess(self, csvFile, indexList):
        """Converts csv file to a Numpy array to be used in KNN Algorithm.
        Categorical columns, given as a list of indices, are
        one-hot encoded.

        Args:
            csvFile ([Str]):  a path to a csv file.
            indexList ([List]): A list of indices of columns to be one-hot
            encoded.
        """
        pass

    def Train(self, TrainData, K):
        """Trains a KNN classifier.

        Args:
            TrainData ([Numpy Array]): A Numpy Array returned after using
            KNN.PreProcess.
            K ([int]): number of nearest neighbors.
        """
        pass

    def Predict(self, TestData, TrainObject):
        """Returns predictions on a test dataset.

        Args:
            TestData ([Numpy Array]): [description]
            TrainObject ([Numpy Array]): [description]
        """
        pass