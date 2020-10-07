import numpy as np

class KNN:


    def __init__(self):
        self.featureMap = {}


    def Train(self, TrainDataCsvFile, IndexList):
        """Converts csv file to a Numpy array to be used in KNN Algorithm.
        Categorical columns, given as a list of indices, are
        one-hot encoded.

        Args:
            TrainCsvFile ([Str]):  a path to a csv file.
            IndexList ([List]): A list of indices of columns to be one-hot
            encoded.
        """
        # Import data as plain text from file, separate each line on comma, and
        # Strip any leading/trailing spaces or /n's.
        data = [line.strip().split(",") for line in open(TrainDataCsvFile)\
            .readlines()]

        # Separate the numerical values from the categorical values (as given
        # in the indexList) and separate the classifications into a third list.
        # Do this by iterating through through each row of the data and storing
        # the vlaues in their respective list.
        numerical = []
        categorical = []
        classifications = []

        for row in data:
            row_num = []
            row_cat = []
            
            for index, value in enumerate(row):
                if index == len(row)-1:
                    classifications.append([value])
                elif index in IndexList:
                    row_cat.append(value)
                else:
                    row_num.append(value)
                
            numerical.append(row_num)
            categorical.append(row_cat)

        
        # Create a feature map for the non-numerical variables so we can 
        # One-hot encode (make binary) these features.
        newData = []

        for row in categorical:
            newRow = []
            for index, value in enumerate(row):
                feature = (index, value)
                if feature not in self.featureMap:
                    self.featureMap[feature] = len(self.featureMap)
                newRow.append(self.featureMap[feature])
            newData.append(newRow)

        # Create a blank numpy array equal to the length of the dataset and the
        # width equal to the number of features. Use the feature map to
        # generate the one-hot encoded data.
        oneHot = np.zeros((len(newData), len(self.featureMap)))

        for index, row in enumerate(newData):
            for feature in row:
                oneHot[index][feature] = 1

        
        print(oneHot)


    def Predict(self, NewData):
        """Returns predictions on a test dataset. If actual classifications are
        present they should be in the last column. MSE will be returned if 
        actual classifications exist.

        Args:
            TestData ([Numpy Array]): [description]
            TrainObject ([Numpy Array]): [description]
        """
        pass