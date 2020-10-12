import numpy as np

class knn:


    def __init__(self):
        self.featureMap = {}
        self.trainClassifications = []
        self.newClassifications = []
        self.indexList = []
        # self.trainData and self.testData also exist but cannot be defined
        # until after the the number of features is known.


    def trainDataProcess(self, TrainDataCsvFile, IndexList):
        """Converts csv file to a Numpy array to be used in KNN Algorithm.
        non-numerical columns, given as a list of indices, are
        one-hot encoded.

        Args:
            TrainCsvFile (Str): Path to training data csv file.
            IndexList (List): A list of indices of columns to be one-hot
            encoded.
        """


        self.indexList = IndexList

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

        for row in data:
            row_num = []
            row_cat = []
            
            for index, value in enumerate(row):
                if index == len(row)-1:
                    self.trainClassifications.append([value])
                elif index in self.indexList:
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

        # Append the numerical data that was not one-hot encoded to each line
        # of the one-hot data. This brings all the data back together.
        numerical = np.array(numerical, dtype=int)
        self.trainData = np.concatenate((numerical,oneHot), axis=1)

        print('Preprocessing complete')


    def testDataProcess(self, TestDataCsvFile, Classifications=False):
        """Converts csv file to a Numpy array to be used in KNN Algorithm.
        Must be in the same format as the train data set. Uses the feature map
        and non-numerical index list from trainDataProcess.

        Args:
            TestDataCsvFile (Str): Path to csv file.
            Classifications (boolean): If the csv file contains 
            classifications, set to true.
        """


        
        # Import data as plain text from file, separate each line on comma,
        #  and strip any leading/trailing spaces or /n's.
        data = [line.strip().split(",") for line in open(TestDataCsvFile)\
            .readlines()]

        # Separate the numerical values from the non-numerical values (as
        # given in the indexList) and separate the classifications into a
        # third list. Do this by iterating through through each row of the
        # data and storing the vlaues in their respective list.
        numerical = []
        categorical = []

        for row in data:
            row_num = []
            row_cat = []
            
            # If the test data contains classifications:
            if Classifications:
                for index, value in enumerate(row):
                    if index == len(row)-1:
                        self.newClassifications.append([value])
                    elif index in self.indexList:
                        row_cat.append(value)
                    else:
                        row_num.append(value)
                    
                numerical.append(row_num)
                categorical.append(row_cat)

            # If the test data does not contain classifications:
            else:
                for index, value in enumerate(row):
                    if index in self.indexList:
                        row_cat.append(value)
                    else:
                        row_num.append(value)
                    
                numerical.append(row_num)
                categorical.append(row_cat)

        
        # use the feature map to One-hot encode (make binary) the test 
        # features.
        newData = []

        for row in categorical:
            newRow = []
            for index, value in enumerate(row):
                feature = (index, value)
                if feature not in self.featureMap:
                    continue
                newRow.append(self.featureMap[feature])
            newData.append(newRow)

        # Create a blank numpy array equal to the length of the dataset and
        #  the width equal to the number of features. Use the feature map
        # to generate the one-hot encoded data.
        oneHot = np.zeros((len(newData), len(self.featureMap)))

        for index, row in enumerate(newData):
            for feature in row:
                oneHot[index][feature] = 1

        # Append the numerical data that was not one-hot encoded to each line
        # of the one-hot data. This brings all the data back together.
        numerical = np.array(numerical, dtype=int)
        self.testData = np.concatenate((numerical,oneHot), axis=1)

        print('Preprocessing complete')
           

    def predict(self, K):
        """Predicts the clasifications of the test dataset using K nearest
        neighbors with euclidean distance. 

        Args:
            K (int): number of nearest neighbors to use for classification 
            prediction
        """

        for i in range(len(self.testData)):
            distances = []
            counter = 0

            for j in range(len(self.trainData)):

                if counter < K:
                    distances.append((j, np.linalg.norm(self.trainData[j]-self.testData[i])))
                    counter += 1
                    print(distances)