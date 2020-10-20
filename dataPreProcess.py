import numpy as np


class dataPreProcess:


    def __init__(self):
        self.featMap = {}
        self.labelEncoded = []

    def featureMap(self, listOfRows):
        """Takes a list of comma separateed value rows and generates a feature
        map. The feature map is used to encode data.

        Args:
            listOfRows (list): comma separated string representation of rows
            that need a feature map for encoding.
        """

        for row in listOfRows:
            newRow = []
            for index, value in enumerate(row):
                feature = (index, value)
                if feature not in self.featMap:
                    self.featMap[feature] = len(self.featMap)
                newRow.append(self.featMap[feature])
            self.labelEncoded.append(newRow)

    def oneHot(self):
        """Converts label encoded data to one-hot encoded data.

        Returns:
            numpy array: a numpy array on the one hot encoded conversion from
            the given label encoded data.
        """

        oneHot = np.zeros((len(self.labelEncoded), len(self.featMap)))

        for index, row in enumerate(self.labelEncoded):
            for feature in row:
                oneHot[index][feature] = 1

        return oneHot
