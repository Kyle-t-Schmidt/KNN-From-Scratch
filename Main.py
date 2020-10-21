from KNN import knn


trainData = <path to train data>
testData = <path to test data>
categoricalIndices = <list of indices of any non-numerical rows>

test = knn()

test.trainDataProcess(trainData, categoricalIndices)

test.testDataProcess(testData, True)

test.predict(5)
