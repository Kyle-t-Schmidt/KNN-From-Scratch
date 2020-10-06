from KNN import KNN


trainData = r'/home/kyle/GitHub/KNN-From-Scratch/income.train.txt.5k'
testData = r'/home/kyle/GitHub/KNN-From-Scratch/income.dev.txt'
categoricalIndices = [1, 2, 3, 4, 5, 6, 8]

test = KNN()

test.PreProcess(trainData, [1,2,3])