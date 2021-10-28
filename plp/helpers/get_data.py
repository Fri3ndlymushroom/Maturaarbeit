from ast import literal_eval
from math import log10, floor
import numpy as np



class data:
    def get(self):
        X = self.getX()
        y = self.getY()

        X = self.normX(X)
        y = self.normY(y)

        length = len(X)
        train_amount = length - 10 

        X_train = X[:train_amount]
        X_test = X[train_amount:]
        y_train = y[:train_amount]
        y_test = y[train_amount:]


        return [[np.array(X_train), np.array(y_train)], [np.array(X_test), np.array(y_test)]]

    def round_sig(self,x, sig=5):
        if(x == 0): return 0
        return round(x, sig-int(floor(log10(abs(x))))-1)

    def normX(self,X):
        newX = []

        for val in X:
            newX.append([
                val[0] / 3000,
                val[1] / 3000,
                val[2] / 3000,
                val[3] / 3000,
                val[4] / 10000

            ])


        print(newX[0])


        return newX

    def normY(self,y):
        newY = []
        for val in y:
            newVal = round(val)
            if(newVal>60): newVal = 60
            newY.append(newVal)
        return newY

    def getY(self):
        y = []
        yf = open("plp/data/training_raw/Y.txt", "r")

        for line in yf:
            y.append(literal_eval(line))
        
        return y

    def getX(self):
        X = []

        Xf = open("plp/data/training_raw/X.txt", "r")
        for line in Xf:
            X.append(literal_eval(line))

        return X


data = data()


