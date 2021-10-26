from ast import literal_eval
from math import log10, floor
import numpy as np



class data:
    def get(self):
        X = self.getX()
        y = self.getY()

        X = self.roundX(X)
        y = self.roundY(y)

        length = len(X)
        train_amount = length - 10 

        X_train = X[:train_amount]
        X_test = X[train_amount:]
        y_train = X[:train_amount]
        y_test = X[train_amount:]


        return [[np.array(X_train), np.array(y_train)], [np.array(X_test), np.array(y_test)]]

    def round_sig(self,x, sig=5):
        if(x == 0): return 0
        return round(x, sig-int(floor(log10(abs(x))))-1)

    def roundX(self,X):
        newX = []
        for vals in X:
            newVals = []
            for val in vals:
                newVals.append(self.round_sig(val))
            newX.append(newVals)
        return newX

    def roundY(self,y):
        newY = []
        for val in y:
            newY.append(self.round_sig(val))
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