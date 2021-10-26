from ast import literal_eval
from math import log10, floor

def getData():
    X = getX()
    y = getY()

    X = roundX(X)
    y = roundY(y)

    return [X, y]

def round_sig(x, sig=5):
    if(x == 0): return 0
    return round(x, sig-int(floor(log10(abs(x))))-1)

def roundX(X):
    newX = []
    for vals in X:
        newVals = []
        for val in vals:
            newVals.append(round_sig(val))
        newX.append(newVals)
    return newX

def roundY(y):
    newY = []
    for val in y:
        newY.append(round_sig(val))
    return newY

def getY():
    y = []
    yf = open("plp/data/training_raw/Y.txt", "r")

    for line in yf:
        y.append(literal_eval(line))
    
    return y

def getX():
    X = []

    Xf = open("plp/data/training_raw/X.txt", "r")
    for line in Xf:
        X.append(literal_eval(line))

    return X



getData()
