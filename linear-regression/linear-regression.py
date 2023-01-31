# Import dependecies
import numpy as np
import numpy.linalg as LA


def read(filename):
    X = []
    y = []
    label = 0

    for i, digit in enumerate(open(filename, "r").read().splitlines()):
        num = digit.split(" ")
        x = 6 - np.array([int(n) for n in num if not n == ""])
        X.append(x) # Normalizing the data
        y.append(label)

        if(i + 1) % 200 == 0:
            label += 1
    
    return np.array(X), np.array(y)

class LinearRegression:
    def __init__(self):
        self.theta = None
    
    def fit(self, X, y):
        alpha = 5
        X_2 = np.dot(X.T, X)
        I = alpha * alpha * np.identity(np.size(X_2, 1))
        #pseudo inverse
        #self.theta = np.dot(LA.pinv(X), y) # Another alternative!
        self.theta = np.dot(LA.pinv(X_2 + I), np.dot(X.T, y)) #When we add bias, matrix become singular!
    
    def predict(self, X):
        return np.dot(X, self.theta).argmax(0) # only works for vectors
    
    def avgcost(self, X, y):
        cost = 0
        
        for i, x in enumerate(X):
            p = np.dot(x, self.theta)
            cost += np.dot((y[i,:]-p).T,(y[i,:]-p))
            
        return cost / y.shape[0]
    
    def percentage(self, X, y):
        
        misclassification = 0
        
        for i, x in enumerate(X):
            if y[i, self.predict(x)] != 1:
                misclassification += 1
        
        return 100*(misclassification/y.shape[0])#for missclassification
    