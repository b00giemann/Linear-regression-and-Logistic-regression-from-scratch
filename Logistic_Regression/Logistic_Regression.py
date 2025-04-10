import numpy as np

def sigmoid(x):
    return 1/(1+ np.exp(-x))

class logisticRegression():

    def __init__(self,lr = 0.001 , n_iterations = 1000 ):

        self.lr = lr # learning rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self , X , y):

        # rows(data points) , columns(variables, features)
        n_samples , n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            # y = wX + b
            linear_prod = np.dot(X,self.weights) + self.bias 
            predictions = sigmoid(linear_prod)

            dw = (1/n_samples)* np.dot(X.T,(predictions - y))
            db = (1/n_samples)* np.sum(predictions - y)

            self.weights  = self.weights - self.lr * dw
            self.bias  = self.bias - self.lr * db


    def predict(self,X):

        # function using updated weights and biases
        linear_prod = np.dot(X,self.weights) + self.bias 
        y_pred = sigmoid(linear_prod)

        class_pred = [0 if y<=0.5 else 1 for y in y_pred ]
        return class_pred
        
        # clas_pred = []
        # for y in y_pred:
        #     if y <= 0.5:
        #         class_pred.append(0)

        #     else:
        #         class_pred.append(1)





        